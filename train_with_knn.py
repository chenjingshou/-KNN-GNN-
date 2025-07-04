"""
改进的KNN训练脚本 - 整合空间关系模型
基于spatial_train.py的改进
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import random
import pandas as pd
import ast
import sys
import numpy as np
import os
import logging
from sklearn.model_selection import KFold

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool

# 导入KNN图构建器
from knn_graph_builder import KNNGraphBuilder, KNNSubgraphBuilder

# 导入特征增强器
try:
    from feature_enhancer import enhance_graph_features
except ImportError:
    print("警告: 无法导入feature_enhancer，将使用基础特征")
    enhance_graph_features = None

# 从你的代码导入必要的函数
try:
    from process_grid_data_new import build_brand_type_maps
except ImportError:
    print("警告: 无法导入build_brand_type_maps，将使用简化版本")
    def build_brand_type_maps(dfs, max_levels, pad_token, unknown_token):
        # 简化的实现
        maps = []
        sizes = []
        for level in range(max_levels):
            level_map = {pad_token: 0, unknown_token: 1}
            maps.append(level_map)
            sizes.append(2)
        return maps, sizes

# 配置
NUM_GLOBAL_NODES = 106
MAX_HIERARCHY_LEVELS = 3
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"

# 改进的KNN配置
KNN_CONFIG = {
    'k_spatial': 10,        # 减少空间KNN的邻居数
    'k_semantic': 5,        # 减少语义相似度的邻居数
    'k_cooccurrence': 8,    # 减少共现的邻居数
    'use_brand_similarity': True,
    'use_cooccurrence': True,
    'edge_weight_type': 'combined',
    'distance_metric': 'euclidean'
}

# 改进的模型配置
MODEL_CONFIG = {
    'gnn_hidden_dim': 128,
    'num_gnn_layers': 3,
    'brand_embedding_dim': 32,
    'mlp_hidden_dim': 256,
    'dropout_rate': 0.3,
    'use_attention_pool': True,
    'use_contrastive': True,
    'temperature': 0.07,
    'gat_heads': 4
}

# 改进的训练配置
TRAIN_CONFIG = {
    'learning_rate': 0.001,
    'num_epochs': 200,
    'batch_size': 128,      # 增大batch size
    'patience': 30,
    'num_warmup_epochs': 10,
    'weight_decay': 0.001,  # 增加正则化
    'min_seq_len': 2,
    'label_smoothing': 0.1,  # 标签平滑
    'gradient_clip': 1.0     # 梯度裁剪
}

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialContextSelector:
    """基于空间和语义的上下文选择器"""
    
    def __init__(self, grid_coords_df, brand_type_encoder=None):
        # 预计算所有网格的中心坐标
        self.grid_centers = {}
        for _, row in grid_coords_df.iterrows():
            grid_id = int(row['grid_id']) - 1  # 0-based
            lon_center = (row['grid_lon_min'] + row['grid_lon_max']) / 2
            lat_center = (row['grid_lat_min'] + row['grid_lat_max']) / 2
            self.grid_centers[grid_id] = np.array([lon_center, lat_center])
        
        # 预计算距离矩阵
        num_grids = len(self.grid_centers)
        self.distance_matrix = np.zeros((num_grids, num_grids))
        for i in range(num_grids):
            for j in range(num_grids):
                if i != j:
                    dist = np.linalg.norm(self.grid_centers[i] - self.grid_centers[j])
                    self.distance_matrix[i, j] = dist
        
        self.brand_type_encoder = brand_type_encoder
    
    def select_context_points(self, grid_list, target_idx, num_context=5, 
                            spatial_weight=0.7, semantic_weight=0.2, random_weight=0.1):
        """选择上下文点 - 主要基于空间距离"""
        if len(grid_list) <= num_context + 1:
            return [i for i in range(len(grid_list)) if i != target_idx]
        
        target_grid = grid_list[target_idx] - 1  # 0-based
        candidate_indices = [i for i in range(len(grid_list)) if i != target_idx]
        
        # 计算每个候选点的得分
        scores = []
        for idx in candidate_indices:
            candidate_grid = grid_list[idx] - 1  # 0-based
            
            # 空间距离得分（距离越近得分越高）
            spatial_dist = self.distance_matrix[target_grid, candidate_grid]
            spatial_score = 1.0 / (1.0 + spatial_dist)  # 距离衰减
            
            # 语义相似度得分
            semantic_score = random.random() * 0.5  # 降低随机性
            
            # 随机得分
            random_score = random.random() * 0.3
            
            # 综合得分
            total_score = (spatial_weight * spatial_score + 
                          semantic_weight * semantic_score + 
                          random_weight * random_score)
            scores.append(total_score)
        
        # 选择得分最高的上下文点
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected = [candidate_indices[i] for i in sorted_indices[:num_context]]
        
        return selected


class AttentionAggregator(nn.Module):
    """注意力聚合模块"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_features, batch, input_mask=None):
        """注意力加权聚合"""
        # 计算注意力权重
        attn_scores = self.attention(node_features)  # [num_nodes, 1]
        
        # 按批次聚合
        batch_size = batch.max().item() + 1
        aggregated = []
        
        for b in range(batch_size):
            mask = (batch == b)
            if input_mask is not None:
                mask = mask & input_mask
            
            if mask.sum() == 0:  # 防止空批次
                aggregated.append(torch.zeros(node_features.shape[1], device=node_features.device))
                continue
            
            batch_features = node_features[mask]  # [num_batch_nodes, feature_dim]
            batch_scores = attn_scores[mask]  # [num_batch_nodes, 1]
            
            # Softmax归一化
            batch_weights = F.softmax(batch_scores, dim=0)  # [num_batch_nodes, 1]
            
            # 加权求和
            weighted_features = batch_features * batch_weights
            aggregated_feature = weighted_features.sum(dim=0)  # [feature_dim]
            aggregated.append(aggregated_feature)
        
        return torch.stack(aggregated)  # [batch_size, feature_dim]


class ImprovedSpatialGNN(nn.Module):
    """改进的空间GNN模型"""
    
    def __init__(
        self,
        num_global_nodes,
        node_feature_dim,
        gnn_hidden_dim=128,
        num_gnn_layers=3,
        brand_vocab_sizes=None,
        brand_embedding_dim=32,
        mlp_hidden_dim=256,
        dropout_rate=0.3,
        use_attention_pool=True,
        use_contrastive=True,
        temperature=0.1,
        gat_heads=4
    ):
        super().__init__()
        
        self.num_global_nodes = num_global_nodes
        self.use_attention_pool = use_attention_pool
        self.use_contrastive = use_contrastive
        self.temperature = temperature
        
        # GNN层（使用更深的网络）
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        in_dim = node_feature_dim
        for i in range(num_gnn_layers):
            self.gnn_layers.append(GATConv(in_dim, gnn_hidden_dim, heads=gat_heads, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(gnn_hidden_dim))
            in_dim = gnn_hidden_dim
        
        # 品牌嵌入
        self.brand_embeddings = nn.ModuleList()
        total_brand_dim = 0
        if brand_vocab_sizes:
            for vocab_size in brand_vocab_sizes:
                self.brand_embeddings.append(nn.Embedding(vocab_size, brand_embedding_dim))
                total_brand_dim += brand_embedding_dim
        
        # 注意力聚合
        if use_attention_pool:
            self.attention_agg = AttentionAggregator(gnn_hidden_dim, mlp_hidden_dim)
        
        # 全局节点嵌入（学习每个网格的偏置）
        self.global_node_embedding = nn.Embedding(num_global_nodes, 64)
        
        # 特征组合层
        combined_dim = gnn_hidden_dim + total_brand_dim
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.BatchNorm1d(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_global_nodes)
        )
        
        # 对比学习投影头
        if use_contrastive:
            self.projector = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, batch, return_embeddings=False):
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        # GNN前向传播（带残差连接）
        h = x
        for i, (gnn, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h_new = gnn(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            
            # 残差连接（如果维度匹配）
            if i > 0 and h.shape[1] == h_new.shape[1]:
                h = h + h_new
            else:
                h = h_new
            
            h = self.dropout(h)
        
        # 图聚合
        if self.use_attention_pool:
            # 使用注意力聚合
            graph_embed = self.attention_agg(h, batch_idx)
        else:
            # 使用mean和max pooling的组合
            graph_embed_mean = global_mean_pool(h, batch_idx)
            graph_embed_max = global_max_pool(h, batch_idx)
            graph_embed = (graph_embed_mean + graph_embed_max) / 2
        
        # 品牌嵌入
        brand_embeds = []
        for level_idx, embed_layer in enumerate(self.brand_embeddings):
            level_attr = f'brand_type_level_{level_idx}'
            if hasattr(batch, level_attr):
                brand_indices = getattr(batch, level_attr)
                brand_embeds.append(embed_layer(brand_indices))
        
        if brand_embeds:
            brand_embed = torch.cat(brand_embeds, dim=1)
            combined = torch.cat([graph_embed, brand_embed], dim=1)
        else:
            combined = graph_embed
        
        # 返回嵌入（用于对比学习）
        if return_embeddings:
            embeddings = self.projector(combined) if self.use_contrastive else combined
            return embeddings
        
        # 预测
        logits = self.predictor(combined)
        
        # 添加全局节点嵌入的偏置
        node_bias = self.global_node_embedding.weight.mean(dim=1)  # [num_nodes]
        logits = logits + 0.1 * node_bias.unsqueeze(0)  # 缩放偏置的影响
        
        return logits
    
    def compute_contrastive_loss(self, embeddings, labels, temperature=None):
        """计算对比损失"""
        if temperature is None:
            temperature = self.temperature
        
        # 归一化嵌入
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # 创建标签掩码（相同标签的为正样本）
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 排除对角线（自己与自己）
        mask.fill_diagonal_(0)
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只计算正样本的损失
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ImprovedKNNGraphDataset(Dataset):
    """改进的使用KNN子图的数据集"""
    
    def __init__(
        self, 
        brand_df,
        global_graph,
        knn_subgraph_builder,
        brand_type_maps,
        brand_vocab_sizes,
        grid_coords_df,
        spatial_context_selector,
        min_seq_len=2,
        subsampling_ratio=1.0,
        max_context_points=5,
        k_hops=1,
        max_subgraph_size=20
    ):
        self.global_graph = global_graph
        self.knn_builder = knn_subgraph_builder
        self.brand_type_maps = brand_type_maps
        self.brand_vocab_sizes = brand_vocab_sizes
        self.spatial_selector = spatial_context_selector
        self.min_seq_len = min_seq_len
        self.max_context_points = max_context_points
        self.k_hops = k_hops
        self.max_subgraph_size = max_subgraph_size
        
        # 处理序列数据
        self.samples = []
        
        for _, row in brand_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                if len(grid_list) < min_seq_len:
                    continue
                
                # 处理品牌类型
                brand_types = self._process_brand_type(row['brand_type'])
                
                # 为序列中每个点创建样本
                for target_idx in range(len(grid_list)):
                    # 使用空间上下文选择器
                    context_indices = self.spatial_selector.select_context_points(
                        grid_list, target_idx, self.max_context_points
                    )
                    
                    if len(context_indices) == 0:
                        continue
                    
                    # 构建输入序列
                    input_grids = [grid_list[i] for i in context_indices]
                    target_grid = grid_list[target_idx]
                    
                    # 构建KNN子图
                    input_grid_ids = [g - 1 for g in input_grids]  # 转为0-based
                    subgraph = self.knn_builder.build_instance_subgraph(
                        input_grid_ids=input_grid_ids,
                        global_graph=self.global_graph,
                        k_hops=self.k_hops,
                        max_subgraph_size=self.max_subgraph_size
                    )
                    
                    # 确保子图有正确的属性
                    if not hasattr(subgraph, 'x') or subgraph.x is None:
                        subgraph.x = torch.zeros(subgraph.num_nodes, 1, dtype=torch.float)
                    
                    # 添加目标标签
                    subgraph.y = torch.tensor([target_grid - 1], dtype=torch.long)
                    
                    # 添加品牌类型（分别存储每个层级）
                    for level_idx in range(len(brand_types)):
                        setattr(subgraph, f'brand_type_level_{level_idx}', 
                               torch.tensor([brand_types[level_idx]], dtype=torch.long))
                    
                    # 删除可能导致问题的属性
                    if hasattr(subgraph, 'node_mapping'):
                        delattr(subgraph, 'node_mapping')
                    
                    self.samples.append(subgraph)
                    
            except Exception as e:
                logger.warning(f"处理样本时出错: {e}")
                continue
        
        # 子采样
        if subsampling_ratio < 1.0:
            n_samples = int(len(self.samples) * subsampling_ratio)
            self.samples = random.sample(self.samples, n_samples)
        
        logger.info(f"创建了 {len(self.samples)} 个训练样本")
    
    def _process_brand_type(self, brand_type_str):
        """处理品牌类型字符串"""
        try:
            levels = str(brand_type_str).split(';')
            type_indices = []
            
            for level_idx in range(MAX_HIERARCHY_LEVELS):
                if level_idx < len(levels):
                    type_str = levels[level_idx].strip()
                    if type_str in self.brand_type_maps[level_idx]:
                        type_idx = self.brand_type_maps[level_idx][type_str]
                    else:
                        type_idx = self.brand_type_maps[level_idx].get(UNKNOWN_TOKEN, 1)
                else:
                    type_idx = self.brand_type_maps[level_idx].get(PAD_TOKEN, 0)
                    
                type_indices.append(type_idx)
                
            return torch.tensor(type_indices, dtype=torch.long)
        except:
            return torch.zeros(MAX_HIERARCHY_LEVELS, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def train_with_improvements(model, train_loader, optimizer, device, 
                           use_contrastive=True, contrastive_weight=0.1):
    """改进的训练函数"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_cont_loss = 0
    total_samples = 0
    
    # 使用标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG['label_smoothing'])
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(batch)
        targets = batch.y
        
        # 交叉熵损失
        ce_loss = criterion(logits, targets)
        
        loss = ce_loss
        
        # 对比损失
        if use_contrastive and model.use_contrastive:
            embeddings = model(batch, return_embeddings=True)
            cont_loss = model.compute_contrastive_loss(embeddings, targets)
            loss = ce_loss + contrastive_weight * cont_loss
            total_cont_loss += cont_loss.item() * batch.num_graphs
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAIN_CONFIG['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_ce_loss += ce_loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
    
    results = {
        'total_loss': total_loss / total_samples,
        'ce_loss': total_ce_loss / total_samples,
    }
    
    if use_contrastive and model.use_contrastive:
        results['cont_loss'] = total_cont_loss / total_samples
    
    return results


def evaluate(model: nn.Module, 
            loader, 
            device: torch.device, 
            k_values: list = [1, 5, 10]):
    """评估模型"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    all_ranks = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            if batch.num_graphs == 0:
                continue
            
            # 预测
            logits = model(batch)
            targets = batch.y
            loss = criterion(logits, targets)
            
            # 计算排名
            probs = F.softmax(logits, dim=1)
            sorted_indices = torch.argsort(probs, dim=1, descending=True)
            
            for i in range(targets.size(0)):
                target = targets[i].item()
                rank = (sorted_indices[i] == target).nonzero(as_tuple=True)[0].item() + 1
                all_ranks.append(rank)
                
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
    
    # 计算指标
    all_ranks = np.array(all_ranks)
    mrr = np.mean(1.0 / all_ranks) if len(all_ranks) > 0 else 0
    
    results = {'loss': total_loss / total_samples if total_samples > 0 else 0, 'mrr': mrr}
    
    for k in k_values:
        hr_k = np.mean(all_ranks <= k) if len(all_ranks) > 0 else 0
        results[f'hr@{k}'] = hr_k
        
    return results


def main():
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
    from torch.utils.data import Dataset, DataLoader
    """主训练函数"""
    # 删除旧的缓存文件
    cache_file = 'graph_data_knn_enhanced.pt'
    if os.path.exists(cache_file):
        logger.info(f"删除旧的缓存文件: {cache_file}")
        os.remove(cache_file)
    
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 读取数据
    logger.info("读取数据...")
    train_df = pd.read_csv('train_data(1).csv')
    test_df = pd.read_csv('test_data(1).csv')
    grid_coords = pd.read_csv('grid_coordinates(1).csv')
    
    # 数据统计
    logger.info(f"数据统计:")
    logger.info(f"  训练集大小: {len(train_df)}")
    logger.info(f"  测试集大小: {len(test_df)}")
    logger.info(f"  网格数量: {len(grid_coords)}")
    
    # 序列长度分析
    seq_lengths = []
    for _, row in train_df.iterrows():
        try:
            grid_list = ast.literal_eval(row['grid_id_list'])
            seq_lengths.append(len(grid_list))
        except:
            pass
    
    if seq_lengths:
        logger.info(f"  序列长度 - 平均: {np.mean(seq_lengths):.2f}, "
                   f"最小: {np.min(seq_lengths)}, 最大: {np.max(seq_lengths)}, "
                   f"中位数: {np.median(seq_lengths):.0f}")
    
    # 构建品牌类型映射
    logger.info("构建品牌类型映射...")
    brand_type_maps, brand_vocab_sizes = build_brand_type_maps(
        [train_df, test_df], MAX_HIERARCHY_LEVELS, PAD_TOKEN, UNKNOWN_TOKEN
    )
    
    # 构建KNN全局图
    logger.info("构建KNN全局图...")
    knn_builder = KNNGraphBuilder(**KNN_CONFIG)
    global_graph = knn_builder.build_combined_graph(
        grid_coords=grid_coords,
        train_df=train_df,
        brand_type_maps=brand_type_maps
    )
    
    # 增强节点特征
    if enhance_graph_features is not None:
        logger.info("增强节点特征...")
        global_graph = enhance_graph_features(
            global_graph, 
            train_df, 
            grid_coords,
            brand_type_maps
        )
        logger.info(f"特征维度增强至: {global_graph.x.shape}")
    
    # 如果有预训练的特征，可以加载并合并
    if os.path.exists('graph_data_tencent_rich_ip_features.pt'):
        logger.info("尝试加载预训练特征...")
        try:
            pretrained_graph = torch.load('graph_data_tencent_rich_ip_features.pt', weights_only=False)
            if hasattr(pretrained_graph, 'x'):
                logger.info(f"预训练特征维度: {pretrained_graph.x.shape}")
                if global_graph.x.shape[0] == pretrained_graph.x.shape[0]:
                    global_graph.x = torch.cat([global_graph.x, pretrained_graph.x], dim=1)
                    logger.info(f"合并后特征维度: {global_graph.x.shape}")
        except Exception as e:
            logger.warning(f"加载预训练特征失败: {e}")
    
    # 保存构建好的图
    logger.info(f"保存增强后的图到: {cache_file}")
    torch.save(global_graph, cache_file)
    
    # 创建空间上下文选择器
    spatial_selector = SpatialContextSelector(grid_coords)
    
    # 创建子图构建器
    subgraph_builder = KNNSubgraphBuilder(**KNN_CONFIG)
    
    # 划分训练/验证集
    train_size = int(0.8 * len(train_df))
    train_subset = train_df.iloc[:train_size]
    val_subset = train_df.iloc[train_size:]
    
    # 创建改进的数据集
    logger.info("创建数据集...")
    train_dataset = ImprovedKNNGraphDataset(
        train_subset, global_graph, subgraph_builder,
        brand_type_maps, brand_vocab_sizes,
        grid_coords, spatial_selector,
        min_seq_len=TRAIN_CONFIG['min_seq_len'],
        max_context_points=5,
        k_hops=1,
        max_subgraph_size=20
    )
    
    val_dataset = ImprovedKNNGraphDataset(
        val_subset, global_graph, subgraph_builder,
        brand_type_maps, brand_vocab_sizes,
        grid_coords, spatial_selector,
        min_seq_len=TRAIN_CONFIG['min_seq_len'],
        max_context_points=5,
        k_hops=1,
        max_subgraph_size=20
    )
    
    test_dataset = ImprovedKNNGraphDataset(
        test_df, global_graph, subgraph_builder,
        brand_type_maps, brand_vocab_sizes,
        grid_coords, spatial_selector,
        min_seq_len=TRAIN_CONFIG['min_seq_len'],
        max_context_points=5,
        k_hops=1,
        max_subgraph_size=20
    )
    
    logger.info(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
    
    # 创建数据加载器
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    train_loader = PyGDataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False
    )
    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False
    )
    
    # 创建改进的模型
    logger.info("创建模型...")
    model = ImprovedSpatialGNN(
        num_global_nodes=NUM_GLOBAL_NODES,
        node_feature_dim=global_graph.x.shape[1],
        brand_vocab_sizes=brand_vocab_sizes,
        **MODEL_CONFIG
    ).to(device)
    
    # 优化器 - 使用AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # 学习率调度器 - 使用余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['num_epochs'])
    
    # 训练循环
    logger.info("开始训练...")
    best_val_mrr = 0
    patience_counter = 0
    
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # 训练
        train_results = train_with_improvements(
            model, train_loader, optimizer, device,
            use_contrastive=MODEL_CONFIG['use_contrastive'],
            contrastive_weight=0.1
        )
        
        # 验证
        val_results = evaluate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 日志
        log_str = f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}: "
        log_str += f"Train Loss: {train_results['total_loss']:.4f} "
        log_str += f"(CE: {train_results['ce_loss']:.4f}"
        if 'cont_loss' in train_results:
            log_str += f", Cont: {train_results['cont_loss']:.4f}"
        log_str += f"), Val Loss: {val_results['loss']:.4f}, "
        log_str += f"Val MRR: {val_results['mrr']:.4f}, "
        log_str += f"Val HR@1: {val_results.get('hr@1', 0):.4f}, "
        log_str += f"Val HR@5: {val_results.get('hr@5', 0):.4f}, "
        log_str += f"Val HR@10: {val_results['hr@10']:.4f}"
        logger.info(log_str)
        
        # 早停
        if val_results['mrr'] > best_val_mrr:
            best_val_mrr = val_results['mrr']
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mrr': best_val_mrr,
                'config': {**MODEL_CONFIG, **TRAIN_CONFIG, **KNN_CONFIG}
            }, 'best_model_improved_knn.pt')
            
            torch.save(model.state_dict(), 'best_model_improved_knn_state.pt')
            
            logger.info(f"保存最佳模型，MRR: {best_val_mrr:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= TRAIN_CONFIG['patience']:
            logger.info("早停触发")
            break
    
    # 加载最佳模型并在测试集上评估
    logger.info("在测试集上评估...")
    
    import torch.serialization
    from numpy.core.multiarray import scalar
    
    try:
        checkpoint = torch.load('best_model_improved_knn.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("成功加载最佳模型")
    except Exception as e:
        logger.warning(f"加载失败: {e}")
        try:
            # 备用方案
            model.load_state_dict(torch.load('best_model_improved_knn_state.pt'))
            logger.info("使用备用方案加载模型")
        except Exception as e2:
            logger.error(f"备用方案也失败: {e2}")
    
    test_results = evaluate(model, test_loader, device)
    logger.info(
        f"测试集结果: "
        f"MRR: {test_results['mrr']:.4f}, "
        f"HR@1: {test_results['hr@1']:.4f}, "
        f"HR@5: {test_results['hr@5']:.4f}, "
        f"HR@10: {test_results['hr@10']:.4f}"
    )


if __name__ == "__main__":
    main()