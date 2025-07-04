import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GATConv
import seaborn as sns
import os
import logging
from sklearn.neighbors import KDTree
import matplotlib

# 修复中文字体显示问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 额外的字体设置函数
def set_plot_font(font_size=10):
    """设置支持中文的字体"""
    try:
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['font.size'] = font_size
        print("使用 SimHei 字体")
    except:
        try:
            plt.rcParams['font.family'] = 'Microsoft YaHei'
            plt.rcParams['font.size'] = font_size
            print("使用 Microsoft YaHei 字体")
        except:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            print("使用 Arial Unicode MS 字体")
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('visualization')

# 常量配置
NUM_GLOBAL_NODES = 106
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
MAX_HIERARCHY_LEVELS = 3

class AttentionAggregator(nn.Module):
    """注意力聚合模块（与训练脚本一致）"""
    
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
    """完整模型定义（与训练脚本一致）"""
    
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
        
        # GNN层 - 使用与训练脚本完全相同的GATConv
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        in_dim = node_feature_dim
        for i in range(num_gnn_layers):
            self.gnn_layers.append(GATConv(
                in_dim, 
                gnn_hidden_dim, 
                heads=gat_heads, 
                concat=False,  # 与训练脚本一致
                dropout=dropout_rate,
                add_self_loops=True
            ))
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
        
        # 全局节点嵌入
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
        # 获取图数据
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        # GNN前向传播
        h = x
        for i, (gnn, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h = gnn(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        # 图聚合
        if self.use_attention_pool:
            graph_embed = self.attention_agg(h, batch_idx)
        else:
            graph_embed_mean = h.mean(dim=0).unsqueeze(0)
            graph_embed = graph_embed_mean
        
        # 品牌嵌入 - 在实际可视化中可能不需要
        # 但为了匹配模型结构，保留空列表
        brand_embeds = []
        combined = graph_embed  # 简化：仅使用图嵌入
        
        # 返回嵌入
        if return_embeddings:
            return combined
        
        # 预测
        logits = self.predictor(combined)
        
        return logits

class Visualizer:
    def __init__(self):
        self.grid_coords_df = None
        self.global_graph = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 使用训练脚本中的品牌词汇大小
        self.brand_vocab_sizes = [13, 62, 136]  # 从您的训练输出中提取
    
    def load_data(self):
        """加载所有必要的数据"""
        logger.info("加载数据...")
        
        # 1. 网格坐标数据
        coord_files = ['grid_coordinates.csv', 'grid_coordinates(1).csv']
        found = False
        for file in coord_files:
            if os.path.exists(file):
                self.grid_coords_df = pd.read_csv(file)
                logger.info(f"加载网格坐标数据: {file}，共有 {len(self.grid_coords_df)} 个网格")
                found = True
                break
        
        if not found:
            logger.error("找不到网格坐标文件")
            return False
        
        # 2. 加载全局图
        if os.path.exists('graph_data_knn_enhanced.pt'):
            try:
                self.global_graph = torch.load(
                    'graph_data_knn_enhanced.pt', 
                    map_location='cpu', 
                    weights_only=False
                )
                logger.info(f"加载全局图: 节点数={self.global_graph.num_nodes}, 特征维度={self.global_graph.x.shape[1]}")
            except Exception as e:
                logger.error(f"加载全局图失败: {e}")
                return False
        else:
            logger.error("找不到全局图文件 'graph_data_knn_enhanced.pt'")
            return False
        
        # 3. 加载模型（使用完整模型定义）
        model_files = ['best_model_improved_knn_state.pt']
        found_model = False
        
        for file in model_files:
            if os.path.exists(file):
                try:
                    # 使用完整模型定义
                    self.model = ImprovedSpatialGNN(
                        num_global_nodes=NUM_GLOBAL_NODES,
                        node_feature_dim=self.global_graph.x.shape[1],
                        brand_vocab_sizes=self.brand_vocab_sizes,
                        gnn_hidden_dim=128,
                        num_gnn_layers=3,
                        brand_embedding_dim=32,
                        mlp_hidden_dim=256,
                        dropout_rate=0.3,
                        use_attention_pool=True,
                        use_contrastive=True,
                        temperature=0.07,
                        gat_heads=4
                    )
                    
                    # 加载模型状态
                    self.model.load_state_dict(torch.load(file, map_location='cpu', weights_only=False))
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"成功加载模型: {file}")
                    found_model = True
                    break
                except Exception as e:
                    logger.error(f"加载模型失败 ({file}): {e}")
        
        if not found_model:
            logger.error("找不到或无法加载模型文件")
            return False
        
        return True
    
    def visualize_global_knn_graph(self):
        """可视化全局KNN图"""
        if self.global_graph is None or self.grid_coords_df is None:
            logger.error("无法可视化，必要数据未加载")
            return
        
        logger.info("可视化全局KNN图...")
        set_plot_font(12)
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 提取节点位置（0-based ID）
        node_positions = {}
        for _, row in self.grid_coords_df.iterrows():
            grid_id = int(row['grid_id']) - 1
            lon_center = (row['grid_lon_min'] + row['grid_lon_max']) / 2
            lat_center = (row['grid_lat_min'] + row['grid_lat_max']) / 2
            node_positions[grid_id] = (lon_center, lat_center)
            G.add_node(grid_id, pos=(lon_center, lat_center))
        
        # 添加边
        edge_index = self.global_graph.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            if src < len(node_positions) and dst < len(node_positions):
                G.add_edge(src, dst)
        
        # 计算节点度中心性
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [300 + 1500 * (degrees[n] / max_degree) for n in G.nodes()]
        
        plt.figure(figsize=(15, 12))
        
        # 绘制节点
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=node_sizes, 
            node_color='skyblue',
            alpha=0.8
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            G, pos, 
            edge_color='gray', 
            alpha=0.3,
            width=1.0  # 调细线宽
        )
        
        # 标注高连接度节点
        hub_nodes = [n for n, d in degrees.items() if d > max_degree * 0.7]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=hub_nodes,
            node_size=[node_sizes[i] for i in hub_nodes],
            node_color='orange',
            alpha=0.8
        )
        
        # 添加标题
        plt.title(f'全局KNN图 (节点数: {len(G.nodes)}, 边数: {len(G.edges)})\n橙色节点表示高连接度中心')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.grid(True, alpha=0.2)
        
        # 保存图像
        plt.savefig('global_knn_graph.png', bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("全局KNN图已保存为 'global_knn_graph.png'")
    
    def visualize_training_subgraphs(self):
        """可视化训练子图示例"""
        if self.global_graph is None or self.grid_coords_df is None:
            logger.error("无法可视化训练子图，数据未完全加载")
            return
        
        logger.info("可视化训练子图示例...")
        set_plot_font(10)
        
        # 创建3个示例网格序列
        example_sequences = [
            [3, 5, 7, 10, 15],  # 小范围序列
            [25, 30, 35, 40, 45, 50],  # 中等范围序列
            [60, 65, 70, 75, 80, 85, 90]  # 大范围序列
        ]
        
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, grid_ids in enumerate(example_sequences):
            # 获取节点位置
            positions = []
            node_positions = {}
            
            for grid_id in grid_ids:
                row = self.grid_coords_df[self.grid_coords_df['grid_id'] == grid_id].iloc[0]
                lon = (row['grid_lon_min'] + row['grid_lon_max']) / 2
                lat = (row['grid_lat_min'] + row['grid_lat_max']) / 2
                positions.append([lon, lat])
                node_positions[grid_id] = (lon, lat)
            
            positions = np.array(positions)
            
            # 使用KDTree查找邻居
            kdtree = KDTree(positions)
            neighbors = kdtree.query(positions, k=min(3, len(positions)), return_distance=False)
            
            # 创建边列表
            edges = []
            for i_idx, nbrs in enumerate(neighbors):
                for nbr_idx in nbrs:
                    if i_idx != nbr_idx:  # 避免自环
                        edges.append([i_idx, nbr_idx])
            
            # 创建图结构
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            x = torch.zeros(len(grid_ids), 1)  # 简单特征
            
            graph = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([0], dtype=torch.long),  # 目标节点索引
                num_nodes=len(grid_ids)
            )
            
            # 转换为NetworkX图
            G = to_networkx(graph, to_undirected=True)
            
            # 直接使用经纬度位置
            pos = {i: positions[i] for i in range(len(grid_ids))}
            
            # 获取目标节点（索引0）
            target_idx = 0
            degrees = dict(G.degree())
            
            # 绘制节点和边
            node_colors = ['red' if n == target_idx else 'skyblue' for n in G.nodes()]
            node_sizes = [500 + 1000 * degrees[n] for n in G.nodes()]
            
            nx.draw_networkx_nodes(
                G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                ax=axs[i]
            )
            nx.draw_networkx_edges(
                G, pos, 
                edge_color='gray',
                alpha=0.6,
                width=1.5,
                ax=axs[i]
            )
            
            # 添加标签
            for node_idx, (lon, lat) in pos.items():
                axs[i].text(lon, lat, f"{grid_ids[node_idx]}", 
                            fontsize=10, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))
            
            # 添加标题
            axs[i].set_title(f'训练子图 {i+1}\n节点数: {graph.num_nodes}, 目标节点: {grid_ids[target_idx]}')
            axs[i].set_xlabel('经度')
            axs[i].set_ylabel('纬度')
            
            # 添加网格线
            axs[i].grid(True, alpha=0.1)
            
            # 添加图例
            axs[i].scatter([], [], c='red', s=100, label='目标节点')
            axs[i].scatter([],
                         [], c='skyblue', s=100, label='上下文节点')
            axs[i].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('training_subgraphs.png', dpi=300)
        plt.close()
        logger.info("训练子图已保存为 'training_subgraphs.png'")
    
    def visualize_training_metrics(self):
        """可视化训练过程中的各项指标变化"""
        set_plot_font(10)
        
        if not os.path.exists('training_log.csv'):
            logger.warning("找不到训练日志文件 'training_log.csv'")
            # 使用测试结果作为替代
            self._create_performance_summary()
            return
        
        logger.info("可视化训练指标...")
        
        try:
            # 读取日志数据
            df = pd.read_csv('training_log.csv')
            
            plt.figure(figsize=(15, 10))
            
            # 绘制损失曲线
            plt.subplot(2, 2, 1)
            sns.lineplot(data=df, x='epoch', y='train_loss', label='训练损失')
            if 'val_loss' in df.columns:
                sns.lineplot(data=df, x='epoch', y='val_loss', label='验证损失')
            plt.title('损失变化曲线')
            plt.ylabel('损失值')
            plt.legend()
            
            # 绘制MRR指标
            if 'val_mrr' in df.columns:
                plt.subplot(2, 2, 2)
                sns.lineplot(data=df, x='epoch', y='val_mrr')
                plt.title('验证MRR变化')
                plt.ylabel('MRR')
            
            # 绘制命中率指标
            if 'val_hr@1' in df.columns:
                plt.subplot(2, 2, 3)
                sns.lineplot(data=df, x='epoch', y='val_hr@1', label='HR@1')
                sns.lineplot(data=df, x='epoch', y='val_hr@5', label='HR@5')
                sns.lineplot(data=df, x='epoch', y='val_hr@10', label='HR@10')
                plt.title('命中率变化')
                plt.ylabel('命中率')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_metrics.png', dpi=300)
            plt.close()
            logger.info("训练指标图已保存为 'training_metrics.png'")
        except Exception as e:
            logger.error(f"处理训练日志出错: {e}")
    
    def _create_performance_summary(self):
        """创建性能摘要（当找不到训练日志时）"""
        logger.info("创建性能摘要...")
        set_plot_font(12)
        
        # 使用您的测试结果
        metrics = {
            'MRR': 0.2778,
            'HR@1': 0.1446,
            'HR@5': 0.4085,
            'HR@10': 0.5749
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 创建条形图
        bars = ax.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 设置标题和标签
        ax.set_title('模型测试性能')
        ax.set_ylabel('得分')
        ax.set_ylim(0, 0.7)
        
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('test_performance_summary.png', dpi=300)
        plt.close()
        logger.info("性能摘要已保存为 'test_performance_summary.png'")
    
    def visualize_attention_weights(self):
        """可视化注意力权重如何影响节点重要性（简化版）"""
        set_plot_font(12)
        
        # 如果模型加载成功，尝试计算真实注意力权重
        if self.model and self.global_graph:
            try:
                logger.info("尝试计算真实注意力权重...")
                # 使用全局图
                graph = self.global_graph.clone().to(self.device)
                graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
                
                # 获取图嵌入
                with torch.no_grad():
                    graph_embed = self.model(graph, return_embeddings=True)
                
                # 计算注意力得分
                attn_scores = self.model.attention_agg.attention(graph_embed).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=0).cpu().numpy()
                logger.info("成功计算真实注意力权重")
            except Exception as e:
                logger.error(f"计算注意力权重失败: {e}")
                # 使用随机权重作为备用
                attn_weights = np.random.rand(self.global_graph.num_nodes)
                attn_weights = attn_weights / attn_weights.sum()
                logger.info("使用随机权重作为替代")
        else:
            logger.warning("模型未加载或全局图不可用，使用随机注意力权重")
            # 使用随机权重作为演示
            attn_weights = np.random.rand(self.global_graph.num_nodes)
            attn_weights = attn_weights / attn_weights.sum()
        
        # 获取节点位置
        node_positions = []
        for i in range(self.global_graph.num_nodes):
            grid_id = i + 1  # 全局图中的索引对应网格ID
            row = self.grid_coords_df[self.grid_coords_df['grid_id'] == grid_id].iloc[0]
            lon_center = (row['grid_lon_min'] + row['grid_lon_max']) / 2
            lat_center = (row['grid_lat_min'] + row['grid_lat_max']) / 2
            node_positions.append((lon_center, lat_center))
        
        # 绘制所有节点
        positions = np.array(node_positions)
        
        plt.figure(figsize=(14, 12))
        
        # 绘制节点（按注意力权重）
        scatter = plt.scatter(
            positions[:, 0], positions[:, 1],
            s=100 + 900 * attn_weights,
            c=attn_weights,
            cmap='coolwarm',
            alpha=0.8
        )
        
        # 添加标签
        for i, (lon, lat) in enumerate(positions):
            plt.annotate(f"{i+1}", (lon, lat), fontsize=8, ha='center', va='center')
        
        # 添加网格边界
        for _, row in self.grid_coords_df.iterrows():
            lon_min, lon_max = row['grid_lon_min'], row['grid_lon_max']
            lat_min, lat_max = row['grid_lat_min'], row['grid_lat_max']
            plt.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                     [lat_min, lat_min, lat_max, lat_max, lat_min],
                     'k-', linewidth=0.5, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, label='注意力权重')
        
        # 标题和标签
        plt.title('节点注意力权重分布', fontsize=16)
        plt.xlabel('经度', fontsize=12)
        plt.ylabel('纬度', fontsize=12)
        
        plt.grid(True, alpha=0.1)
        plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("注意力权重图已保存为 'attention_weights.png'")

def main():
    """主函数"""
    visualizer = Visualizer()
    if not visualizer.load_data():
        logger.error("数据加载失败，无法进行全部可视化")
        logger.info("尝试进行部分可视化...")
        
        # 尝试进行不需要模型的可视化
        if visualizer.grid_coords_df is not None and visualizer.global_graph is not None:
            visualizer.visualize_global_knn_graph()
            visualizer.visualize_training_subgraphs()
            visualizer.visualize_training_metrics()
        else:
            logger.error("无法进行任何可视化，缺少必要数据")
        return
    
    # 进行所有可视化
    visualizer.visualize_global_knn_graph()
    visualizer.visualize_training_subgraphs()
    visualizer.visualize_training_metrics()
    visualizer.visualize_attention_weights()
    logger.info("所有可视化已完成，请查看当前目录下的PNG图片")

if __name__ == "__main__":
    main()