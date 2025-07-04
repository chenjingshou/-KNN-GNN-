"""
基于KNN的图构建方法
用于品牌地理位置序列的下一地点预测任务
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import ast
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNGraphBuilder:
    """
    使用KNN构建图的类，支持多种距离度量和特征组合
    """
    
    def __init__(
        self,
        k_spatial: int = 10,  # 基于空间距离的邻居数
        k_semantic: int = 5,  # 基于语义相似度的邻居数
        k_cooccurrence: int = 5,  # 基于共现频率的邻居数
        distance_metric: str = 'euclidean',
        use_brand_similarity: bool = True,
        use_cooccurrence: bool = True,
        edge_weight_type: str = 'combined'  # 'distance', 'similarity', 'combined'
    ):
        self.k_spatial = k_spatial
        self.k_semantic = k_semantic
        self.k_cooccurrence = k_cooccurrence
        self.distance_metric = distance_metric
        self.use_brand_similarity = use_brand_similarity
        self.use_cooccurrence = use_cooccurrence
        self.edge_weight_type = edge_weight_type
        
    def build_spatial_knn_edges(
        self, 
        grid_coords: pd.DataFrame,
        k: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于网格中心点的空间距离构建KNN边
        
        Args:
            grid_coords: 包含grid_id, grid_lon_min/max, grid_lat_min/max的DataFrame
            k: 邻居数量
            
        Returns:
            edge_index: [2, num_edges] 边索引
            edge_weights: [num_edges] 边权重（基于距离的相似度）
        """
        if k is None:
            k = self.k_spatial
            
        # 计算每个网格的中心点
        grid_centers = []
        grid_ids = []
        
        for _, row in grid_coords.iterrows():
            lon_center = (row['grid_lon_min'] + row['grid_lon_max']) / 2
            lat_center = (row['grid_lat_min'] + row['grid_lat_max']) / 2
            grid_centers.append([lon_center, lat_center])
            grid_ids.append(int(row['grid_id']))  # 确保是整数
            
        grid_centers = np.array(grid_centers)
        
        # 使用sklearn的NearestNeighbors
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,  # +1 因为包含自己
            metric=self.distance_metric
        ).fit(grid_centers)
        
        distances, indices = nbrs.kneighbors(grid_centers)
        
        # 构建边（排除自环）
        edge_list = []
        edge_weights = []
        
        for i in range(len(grid_centers)):
            for j in range(1, k + 1):  # 跳过第0个（自己）
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                
                # 双向边
                edge_list.append([i, neighbor_idx])
                edge_list.append([neighbor_idx, i])
                
                # 将距离转换为相似度权重
                # 使用高斯核：exp(-distance^2 / (2 * sigma^2))
                sigma = np.median(distances[:, 1:])  # 使用中位数作为带宽
                weight = np.exp(-distance**2 / (2 * sigma**2))
                edge_weights.extend([weight, weight])
                
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # 去重
        edge_index, edge_weights = self._remove_duplicate_edges(edge_index, edge_weights)
        
        logger.info(f"构建了 {edge_index.shape[1]} 条空间KNN边")
        
        return edge_index, edge_weights
    
    def build_brand_similarity_edges(
        self,
        train_df: pd.DataFrame,
        brand_type_maps: List[Dict],
        k: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于品牌类型相似度构建KNN边
        
        Args:
            train_df: 训练数据
            brand_type_maps: 品牌类型映射
            k: 邻居数量
            
        Returns:
            edge_index: [2, num_edges] 边索引
            edge_weights: [num_edges] 边权重
        """
        if k is None:
            k = self.k_semantic
            
        # 统计每个网格的品牌类型分布
        grid_brand_features = self._compute_grid_brand_features(train_df, brand_type_maps)
        
        # 计算品牌类型相似度
        similarity_matrix = F.cosine_similarity(
            grid_brand_features.unsqueeze(1),
            grid_brand_features.unsqueeze(0),
            dim=2
        )
        
        # 获取top-k相似的网格
        edge_list = []
        edge_weights = []
        
        for i in range(similarity_matrix.shape[0]):
            # 获取top-k+1（包含自己）
            values, indices = torch.topk(similarity_matrix[i], k + 1)
            
            for j in range(1, k + 1):  # 跳过自己
                neighbor_idx = indices[j].item()
                similarity = values[j].item()
                
                if similarity > 0:  # 只添加有正相似度的边
                    edge_list.append([i, neighbor_idx])
                    edge_weights.append(similarity)
                    
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        logger.info(f"构建了 {edge_index.shape[1]} 条品牌相似度边")
        
        return edge_index, edge_weights
    
    def build_cooccurrence_edges(
        self,
        train_df: pd.DataFrame,
        num_grids: int,
        k: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于共现频率构建KNN边
        
        Args:
            train_df: 训练数据
            num_grids: 网格总数
            k: 每个节点保留的top-k共现边
            
        Returns:
            edge_index: [2, num_edges] 边索引
            edge_weights: [num_edges] 边权重（归一化的共现频率）
        """
        if k is None:
            k = self.k_cooccurrence
            
        # 统计共现矩阵
        cooccurrence_matrix = torch.zeros(num_grids, num_grids)
        
        for _, row in train_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                # 统计序列中相邻网格的共现
                for i in range(len(grid_list) - 1):
                    grid1 = int(grid_list[i]) - 1  # 转为0-based整数
                    grid2 = int(grid_list[i + 1]) - 1
                    cooccurrence_matrix[grid1, grid2] += 1
                    cooccurrence_matrix[grid2, grid1] += 1  # 对称
            except:
                continue
                
        # 对每个节点，保留top-k共现的边
        edge_list = []
        edge_weights = []
        
        for i in range(num_grids):
            if cooccurrence_matrix[i].sum() > 0:
                # 获取top-k共现的网格
                values, indices = torch.topk(cooccurrence_matrix[i], min(k, (cooccurrence_matrix[i] > 0).sum()))
                
                for j in range(len(indices)):
                    if values[j] > 0:
                        edge_list.append([i, indices[j].item()])
                        edge_weights.append(values[j].item())
                        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # 归一化权重
        if len(edge_weights) > 0:
            edge_weights = edge_weights / edge_weights.max()
            
        logger.info(f"构建了 {edge_index.shape[1]} 条共现边")
        
        return edge_index, edge_weights
    
    def build_combined_graph(
        self,
        grid_coords: pd.DataFrame,
        train_df: pd.DataFrame,
        brand_type_maps: Optional[List[Dict]] = None,
        node_features: Optional[torch.Tensor] = None
    ) -> Data:
        """
        构建组合的KNN图
        
        Args:
            grid_coords: 网格坐标数据
            train_df: 训练数据
            brand_type_maps: 品牌类型映射
            node_features: 节点特征
            
        Returns:
            PyG Data对象
        """
        num_grids = len(grid_coords)
        all_edges = []
        all_weights = []
        
        # 1. 空间KNN边
        spatial_edges, spatial_weights = self.build_spatial_knn_edges(grid_coords)
        all_edges.append(spatial_edges)
        all_weights.append(spatial_weights)
        
        # 2. 品牌相似度边（如果启用）
        if self.use_brand_similarity and brand_type_maps is not None:
            brand_edges, brand_weights = self.build_brand_similarity_edges(
                train_df, brand_type_maps
            )
            if brand_edges.shape[1] > 0:
                all_edges.append(brand_edges)
                all_weights.append(brand_weights)
        
        # 3. 共现边（如果启用）
        if self.use_cooccurrence:
            cooc_edges, cooc_weights = self.build_cooccurrence_edges(
                train_df, num_grids
            )
            if cooc_edges.shape[1] > 0:
                all_edges.append(cooc_edges)
                all_weights.append(cooc_weights)
        
        # 合并所有边
        edge_index = torch.cat(all_edges, dim=1)
        
        # 合并权重
        if self.edge_weight_type == 'combined':
            # 对不同类型的权重进行加权组合
            weights = []
            offset = 0
            weight_scales = [1.0, 0.5, 0.8]  # 空间、品牌、共现的权重系数
            
            for i, w in enumerate(all_weights):
                scale = weight_scales[i] if i < len(weight_scales) else 1.0
                weights.append(w * scale)
                
            edge_weights = torch.cat(weights)
        else:
            edge_weights = torch.cat(all_weights)
            
        # 去重并合并权重
        edge_index, edge_weights = self._merge_duplicate_edges(edge_index, edge_weights)
        
        # 创建节点特征（如果没有提供）
        if node_features is None:
            # 使用网格中心坐标作为基础特征
            centers = [None] * num_grids  # 预分配空间
            for _, row in grid_coords.iterrows():
                grid_idx = int(row['grid_id']) - 1  # 转为0-based
                lon_center = (row['grid_lon_min'] + row['grid_lon_max']) / 2
                lat_center = (row['grid_lat_min'] + row['grid_lat_max']) / 2
                centers[grid_idx] = [lon_center, lat_center]
            
            # 检查是否所有网格都有坐标
            for i, center in enumerate(centers):
                if center is None:
                    raise ValueError(f"网格 {i+1} 没有坐标数据")
                    
            node_features = torch.tensor(centers, dtype=torch.float)
            
        # 创建PyG Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights.unsqueeze(1),
            num_nodes=num_grids
        )
        
        logger.info(f"构建的图: {data}")
        logger.info(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
        logger.info(f"平均度: {data.num_edges / data.num_nodes:.2f}")
        
        return data
    
    def _compute_grid_brand_features(
        self,
        train_df: pd.DataFrame,
        brand_type_maps: List[Dict]
    ) -> torch.Tensor:
        """计算每个网格的品牌类型特征向量"""
        # 这里简化处理，实际可以更复杂
        num_grids = 106  # 根据你的数据
        feature_dim = sum(len(m) for m in brand_type_maps)
        features = torch.zeros(num_grids, feature_dim)
        
        # 统计每个网格出现的品牌类型
        for _, row in train_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                brand_types = row['brand_type'].split(';')
                
                for grid_id in grid_list:
                    grid_idx = int(grid_id) - 1  # 转为0-based整数
                    
                    # 编码品牌类型
                    offset = 0
                    for level, brand_type in enumerate(brand_types):
                        if level < len(brand_type_maps):
                            brand_type = brand_type.strip()
                            if brand_type in brand_type_maps[level]:
                                type_idx = brand_type_maps[level][brand_type]
                                features[grid_idx, offset + type_idx] += 1
                            offset += len(brand_type_maps[level])
            except:
                continue
                
        # 归一化
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def _remove_duplicate_edges(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """去除重复边"""
        # 创建边的唯一标识
        edge_set = set()
        unique_edges = []
        unique_weights = []
        
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge not in edge_set:
                edge_set.add(edge)
                unique_edges.append(i)
                
        edge_index = edge_index[:, unique_edges]
        edge_weights = edge_weights[unique_edges]
        
        return edge_index, edge_weights
    
    def _merge_duplicate_edges(
        self,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """合并重复边的权重"""
        # 创建边到权重的映射
        edge_weight_dict = {}
        
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            weight = edge_weights[i].item()
            
            if edge in edge_weight_dict:
                # 合并策略：取最大值
                edge_weight_dict[edge] = max(edge_weight_dict[edge], weight)
            else:
                edge_weight_dict[edge] = weight
                
        # 重建边和权重
        edges = list(edge_weight_dict.keys())
        weights = list(edge_weight_dict.values())
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weights = torch.tensor(weights, dtype=torch.float)
        
        return edge_index, edge_weights


# 使用示例
def build_knn_graph_for_training(
    train_csv_path: str,
    grid_coords_csv_path: str,
    brand_type_maps: Optional[List[Dict]] = None,
    save_path: str = 'graph_data_knn.pt',
    **kwargs
):
    """
    为训练构建KNN图
    
    Args:
        train_csv_path: 训练数据CSV路径
        grid_coords_csv_path: 网格坐标CSV路径
        brand_type_maps: 品牌类型映射
        save_path: 保存路径
        **kwargs: 传递给KNNGraphBuilder的参数
    """
    # 读取数据
    train_df = pd.read_csv(train_csv_path)
    grid_coords = pd.read_csv(grid_coords_csv_path)
    
    # 创建图构建器
    builder = KNNGraphBuilder(**kwargs)
    
    # 构建图
    graph_data = builder.build_combined_graph(
        grid_coords=grid_coords,
        train_df=train_df,
        brand_type_maps=brand_type_maps
    )
    
    # 保存图
    torch.save(graph_data, save_path)
    logger.info(f"图已保存到 {save_path}")
    
    return graph_data


# 如果需要为每个样本构建子图的版本
class KNNSubgraphBuilder(KNNGraphBuilder):
    """
    为每个序列样本构建KNN子图
    """
    
    def build_instance_subgraph(
        self,
        input_grid_ids: List[int],
        global_graph: Data,
        k_hops: int = 2,
        max_subgraph_size: int = 50
    ) -> Data:
        """
        为单个序列构建KNN子图
        
        Args:
            input_grid_ids: 输入序列的网格ID列表
            global_graph: 全局图
            k_hops: 扩展的跳数
            max_subgraph_size: 最大子图大小
            
        Returns:
            子图Data对象
        """
        # 获取k跳邻居
        subgraph_nodes = set(int(gid) for gid in input_grid_ids)
        current_nodes = set(int(gid) for gid in input_grid_ids)
        
        edge_index = global_graph.edge_index
        
        for hop in range(k_hops):
            next_nodes = set()
            for node in current_nodes:
                # 找到所有邻居
                mask = edge_index[0] == node
                neighbors = edge_index[1][mask].tolist()
                next_nodes.update(neighbors)
                
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            # 限制子图大小
            if len(subgraph_nodes) > max_subgraph_size:
                # 可以基于某种评分选择最重要的节点
                subgraph_nodes = set(list(subgraph_nodes)[:max_subgraph_size])
                break
                
        # 提取子图
        subgraph_nodes = sorted(list(subgraph_nodes))
        node_mapping = {node: i for i, node in enumerate(subgraph_nodes)}
        
        # 重映射边
        subgraph_edges = []
        subgraph_weights = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_mapping and dst in node_mapping:
                subgraph_edges.append([node_mapping[src], node_mapping[dst]])
                if global_graph.edge_attr is not None:
                    subgraph_weights.append(global_graph.edge_attr[i])
                    
        # 创建子图
        subgraph = Data(
            x=global_graph.x[subgraph_nodes] if global_graph.x is not None else None,
            edge_index=torch.tensor(subgraph_edges, dtype=torch.long).t() if subgraph_edges else torch.zeros((2, 0), dtype=torch.long),
            num_nodes=len(subgraph_nodes)
        )
        
        # 只在有边权重且有边的情况下添加edge_attr
        if subgraph_weights and hasattr(global_graph, 'edge_attr') and global_graph.edge_attr is not None:
            subgraph.edge_attr = torch.stack(subgraph_weights)
        
        # 保存映射信息 - 不保存字典，只保存必要的tensor信息
        # subgraph.node_mapping = node_mapping  # 不要保存字典
        subgraph.global_node_ids = torch.tensor(subgraph_nodes, dtype=torch.long)
        
        # 如果需要知道输入节点在子图中的位置，创建一个mask
        input_positions = []
        for gid in input_grid_ids:
            if gid in node_mapping:
                input_positions.append(node_mapping[gid])
        subgraph.input_positions = torch.tensor(input_positions, dtype=torch.long) if input_positions else torch.tensor([], dtype=torch.long)
        
        return subgraph


if __name__ == "__main__":
    # 测试代码
    # print("KNN图构建器已准备就绪")
    
    # 示例：构建KNN图
    graph = build_knn_graph_for_training(
        train_csv_path='train_data(1).csv',
        grid_coords_csv_path='grid_coordinates(1).csv',
        k_spatial=15,  # 空间邻居数
        k_semantic=8,  # 语义邻居数
        k_cooccurrence=10,  # 共现邻居数
        use_brand_similarity=True,
        use_cooccurrence=True,
        edge_weight_type='combined'
    )