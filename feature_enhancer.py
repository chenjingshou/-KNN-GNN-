"""
节点特征增强器
为网格节点添加更丰富的特征
"""

import torch
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class NodeFeatureEnhancer:
    """
    为图节点添加更丰富的特征
    """
    
    def __init__(self, num_grids=106):
        self.num_grids = num_grids
        self.scaler = StandardScaler()
        
    def enhance_node_features(
        self,
        base_features,  # 基础特征 (如经纬度)
        train_df,       # 训练数据
        grid_coords_df, # 网格坐标数据
        brand_type_maps=None
    ):
        """
        增强节点特征
        
        Args:
            base_features: 基础节点特征 [num_nodes, base_dim]
            train_df: 训练数据DataFrame
            grid_coords_df: 网格坐标DataFrame
            brand_type_maps: 品牌类型映射
            
        Returns:
            enhanced_features: 增强后的特征 [num_nodes, enhanced_dim]
        """
        feature_list = [base_features]
        
        # 1. 网格面积特征
        grid_areas = self._compute_grid_areas(grid_coords_df)
        feature_list.append(grid_areas.unsqueeze(1))
        
        # 2. 访问频率特征
        visit_freq = self._compute_visit_frequency(train_df)
        feature_list.append(visit_freq.unsqueeze(1))
        
        # 3. 品牌多样性特征
        brand_diversity = self._compute_brand_diversity(train_df)
        feature_list.append(brand_diversity.unsqueeze(1))
        
        # 4. 序列位置特征（开始/中间/结束的概率）
        position_features = self._compute_position_features(train_df)
        feature_list.append(position_features)
        
        # 5. 转移概率特征
        transition_features = self._compute_transition_features(train_df)
        feature_list.append(transition_features)
        
        # 6. 品牌类型分布特征（如果提供了映射）
        if brand_type_maps is not None:
            brand_type_features = self._compute_brand_type_distribution(train_df, brand_type_maps)
            feature_list.append(brand_type_features)
        
        # 7. 时间特征（如果数据中有时间信息）
        # time_features = self._compute_temporal_features(train_df)
        # if time_features is not None:
        #     feature_list.append(time_features)
        
        # 合并所有特征
        enhanced_features = torch.cat(feature_list, dim=1)
        
        # 标准化
        enhanced_features_np = enhanced_features.numpy()
        enhanced_features_normalized = self.scaler.fit_transform(enhanced_features_np)
        enhanced_features = torch.tensor(enhanced_features_normalized, dtype=torch.float)
        
        logger.info(f"特征增强完成: {base_features.shape[1]}维 -> {enhanced_features.shape[1]}维")
        
        return enhanced_features
    
    def _compute_grid_areas(self, grid_coords_df):
        """计算每个网格的面积"""
        areas = torch.zeros(self.num_grids)
        
        for _, row in grid_coords_df.iterrows():
            grid_id = int(row['grid_id']) - 1  # 转为0-based整数
            lon_span = row['grid_lon_max'] - row['grid_lon_min']
            lat_span = row['grid_lat_max'] - row['grid_lat_min']
            # 简化计算，实际应该考虑地球曲率
            area = lon_span * lat_span * 111 * 111  # 大约的平方公里
            areas[grid_id] = area
            
        return areas
    
    def _compute_visit_frequency(self, train_df):
        """计算每个网格的访问频率"""
        visit_counts = torch.zeros(self.num_grids)
        
        for _, row in train_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                for grid_id in grid_list:
                    visit_counts[grid_id - 1] += 1
            except:
                continue
                
        # 归一化
        visit_freq = visit_counts / (visit_counts.sum() + 1e-8)
        
        return visit_freq
    
    def _compute_brand_diversity(self, train_df):
        """计算每个网格的品牌多样性（熵）"""
        from collections import defaultdict
        import math
        
        grid_brands = defaultdict(set)
        
        for _, row in train_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                brand = row['brand_name']
                for grid_id in grid_list:
                    grid_brands[grid_id - 1].add(brand)
            except:
                continue
        
        diversity = torch.zeros(self.num_grids)
        for grid_id in range(self.num_grids):
            num_brands = len(grid_brands[grid_id])
            if num_brands > 1:
                # 使用对数作为多样性度量
                diversity[grid_id] = math.log(num_brands)
                
        # 归一化
        if diversity.max() > 0:
            diversity = diversity / diversity.max()
            
        return diversity
    
    def _compute_position_features(self, train_df):
        """计算网格在序列中的位置特征"""
        position_counts = torch.zeros(self.num_grids, 3)  # [开始, 中间, 结束]
        
        for _, row in train_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                if len(grid_list) >= 2:
                    # 开始位置
                    position_counts[grid_list[0] - 1, 0] += 1
                    # 结束位置
                    position_counts[grid_list[-1] - 1, 2] += 1
                    # 中间位置
                    for grid_id in grid_list[1:-1]:
                        position_counts[grid_id - 1, 1] += 1
            except:
                continue
        
        # 归一化到概率
        row_sums = position_counts.sum(dim=1, keepdim=True)
        position_probs = position_counts / (row_sums + 1e-8)
        
        return position_probs
    
    def _compute_transition_features(self, train_df):
        """计算转移特征（入度、出度、自环）"""
        in_degree = torch.zeros(self.num_grids)
        out_degree = torch.zeros(self.num_grids)
        self_loops = torch.zeros(self.num_grids)
        
        for _, row in train_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                for i in range(len(grid_list) - 1):
                    src = grid_list[i] - 1
                    dst = grid_list[i + 1] - 1
                    
                    out_degree[src] += 1
                    in_degree[dst] += 1
                    
                    if src == dst:
                        self_loops[src] += 1
            except:
                continue
        
        # 归一化
        max_degree = max(in_degree.max(), out_degree.max())
        if max_degree > 0:
            in_degree = in_degree / max_degree
            out_degree = out_degree / max_degree
            
        if self_loops.max() > 0:
            self_loops = self_loops / self_loops.max()
            
        return torch.stack([in_degree, out_degree, self_loops], dim=1)
    
    def _compute_brand_type_distribution(self, train_df, brand_type_maps):
        """计算每个网格的品牌类型分布"""
        # 简化版：只统计一级品牌类型
        num_types = len(brand_type_maps[0])
        type_counts = torch.zeros(self.num_grids, num_types)
        
        for _, row in train_df.iterrows():
            try:
                grid_list = ast.literal_eval(row['grid_id_list'])
                brand_type = row['brand_type'].split(';')[0].strip()
                
                if brand_type in brand_type_maps[0]:
                    type_idx = brand_type_maps[0][brand_type]
                    for grid_id in grid_list:
                        type_counts[grid_id - 1, type_idx] += 1
            except:
                continue
        
        # 归一化到概率分布
        row_sums = type_counts.sum(dim=1, keepdim=True)
        type_distribution = type_counts / (row_sums + 1e-8)
        
        return type_distribution
    
    def _compute_neighbor_features(self, edge_index, node_features):
        """计算邻居聚合特征"""
        # 简单的邻居平均
        num_nodes = node_features.shape[0]
        neighbor_sum = torch.zeros_like(node_features)
        neighbor_count = torch.zeros(num_nodes)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            neighbor_sum[dst] += node_features[src]
            neighbor_count[dst] += 1
            
        neighbor_mean = neighbor_sum / (neighbor_count.unsqueeze(1) + 1e-8)
        
        return neighbor_mean


def enhance_graph_features(graph_data, train_df, grid_coords_df, brand_type_maps=None):
    """
    便捷函数：增强图的节点特征
    
    Args:
        graph_data: PyG Data对象
        train_df: 训练数据
        grid_coords_df: 网格坐标数据
        brand_type_maps: 品牌类型映射
        
    Returns:
        增强后的graph_data
    """
    enhancer = NodeFeatureEnhancer(num_grids=graph_data.num_nodes)
    
    # 获取基础特征
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        base_features = graph_data.x
    else:
        # 如果没有特征，使用网格中心坐标
        centers = []
        # 确保按grid_id排序
        grid_coords_df_sorted = grid_coords_df.sort_values('grid_id')
        for _, row in grid_coords_df_sorted.iterrows():
            lon_center = (row['grid_lon_min'] + row['grid_lon_max']) / 2
            lat_center = (row['grid_lat_min'] + row['grid_lat_max']) / 2
            centers.append([lon_center, lat_center])
        base_features = torch.tensor(centers, dtype=torch.float)
    
    # 增强特征
    enhanced_features = enhancer.enhance_node_features(
        base_features,
        train_df,
        grid_coords_df,
        brand_type_maps
    )
    
    # 更新图数据
    graph_data.x = enhanced_features
    
    return graph_data


if __name__ == "__main__":
    print("节点特征增强器已准备就绪")
    
    # 使用示例：
    # from torch_geometric.data import Data
    # import pandas as pd
    # 
    # # 加载数据
    # train_df = pd.read_csv('train_data(1).csv')
    # grid_coords_df = pd.read_csv('grid_coordinates(1).csv')
    # graph_data = torch.load('graph_data_knn.pt')
    # 
    # # 增强特征
    # enhanced_graph = enhance_graph_features(
    #     graph_data, 
    #     train_df, 
    #     grid_coords_df,
    #     brand_type_maps
    # )
    # 
    # # 保存增强后的图
    # torch.save(enhanced_graph, 'graph_data_knn_enhanced.pt')