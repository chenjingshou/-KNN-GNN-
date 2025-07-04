# 文件名: process_grid_data_new.py
# 描述: (已修改以正确处理全局edge_attr并传递到子图, 并修正brand_type_tensor形状)
#       从全局图数据为每个序列构建诱导子图样本。

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader # 用于测试
from torch_geometric.utils import subgraph # 用于构建诱导子图
import ast
import random
import os
import sys
from collections import defaultdict, Counter

# --- 配置 ---
GRID_COORDS_FILE = 'grid_coordinates(1).csv'
# BRAND_DATA_FILE = 'train_data(1).csv' # 将在创建Dataset实例时传入

GLOBAL_GRAPH_DATA_PATH = 'graph_data_tencent_rich_ip_features.pt'
# GLOBAL_GRAPH_DATA_PATH = None # 取消注释此行以从零开始计算全局图 (仅用于测试或无预计算图时)

MAX_HIERARCHY_LEVELS = 3
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
TARGET_SUBSAMPLING_RATIO = 1
MIN_ORIGINAL_SEQ_LEN = 2
NUM_GLOBAL_NODES = 106

# --- 辅助函数 ---
def calculate_grid_centers(df_coords):
    df_coords['lon_center'] = (df_coords['grid_lon_min'] + df_coords['grid_lon_max']) / 2
    df_coords['lat_center'] = (df_coords['grid_lat_min'] + df_coords['grid_lat_max']) / 2
    return df_coords

def find_geometric_global_edges(df_coords, tolerance=1e-9):
    """
    (Fallback function)
    从 grid_coordinates DataFrame 构建基于地理邻接的全局 PyG edge_index.
    grid_id 是 1-based, 输出的 edge_index 是 0-based.
    此函数现在只返回 edge_index，因为地理邻接边通常无特定权重（或权重为1）。
    """
    print("构建基于地理邻接的全局边信息 (Fallback)...")
    edge_list_tuples = []
    num_grids = len(df_coords)
    
    grid_data_dict = df_coords.set_index('grid_id')[['grid_lon_min', 'grid_lat_min', 'grid_lon_max', 'grid_lat_max']].to_dict('index')
    grid_ids_sorted_list = sorted(df_coords['grid_id'].unique().tolist())

    # ... (与您之前脚本中 find_adjacent_grids 或 find_global_adjacencies 类似的逻辑)
    for i in range(num_grids):
        for j in range(i + 1, num_grids):
            grid_id1 = grid_ids_sorted_list[i]
            grid_id2 = grid_ids_sorted_list[j]
            if grid_id1 not in grid_data_dict or grid_id2 not in grid_data_dict: continue
            g1, g2 = grid_data_dict[grid_id1], grid_data_dict[grid_id2]
            is_h_adj = (abs(g1['grid_lat_min'] - g2['grid_lat_min']) < tolerance and 
                        abs(g1['grid_lat_max'] - g2['grid_lat_max']) < tolerance and
                        (abs(g1['grid_lon_max'] - g2['grid_lon_min']) < tolerance or 
                         abs(g1['grid_lon_min'] - g2['grid_lon_max']) < tolerance))
            is_v_adj = (abs(g1['grid_lon_min'] - g2['grid_lon_min']) < tolerance and 
                        abs(g1['grid_lon_max'] - g2['grid_lon_max']) < tolerance and
                        (abs(g1['grid_lat_max'] - g2['grid_lat_min']) < tolerance or 
                         abs(g1['grid_lat_min'] - g2['grid_lat_max']) < tolerance))
            if is_h_adj or is_v_adj:
                edge_list_tuples.append(tuple(sorted((grid_id1, grid_id2))))
    
    edge_list_tuples = sorted(list(set(edge_list_tuples)))
    
    edge_index_list_0based = [[], []]
    for u_gid, v_gid in edge_list_tuples:
        u_idx, v_idx = u_gid - 1, v_gid - 1
        edge_index_list_0based[0].extend([u_idx, v_idx])
        edge_index_list_0based[1].extend([v_idx, u_idx])
        
    global_edge_index = torch.tensor(edge_index_list_0based, dtype=torch.long)
    print(f"地理邻接边构建完成。找到 {len(edge_list_tuples)} 条独特邻接关系, edge_index 形状: {global_edge_index.shape}")
    return global_edge_index

def process_hierarchical_brand_type(type_str, max_levels, pad_token, unknown_token):
    try:
        levels = str(type_str).split(';')
        processed_levels = [s.strip() for s in levels if s.strip()][:max_levels]
        while len(processed_levels) < max_levels:
            processed_levels.append(pad_token)
        return processed_levels
    except:
        return [unknown_token] * max_levels

def build_brand_type_maps(brand_data_df_list, max_levels, pad_token, unknown_token):
    all_parsed_levels = []
    for df in brand_data_df_list:
        if 'brand_type' in df.columns:
            parsed = df['brand_type'].apply(lambda x: process_hierarchical_brand_type(x, max_levels, pad_token, unknown_token))
            all_parsed_levels.extend(list(parsed))
    if not all_parsed_levels:
        print("警告: 未能从任何数据帧中解析品牌类型。")
        return [defaultdict(lambda: 0) for _ in range(max_levels)], [1 for _ in range(max_levels)]
    brand_type_maps = []
    brand_vocab_sizes = []
    levels_data = [[] for _ in range(max_levels)]
    for item_levels in all_parsed_levels:
        for i in range(max_levels):
            levels_data[i].append(item_levels[i] if i < len(item_levels) else pad_token)
    for level in range(max_levels):
        current_level_types = levels_data[level]
        unique_types = set(current_level_types)
        unique_types.add(unknown_token); unique_types.add(pad_token)
        sorted_types = sorted(list(unique_types))
        type_to_idx_map = defaultdict(lambda: sorted_types.index(unknown_token))
        for idx, type_str_val in enumerate(sorted_types): type_to_idx_map[type_str_val] = idx
        brand_type_maps.append(type_to_idx_map)
        brand_vocab_sizes.append(len(sorted_types))
        print(f"品牌层级 {level+1} 词汇表大小: {len(sorted_types)}")
    return brand_type_maps, brand_vocab_sizes

class InstanceGraphDataset(Dataset):
    def __init__(self, grid_coords_file, brand_data_file,
                 global_graph_path,
                 brand_type_maps, brand_vocab_sizes,
                 num_global_nodes,
                 min_original_len=MIN_ORIGINAL_SEQ_LEN,
                 max_levels=MAX_HIERARCHY_LEVELS,
                 pad_token=PAD_TOKEN, unknown_token=UNKNOWN_TOKEN,
                 target_subsampling_ratio=TARGET_SUBSAMPLING_RATIO,
                 is_test_set=False,
                 dataset_name="Dataset"):
        super().__init__()
        self.brand_data_file = brand_data_file
        self.grid_coords_file = grid_coords_file
        self.global_graph_path = global_graph_path
        self.brand_type_maps = brand_type_maps
        self.brand_vocab_sizes = brand_vocab_sizes
        self.num_global_nodes = num_global_nodes
        self.min_original_len = min_original_len
        self.max_levels = max_levels
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.target_subsampling_ratio = target_subsampling_ratio
        self.dataset_name = dataset_name
        # (is_test_set logic for target_subsampling_ratio can be added if needed)

        self._load_global_graph_info()
        self._generate_samples()

    def _load_global_graph_info(self):
        print(f"[{self.dataset_name}] 加载全局图信息...")
        loaded_from_pt = False
        if self.global_graph_path and os.path.exists(self.global_graph_path):
            try:
                data = torch.load(self.global_graph_path)
                if hasattr(data, 'x') and data.x is not None and \
                   hasattr(data, 'edge_index') and data.edge_index is not None and \
                   data.x.shape[0] == self.num_global_nodes:
                    
                    print(f"[{self.dataset_name}] 从 '{self.global_graph_path}' 加载预计算的全局节点特征和边索引。")
                    self.global_x = data.x
                    self.global_edge_index = data.edge_index # 0-based
                    
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        self.global_edge_attr = data.edge_attr
                        print(f"[{self.dataset_name}] 同时加载了全局 edge_attr, 形状: {self.global_edge_attr.shape}")
                    else:
                        print(f"[{self.dataset_name}] 全局图中未找到 edge_attr，将使用默认权重1（如果需要）。")
                        # 如果边是地理邻接且无权重，可以设为全1；如果是共现图但丢失了权重，这是个问题
                        # 对于诱导子图，如果全局图有权重但这里没加载到，子图的边权重会丢失
                        # PyG 的 subgraph 函数如果 edge_attr=None，则返回的 sub_edge_attr 也是 None
                        self.global_edge_attr = None # 或者 torch.ones((self.global_edge_index.size(1), 1), dtype=torch.float)
                                                    # 具体取决于全局图的性质和后续是否强制需要edge_attr

                    if data.num_nodes != self.num_global_nodes:
                         print(f"警告: 全局图文件中的节点数 ({data.num_nodes}) 与配置 ({self.num_global_nodes}) 不符。")
                    loaded_from_pt = True
                else:
                    print(f"警告: '{self.global_graph_path}' 文件不完整或节点数不匹配。")
            except Exception as e:
                print(f"警告: 加载 '{self.global_graph_path}' 失败 ({e})。")
        
        if not loaded_from_pt:
            print(f"[{self.dataset_name}] 将从 '{self.grid_coords_file}' 计算基础全局图信息 (地理邻接)。")
            df_coords = pd.read_csv(self.grid_coords_file)
            if not (df_coords['grid_id'].min() == 1 and df_coords['grid_id'].max() == self.num_global_nodes and \
                   len(df_coords['grid_id'].unique()) == self.num_global_nodes):
                raise ValueError(f"grid_id 必须是从1到 {self.num_global_nodes} 的连续整数。请预处理 {self.grid_coords_file}。")

            df_coords = calculate_grid_centers(df_coords.copy())
            df_coords_sorted = df_coords.sort_values(by='grid_id').reset_index(drop=True)
            self.global_x = torch.tensor(df_coords_sorted[['lon_center', 'lat_center']].values, dtype=torch.float)
            self.global_edge_index = find_geometric_global_edges(df_coords) # 只返回 edge_index
            # 对于地理邻接，通常不带特定权重，或权重为1。如果模型需要edge_attr，则创建它。
            self.global_edge_attr = torch.ones((self.global_edge_index.size(1), 1), dtype=torch.float) 
            print(f"[{self.dataset_name}] 基于地理邻接的全局图已构建。edge_attr 设为全1。")

    def _generate_samples(self):
        print(f"[{self.dataset_name}] 开始从 '{self.brand_data_file}' 生成实例图样本...")
        df_brand = pd.read_csv(self.brand_data_file)
        self.samples = []
        original_records_count = len(df_brand)
        num_generated_subgraphs = 0

        for _, row in df_brand.iterrows():
            try:
                raw_grid_id_list_str = row['grid_id_list']
                original_gid_sequence_1based = ast.literal_eval(raw_grid_id_list_str)
                if not isinstance(original_gid_sequence_1based, list): continue
                
                original_global_indices_0based = [gid - 1 for gid in original_gid_sequence_1based if 1 <= gid <= self.num_global_nodes]
                if len(original_global_indices_0based) < self.min_original_len: continue

                brand_type_str = row['brand_type']
                processed_levels = process_hierarchical_brand_type(brand_type_str, self.max_levels, self.pad_token, self.unknown_token)
                brand_type_indices = [self.brand_type_maps[lvl_idx][processed_levels[lvl_idx]] for lvl_idx in range(self.max_levels)]
                brand_type_tensor = torch.tensor(brand_type_indices, dtype=torch.long)

                num_elements_in_sequence = len(original_global_indices_0based)
                num_targets_to_sample = max(1, int(num_elements_in_sequence * self.target_subsampling_ratio))
                num_targets_to_sample = min(num_targets_to_sample, num_elements_in_sequence)
                if num_targets_to_sample == 0 and num_elements_in_sequence >= self.min_original_len : num_targets_to_sample = 1
                
                possible_target_positions = list(range(num_elements_in_sequence))
                if not possible_target_positions: continue
                chosen_target_positions = random.sample(possible_target_positions, num_targets_to_sample)

                for target_pos_in_seq in chosen_target_positions:
                    target_global_idx_0based = original_global_indices_0based[target_pos_in_seq]
                    input_global_indices_0based_list = [gid for i, gid in enumerate(original_global_indices_0based) if i != target_pos_in_seq]
                    if not input_global_indices_0based_list: continue

                    subgraph_data = self._construct_subgraph_for_sample(
                        input_global_indices_0based_list, target_global_idx_0based, brand_type_tensor
                    )
                    if subgraph_data:
                        self.samples.append(subgraph_data)
                        num_generated_subgraphs += 1
            except (ValueError, SyntaxError) as e:
                # print(f"警告: 解析行 '{row.get('brand_name', 'N/A')}' (grid_list: {row.get('grid_id_list', 'N/A')}) 失败: {e}. 跳过。")
                continue
            except Exception as ex:
                # print(f"警告: 处理行 '{row.get('brand_name', 'N/A')}' 时发生未知错误: {ex}. 跳过。")
                continue
        print(f"[{self.dataset_name}] 完成样本生成。从 {original_records_count} 条原始记录中生成了 {num_generated_subgraphs} 个子图样本 (采样率: {self.target_subsampling_ratio:.2f})。")

    def _construct_subgraph_for_sample(self, input_global_indices_0based_list, target_global_idx_0based, brand_type_tensor):
        unique_input_global_indices = sorted(list(set(input_global_indices_0based_list)))
        if not unique_input_global_indices: return None

        subgraph_nodes_global_tensor = torch.tensor(unique_input_global_indices, dtype=torch.long)
        
        x_sub = self.global_x[subgraph_nodes_global_tensor]
        
        # 使用 torch_geometric.utils.subgraph 构建诱导子图
        # 它会自动处理节点索引的重新标记 (relabel_nodes=True)
        # 并且能够传递 edge_attr
        edge_index_sub, edge_attr_sub = subgraph(
            subset=subgraph_nodes_global_tensor,
            edge_index=self.global_edge_index,
            edge_attr=self.global_edge_attr, # 传递全局边属性
            relabel_nodes=True,
            num_nodes=self.num_global_nodes # 全局节点数，用于正确处理子集
        )
        
        # 如果原始的 global_edge_attr 是 None (例如地理邻接图且未设权重)
        # 那么 edge_attr_sub 也会是 None。模型需要能处理这种情况或我们在这里赋默认值。
        if edge_attr_sub is None and edge_index_sub.numel() > 0: # 如果有边但没有属性
            # print(f"警告: 子图边属性为None，但存在边。将边属性设为全1。子图节点数: {len(unique_input_global_indices)}")
            edge_attr_sub = torch.ones((edge_index_sub.size(1), 1), dtype=torch.float)
        elif edge_attr_sub is None and edge_index_sub.numel() == 0: # 没有边，自然没有边属性
             edge_attr_sub = torch.empty((0,1), dtype=torch.float) # 保持一致性，给一个空但有正确第二维的张量

        y_target = torch.tensor([target_global_idx_0based], dtype=torch.long)
        
        data = Data(x=x_sub, edge_index=edge_index_sub, edge_attr=edge_attr_sub, y=y_target)
        # 修正 brand_type_tensor 形状以正确批处理
        data.brand_type_tensor = brand_type_tensor.unsqueeze(0) # Shape: [1, num_levels]
        data.input_nodes_for_masking = torch.tensor(unique_input_global_indices, dtype=torch.long)
        # data.num_nodes_in_subgraph = len(unique_input_global_indices) # Data对象有num_nodes属性

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- 主程序示例 (用于测试 Dataset 类) ---
if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric 未安装。请安装后运行。")
        sys.exit()

    if not os.path.exists(GRID_COORDS_FILE): sys.exit(f"错误: '{GRID_COORDS_FILE}' 未找到。")
    train_brand_file = 'train_data(1).csv'
    test_brand_file = 'test_data(1).csv'
    if not os.path.exists(train_brand_file): sys.exit(f"错误: '{train_brand_file}' 未找到。")
    if not os.path.exists(test_brand_file): print(f"警告: '{test_brand_file}' 未找到。测试集部分将跳过。")

    print("\n--- 构建全局品牌类型映射 ---")
    df_train_brands_for_map = pd.read_csv(train_brand_file)
    brand_data_dfs_for_map = [df_train_brands_for_map]
    if os.path.exists(test_brand_file):
        df_test_brands_for_map = pd.read_csv(test_brand_file)
        brand_data_dfs_for_map.append(df_test_brands_for_map)
    
    brand_type_maps, brand_vocab_sizes = build_brand_type_maps(
        brand_data_dfs_for_map, MAX_HIERARCHY_LEVELS, PAD_TOKEN, UNKNOWN_TOKEN
    )

    print("\n--- 创建训练数据集 ---")
    train_dataset = InstanceGraphDataset(
        grid_coords_file=GRID_COORDS_FILE,
        brand_data_file=train_brand_file,
        global_graph_path=GLOBAL_GRAPH_DATA_PATH,
        brand_type_maps=brand_type_maps,
        brand_vocab_sizes=brand_vocab_sizes,
        num_global_nodes=NUM_GLOBAL_NODES,
        target_subsampling_ratio=0.9,
        dataset_name="TrainInstanceGraphs"
    )

    if len(train_dataset) > 0:
        print(f"成功创建训练数据集，包含 {len(train_dataset)} 个子图样本。")
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        print("\n--- DataLoader 示例输出 (训练集) ---")
        for i, batch_data in enumerate(train_loader):
            if i >= 2: break
            print(f"\n批次 {i+1}:")
            print(f"  Batch对象: {batch_data}")
            print(f"  x.shape: {batch_data.x.shape}, edge_index.shape: {batch_data.edge_index.shape}, y.shape: {batch_data.y.shape}")
            if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
                print(f"  edge_attr.shape: {batch_data.edge_attr.shape}")
            print(f"  brand_type_tensor.shape: {batch_data.brand_type_tensor.shape}") # 应为 [batch_size, num_levels]
            print(f"  input_nodes_for_masking (部分): {batch_data.input_nodes_for_masking[:10]}...")
            print(f"  批次向量 (部分): {batch_data.batch[:20]}...")
            print(f"  子图数量: {batch_data.num_graphs}")
    else:
        print("未能生成任何训练样本。")

    if os.path.exists(test_brand_file):
        print("\n--- 创建测试数据集 ---")
        test_dataset = InstanceGraphDataset(
            grid_coords_file=GRID_COORDS_FILE,
            brand_data_file=test_brand_file,
            global_graph_path=GLOBAL_GRAPH_DATA_PATH,
            brand_type_maps=brand_type_maps,
            brand_vocab_sizes=brand_vocab_sizes,
            num_global_nodes=NUM_GLOBAL_NODES,
            target_subsampling_ratio=0.05, 
            is_test_set=True, # is_test_set 可以用来调整内部逻辑，例如采样率
            dataset_name="TestInstanceGraphs"
        )
        if len(test_dataset) > 0:
            print(f"成功创建测试数据集，包含 {len(test_dataset)} 个子图样本。")
        else:
            print("未能生成任何测试样本。")
    print("\n脚本执行完毕。")