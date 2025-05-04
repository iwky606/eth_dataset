import dgl
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_mongo_data():
    from dao.mongo_client import db

    nodes = list(db.train_nodes.find({}, {"address": 1, "flag": 1}))
    node_features = list(db.node_features.find())
    edges = list(db.all_edges_py.find(
        {},
        {"from": 1, "to": 1, "timeStamp": 1, "value": 1, "gasUsed": 1}
    ))

    return pd.DataFrame(nodes), pd.DataFrame(node_features), pd.DataFrame(edges)


# 数据合并与清洗（网页4数据处理思路）
def merge_node_data(nodes_df, features_df):
    # 合并时处理字段类型
    nodes_df = nodes_df.merge(features_df, left_on="address", right_on="Address")

    # 转换所有数值列为float（关键修复）
    numeric_cols = [
        'Sent_tnx', 'Received_tnx', 'Number_of_Created_Contracts',
        'Unique_Sent_To_Addresses20', 'Unique_Received_From_Addresses',
        'Min_Val_Sent', 'Max_Val_Sent', 'Avg_Val_Sent', 'Min_Value_Received',
        'Max_Value_Received', 'Avg_Value_Received5', 'Avg_Gas_Fee', 'Max_Gas_Fee',
        'Min_Gas_Fee', 'Total_Transactions', 'Total_Ether_Sent',
        'Total_Ether_Received', 'Total_Ether_Sent_Contracts',
        'Min_Value_Sent_To_Contract', 'Max_Value_Sent_To_Contract',
        'Avg_Value_Sent_To_Contract', 'Time_Diff_between_first_and_last(Mins)',
        'Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx'
    ]

    # 清洗数值列：移除引号并转换类型
    for col in numeric_cols:
        # 处理字符串类型的数值
        if nodes_df[col].dtype == object:
            nodes_df[col] = nodes_df[col].str.replace('"', '')  # 移除多余引号
            nodes_df[col] = pd.to_numeric(nodes_df[col], errors='coerce')

        # 填充空值（用列均值）
        nodes_df[col].fillna(nodes_df[col].mean(), inplace=True)

    # 标准化
    scaler = StandardScaler()
    nodes_df[numeric_cols] = scaler.fit_transform(nodes_df[numeric_cols])

    return nodes_df


nodes_df, features_df, edges_df = load_mongo_data()
merged_nodes = merge_node_data(nodes_df, features_df)


def build_node_features(merged_nodes):
    # 提取node_features中所有数值型特征（你的数据实际有34个）
    numeric_cols = [
        'Sent_tnx', 'Received_tnx', 'Number_of_Created_Contracts',
        'Unique_Sent_To_Addresses20', 'Unique_Received_From_Addresses',
        'Min_Val_Sent', 'Max_Val_Sent', 'Avg_Val_Sent', 'Min_Value_Received',
        'Max_Value_Received', 'Avg_Value_Received5', 'Avg_Gas_Fee', 'Max_Gas_Fee',
        'Min_Gas_Fee', 'Total_Transactions', 'Total_Ether_Sent',
        'Total_Ether_Received', 'Total_Ether_Sent_Contracts',
        'Min_Value_Sent_To_Contract', 'Max_Value_Sent_To_Contract',
        'Avg_Value_Sent_To_Contract', 'Time_Diff_between_first_and_last(Mins)',
        'Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx',
        'ERC20_Total_Ether_Received', 'ERC20_Total_Ether_Sent',
        'ERC20_Uniq_Sent_Addr', 'ERC20_Uniq_Rec_Addr',
        'ERC20_Avg_Val_Rec', 'ERC20_Avg_Val_Sent',
        'ERC20_Avg_Time_Between_Contract_Tnx',
        'ERC20_Avg_Time_Between_Sent_Tnx', 'ERC20_Avg_Time_Between_Rec_Tnx',
        'ERC20_Avg_gas_fee'
    ]
    X = merged_nodes[numeric_cols].values.astype(np.float32)
    return torch.FloatTensor(X)


node_features = build_node_features(merged_nodes)


def process_edges(edges_df, address_map):
    # 预处理数值字段（关键修复）
    numeric_edge_cols = ['value', 'timeStamp', 'gasUsed']

    # 转换数值类型
    for col in numeric_edge_cols:
        edges_df[col] = pd.to_numeric(edges_df[col], errors='coerce')

    # 过滤无效边（空值/异常值）
    edges_df = edges_df.dropna(subset=numeric_edge_cols)
    edges_df = edges_df[edges_df['value'] >= 0]  # 确保交易金额非负

    # 生成边类型（修复分箱逻辑）
    edges_df["value_type"] = pd.qcut(
        edges_df["value"],
        q=3,
        labels=[0, 1, 2],
        duplicates='drop'  # 处理重复分位点
    )

    edges_df["time_type"] = pd.qcut(
        edges_df["timeStamp"],
        q=3,
        labels=[0, 1, 2],
        duplicates='drop'
    )

    # 处理可能的空值
    edges_df = edges_df.dropna(subset=["value_type", "time_type"])

    # 合并边类型
    edge_type = (edges_df["value_type"].astype(int) + edges_df["time_type"].astype(int)) % 3

    # 地址映射
    edges_df["src_idx"] = edges_df["from"].map(address_map)
    edges_df["dst_idx"] = edges_df["to"].map(address_map)

    # 过滤无效地址
    valid_edges = edges_df.dropna(subset=["src_idx", "dst_idx"])

    return (
        torch.LongTensor(valid_edges["src_idx"].values.astype(int)),
        torch.LongTensor(valid_edges["dst_idx"].values.astype(int)),
        torch.LongTensor(edge_type.values.astype(int))
    )


# 创建地址映射字典
address_map = {addr: idx for idx, addr in enumerate(merged_nodes["address"])}
edge_src, edge_dst, edge_type = process_edges(edges_df, address_map)


def generate_labels_and_mask(merged_nodes):
    # 标签转换（网页6类型转换）
    labels = merged_nodes["flag"].map({1: 1, 2: 0}).values  # 1=欺诈，0=正常

    # 生成训练掩码（70%训练，30%测试）
    N = len(labels)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_indices = torch.randperm(N)[:int(N * 0.7)]
    train_mask[train_indices] = True

    return (
        torch.LongTensor(labels),
        train_mask
    )


labels, train_mask = generate_labels_and_mask(merged_nodes)


def validate_data(node_features, edge_src, edge_dst):
    # 检查维度一致性
    assert len(node_features) == len(labels), "节点数量不匹配"
    assert edge_src.shape == edge_dst.shape, "边列表不匹配"

    # 构建DGL图（参考DGL官方文档）
    g = dgl.graph((edge_src, edge_dst))
    g.ndata['feat'] = node_features
    g.ndata['label'] = labels
    g.ndata['train_mask'] = train_mask
    g.edata['etype'] = edge_type
    return g


data_graph = validate_data(node_features, edge_src, edge_dst)


print(f"节点数量: {data_graph.num_nodes()}")
print(f"边数量: {data_graph.num_edges()}")
print(f"点特征维度: {data_graph.ndata['feat'].shape[1]}")
print(f"边特征维度: {data_graph.edata['etype'].shape[1] if len(data_graph.edata['etype'].shape) > 1 else 1}")
