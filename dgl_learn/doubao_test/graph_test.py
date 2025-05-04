import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from pymongo import MongoClient
import numpy as np
import math

# MongoDB连接配置
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "ethereum_db"  # 请根据实际数据库名称修改


def connect_to_mongodb():
    """创建MongoDB客户端并保持连接"""
    client = MongoClient(MONGODB_URI)
    return client[DATABASE_NAME]


def read_nodes_data(db):
    """读取节点标签数据"""
    nodes_collection = db["train_nodes_v2"]
    nodes = list(nodes_collection.find({}, {"address": 1, "flag": 1}))  # 只获取必要字段
    addresses = [node["address"] for node in nodes]
    labels = [node["flag"] for node in nodes]
    return addresses, labels


def read_node_features_data(db):
    """读取并处理节点特征数据"""
    features_collection = db["node_features"]
    features_cursor = features_collection.find()

    feature_dict = {}
    for feature_doc in features_cursor:
        address = feature_doc.pop("Address")  # 提取地址作为键
        processed_features = {}

        for key, value in feature_doc.items():
            # 处理科学计数法字符串和非数字值
            try:
                if isinstance(value, str):
                    # 处理特殊科学计数法格式（如"0E-18"转换为0）
                    value = value.replace("E-", "e-").replace("E+", "e+")
                    numeric_value = float(value)
                else:
                    numeric_value = float(value)
            except (ValueError, TypeError):
                # 无法转换的非数字值（如空字符串、文本）设为0
                numeric_value = 0.0

            # 处理无穷大和NaN
            if math.isinf(numeric_value) or math.isnan(numeric_value):
                numeric_value = 0.0

            processed_features[key] = numeric_value

        feature_dict[address] = processed_features
    return feature_dict


def read_edges_data(db):
    """读取边数据并提取特征"""
    edges_collection = db["all_edges_py"]
    edges_cursor = edges_collection.find(
        {},
        {"from": 1, "to": 1, "value": 1, "gas": 1, "timeStamp": 1}
    )

    edge_list = []
    edge_features = []
    for edge_doc in edges_cursor:
        from_addr = edge_doc["from"]
        to_addr = edge_doc["to"]
        edge_list.append((from_addr, to_addr))

        # 提取边特征并转换为浮点数
        value = float(edge_doc["value"]) if edge_doc["value"] else 0.0
        gas = float(edge_doc["gas"]) if edge_doc["gas"] else 0.0
        timestamp = float(edge_doc["timeStamp"]) if edge_doc["timeStamp"] else 0.0
        edge_features.append([value, gas, timestamp])

    return edge_list, edge_features


def create_dgl_graph(addresses, labels, edge_list, edge_features, node_features):
    """创建DGL异质图并加载数据"""
    # 建立地址到节点ID的映射
    address_to_id = {addr: idx for idx, addr in enumerate(addresses)}

    # 处理节点特征矩阵
    node_count = len(addresses)
    feat_dim = len(next(iter(node_features.values()))) if node_features else 0
    node_feat_matrix = np.zeros((node_count, feat_dim))

    for idx, addr in enumerate(addresses):
        features = node_features.get(addr, {})
        # 按固定顺序排列特征（需根据实际特征列表调整顺序）
        # 这里假设特征顺序固定，实际使用时应根据特征键排序处理
        feat_vector = [features[key] for key in sorted(features.keys())]
        node_feat_matrix[idx] = feat_vector

    # 创建图结构
    src = [address_to_id[from_addr] for from_addr, _ in edge_list]
    dst = [address_to_id[to_addr] for _, to_addr in edge_list]
    g = dgl.graph((src, dst))

    # 加载节点和边特征
    g.ndata["feat"] = torch.FloatTensor(node_feat_matrix)
    g.edata["feat"] = torch.FloatTensor(edge_features)
    g.ndata["label"] = torch.LongTensor(labels)

    return g


class RGCN(nn.Module):
    """带RelGraphConv的RGCN模型"""

    def __init__(self, in_feat, hidden_feat, out_feat, num_rels):
        super(RGCN, self).__init__()
        self.conv1 = dgl.nn.RelGraphConv(
            in_feat, hidden_feat, num_rels, "basis", num_bases=16
        )
        self.conv2 = dgl.nn.RelGraphConv(
            hidden_feat, out_feat, num_rels, "basis", num_bases=16
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, node_feat, edge_feat):
        # 转换边特征为关系类型（假设边类型统一，实际需根据异质边处理）
        # 这里假设所有边为同一种类型，如需处理异质边需扩展rel参数
        rel = torch.zeros(g.number_of_edges(), dtype=torch.long)  # 单一关系类型示例
        h = self.conv1(g, node_feat, rel, edge_feat=edge_feat)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h, rel, edge_feat=edge_feat)
        return h


def train():
    # 连接数据库
    db = connect_to_mongodb()

    # 读取数据
    addresses, labels = read_nodes_data(db)
    node_features = read_node_features_data(db)
    edge_list, edge_features = read_edges_data(db)

    # 处理特征缺失情况
    if not node_features:
        raise ValueError("No node features found")

    # 创建图
    g = create_dgl_graph(addresses, labels, edge_list, edge_features, node_features)

    # 模型参数
    in_feat = g.ndata["feat"].shape[1]
    hidden_feat = 128
    out_feat = 3  # 标签有0/1/2三种类型
    num_rels = 1  # 假设单一边类型，实际需根据边类型数量调整

    model = RGCN(in_feat, hidden_feat, out_feat, num_rels)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(g, g.ndata["feat"], g.edata["feat"])
        loss = criterion(logits, g.ndata["label"])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()