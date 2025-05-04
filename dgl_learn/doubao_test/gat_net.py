import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dgl.nn import GATConv


def connect_to_mongodb():
    from dao.mongo_client import db as eth_db
    return eth_db


def convert_scientific_notation(val):
    """处理科学计数法和异常数值"""
    if isinstance(val, str):
        if 'E' in val or 'e' in val:
            try:
                return float(val)
            except:
                return 0.0
        elif val == '':
            return 0.0
    return float(val) if not isinstance(val, (int, float)) else val


def load_graph_data():
    """从MongoDB加载并处理图数据"""
    eth_db = connect_to_mongodb()

    # 节点数据处理
    nodes = list(eth_db.train_nodes_v2.find({}, {'address': 1, 'flag': 1}))
    address_to_id = {node['address']: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # 节点特征处理
    feature_keys = [
        'Sent_tnx', 'Received_tnx', 'Number_of_Created_Contracts',
        'Unique_Sent_To_Addresses20', 'Unique_Received_From_Addresses',
        'Min_Val_Sent', 'Max_Val_Sent', 'Avg_Val_Sent',
        'Min_Value_Received', 'Max_Value_Received', 'Avg_Value_Received5',
        'Avg_Gas_Fee', 'Max_Gas_Fee', 'Min_Gas_Fee',
        'Total_Transactions', 'Total_Ether_Sent', 'Total_Ether_Received'
    ]

    all_features = []
    for node in nodes:
        feat_doc = eth_db.node_features.find_one(
            {'Address': node['address']},
            {'_id': 0, 'Address': 0}
        ) or {}

        features = []
        for key in feature_keys:
            val = feat_doc.get(key, 0)
            features.append(convert_scientific_notation(val))
        all_features.append(features)

    # 边特征处理（网页12、13的图数据处理方案）
    edges = list(eth_db.all_edges_py.find(
        {},
        {'from': 1, 'to': 1, 'value': 1, 'gas': 1, 'timeStamp': 1}
    ))

    src, dst, edge_feats = [], [], []
    for edge in edges:
        if edge['from'] in address_to_id and edge['to'] in address_to_id:
            src.append(address_to_id[edge['from']])
            dst.append(address_to_id[edge['to']])
            edge_feats.append([
                convert_scientific_notation(edge.get('value', 0)),
                convert_scientific_notation(edge.get('gas', 0)),
                convert_scientific_notation(edge.get('timeStamp', 0))
            ])

    # 特征标准化（网页4、8的标准化方案）
    node_scaler = StandardScaler()
    edge_scaler = StandardScaler()
    node_features = torch.FloatTensor(node_scaler.fit_transform(all_features))
    edge_features = torch.FloatTensor(edge_scaler.fit_transform(edge_feats))

    # 构建DGL图（网页13的图构建方法）
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
    g.ndata['feat'] = node_features
    g.edata['feat'] = edge_features

    # 标签处理
    labels = torch.LongTensor([n['flag'] if n['flag'] in {0, 1} else -1 for n in nodes])

    # 构建DGL图后添加自环
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)  # 新增自环
    g.ndata['feat'] = node_features
    g.edata['feat'] = edge_features

    return g, labels, edge_features


class EdgeEnhancedGATConv(nn.Module):
    """改进支持零入度的GAT层"""
    def __init__(self, in_feats, out_feats, num_heads, edge_feat_size):
        super().__init__()
        self.num_heads = num_heads
        self.edge_proj = nn.Linear(edge_feat_size, in_feats)
        # 添加allow_zero_in_degree参数
        self.gat_conv = GATConv(in_feats, out_feats, num_heads,
                              allow_zero_in_degree=True)  # 关键修改

    def forward(self, g, node_feats, edge_feats):
        edge_emb = self.edge_proj(edge_feats)
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_emb
            # 添加自环避免孤立节点
            g = dgl.add_self_loop(g)  # 新增自环处理
            return self.gat_conv(g, node_feats)

class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, edge_feat_size, num_heads=4):
        super().__init__()
        # 确保各层输出维度匹配
        self.conv1 = EdgeEnhancedGATConv(in_feats, hidden_size, num_heads, edge_feat_size)
        # 第二层输入维度需要匹配第一层输出
        self.conv2 = EdgeEnhancedGATConv(
            hidden_size * num_heads,  # 修正维度匹配
            out_feats,
            1,
            edge_feat_size
        )

    def forward(self, g, node_feats, edge_feats):
        h = self.conv1(g, node_feats, edge_feats).flatten(1)
        h = F.elu(h)
        h = self.conv2(g, h, edge_feats).mean(1)
        return h


def train_model():
    # 数据准备
    g, labels, edge_feats = load_graph_data()
    node_feats = g.ndata['feat']

    # 划分训练集（网页3、9的数据划分方案）
    labeled_idx = torch.where(labels != -1)[0].numpy()
    lbls = labels[labeled_idx].numpy()

    train_idx, test_idx = train_test_split(labeled_idx, test_size=0.2, stratify=lbls)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=lbls[train_idx])

    # 训练配置（网页4、8的训练参数）
    model = GATModel(
        in_feats=node_feats.shape[1],
        hidden_size=64,
        out_feats=2,
        edge_feat_size=edge_feats.shape[1]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环（网页4、7的训练流程）
    for epoch in range(200):
        model.train()
        logits = model(g, node_feats, edge_feats)
        loss = criterion(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证评估
        model.eval()
        with torch.no_grad():
            logits = model(g, node_feats, edge_feats)
            _, preds = torch.max(logits, 1)

            train_acc = (preds[train_idx] == labels[train_idx]).float().mean()
            val_acc = (preds[val_idx] == labels[val_idx]).float().mean()

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    train_model()