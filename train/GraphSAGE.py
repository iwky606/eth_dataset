import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def connect_to_mongodb():
    from db_connection.mongo_client import db as eth_db
    return eth_db


def convert_value(val):
    """处理各种数值格式转换为浮点数"""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        if isinstance(val, str):
            if 'E' in val or 'e' in val:
                return float(val)
            if val == '':  # 处理空字符串
                return 0.0
            return float(val.split('E')[0])  # 处理类似"0E-18"的情况
        return float(val)
    except:
        return 0.0


def load_and_process_data():
    eth_db = connect_to_mongodb()

    # 加载节点数据
    nodes = list(eth_db.train_nodes_v2.find({}, {'address': 1, 'flag': 1}))
    address_to_id = {node['address']: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    flags = [node['flag'] for node in nodes]

    # 定义节点特征处理规范
    feature_spec = [
        ('Sent_tnx', 0),
        ('Received_tnx', 0),
        ('Number_of_Created_Contracts', 0),
        ('Unique_Sent_To_Addresses20', 0),
        ('Unique_Received_From_Addresses', 0),
        ('Min_Val_Sent', 0.0),
        ('Max_Val_Sent', 0.0),
        ('Avg_Val_Sent', 0.0),
        ('Min_Value_Received', 0.0),
        ('Max_Value_Received', 0.0),
        ('Avg_Value_Received5', 0.0),
        ('Avg_Gas_Fee', 0.0),
        ('Max_Gas_Fee', 0.0),
        ('Min_Gas_Fee', 0.0),
        ('Total_Transactions', 0),
        ('Total_Ether_Sent', 0.0),
        ('Total_Ether_Received', 0.0)
    ]

    # 处理节点特征
    all_features = []
    for node in nodes:
        address = node['address']
        feat_doc = eth_db.node_features.find_one({'Address': address}) or {}
        features = []
        for key, default in feature_spec:
            val = feat_doc.get(key, default)
            features.append(convert_value(val))
        all_features.append(features)

    # 标准化节点特征
    scaler = StandardScaler()
    node_features = torch.FloatTensor(scaler.fit_transform(all_features))

    # 处理边数据
    edges = list(eth_db.all_edges_py.find({}, {'from': 1, 'to': 1, 'value': 1, 'gas': 1, 'timeStamp': 1}))
    src, dst, edge_feats = [], [], []
    for edge in edges:
        if (from_addr := edge['from']) in address_to_id and (to_addr := edge['to']) in address_to_id:
            src.append(address_to_id[from_addr])
            dst.append(address_to_id[to_addr])
            edge_feats.append([
                convert_value(edge.get('value', 0)),
                convert_value(edge.get('gas', 0)),
                convert_value(edge.get('timeStamp', 0))
            ])

    # 标准化边特征
    edge_scaler = StandardScaler()
    edge_features = torch.FloatTensor(edge_scaler.fit_transform(edge_feats))

    # 构建图结构
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
    g.ndata['feat'] = node_features
    g.edata['feat'] = edge_features

    # 创建掩码
    labels = torch.LongTensor([f if f in {0, 1} else -1 for f in flags])
    labeled_idx = torch.where(labels != -1)[0].tolist()
    lbls = labels[labeled_idx].numpy()

    train_idx, test_idx = train_test_split(labeled_idx, test_size=0.2, stratify=lbls)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=lbls[train_idx])

    masks = {
        'train': torch.zeros(num_nodes, dtype=torch.bool),
        'val': torch.zeros(num_nodes, dtype=torch.bool),
        'test': torch.zeros(num_nodes, dtype=torch.bool)
    }
    masks['train'][train_idx] = True
    masks['val'][val_idx] = True
    masks['test'][test_idx] = True

    return g, labels, masks


class EdgeEnhancedSAGE(nn.Module):
    """修正后的支持边特征的GraphSAGE实现"""

    def __init__(self, in_feats, edge_feats, hid_feats, out_feats):
        super().__init__()
        # 边特征处理层
        self.edge_encoder = nn.Linear(edge_feats, in_feats)
        # 邻居聚合层（调整输入维度）
        self.conv1 = dgl.nn.SAGEConv(in_feats * 2, hid_feats, 'mean')  # 输入维度翻倍
        self.conv2 = dgl.nn.SAGEConv(hid_feats, out_feats, 'mean')

    def forward(self, g, nfeat, efeat):
        # 边特征编码
        efeat = F.relu(self.edge_encoder(efeat))

        # 自定义消息传递
        with g.local_scope():
            g.ndata['h'] = nfeat
            g.edata['e'] = efeat

            # 消息传递：节点特征 + 边特征
            g.update_all(
                message_func=fn.u_mul_e('h', 'e', 'm'),  # 节点特征与边特征相乘
                reduce_func=fn.mean('m', 'h_neigh')
            )

            # 拼接中心节点特征和聚合后的邻居特征
            h = torch.cat([nfeat, g.ndata['h_neigh']], dim=1)

            # 通过SAGE层
            h = F.relu(self.conv1(g, h))
            h = self.conv2(g, h)

        return h


def calculate_metrics(preds, labels, mask):
    y_true = labels[mask]
    y_pred = preds[mask]

    # 计算混淆矩阵
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    # 计算各项指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    g, labels, masks = load_and_process_data()
    g = g.to(device)
    labels = labels.to(device)
    train_mask = masks['train'].to(device)
    val_mask = masks['val'].to(device)

    # 模型参数`
    in_feats = g.ndata['feat'].shape[1]
    edge_feats = g.edata['feat'].shape[1]
    model = EdgeEnhancedSAGE(in_feats, edge_feats, 64, 2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(100):
        model.train()
        logits = model(g, g.ndata['feat'], g.edata['feat'])
        loss = criterion(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证评估
        model.eval()
        with torch.no_grad():
            logits = model(g, g.ndata['feat'], g.edata['feat'])
            _, preds = torch.max(logits, 1)

            # 计算训练集指标
            train_prec, train_rec, train_f1 = calculate_metrics(preds, labels, train_mask)
            train_acc = (preds[train_mask] == labels[train_mask]).float().mean()

            # 计算验证集指标
            val_prec, val_rec, val_f1 = calculate_metrics(preds, labels, val_mask)
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
            print(f"Train Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
            print(f"Val   Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
            print("-" * 60)


if __name__ == "__main__":
    main()
