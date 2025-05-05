import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def connect_to_mongodb():
    from db_connection.mongo_client import db as eth_db
    return eth_db


def load_and_process_data():
    eth_db = connect_to_mongodb()

    # 加载节点数据
    nodes = list(eth_db.train_nodes_v2.find({}, {'address': 1, 'flag': 1}))
    address_to_id = {node['address']: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    flags = [node['flag'] for node in nodes]

    # 加载节点特征
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
        address = node['address']
        feat_doc = eth_db.node_features.find_one({'Address': address}, {'_id': 0, 'Address': 0})
        features = []
        for key in feature_keys:
            val = feat_doc.get(key, 0) if feat_doc else 0
            try:
                features.append(float(val) if isinstance(val, str) else val)
            except:
                features.append(0.0)
        all_features.append(features)

    scaler = StandardScaler()
    node_features = torch.FloatTensor(scaler.fit_transform(all_features))

    # 加载边数据
    edges = list(eth_db.all_edges_py.find({}, {'from': 1, 'to': 1, 'value': 1, 'gas': 1, 'timeStamp': 1}))
    src, dst, edge_feats = [], [], []
    for edge in edges:
        from_addr = edge['from']
        to_addr = edge['to']
        if from_addr not in address_to_id or to_addr not in address_to_id:
            continue
        src.append(address_to_id[from_addr])
        dst.append(address_to_id[to_addr])
        edge_feats.append([
            float(edge.get('value', 0)),
            float(edge.get('gas', 0)),
            float(edge.get('timeStamp', 0))
        ])

    edge_scaler = StandardScaler()
    edge_features = torch.FloatTensor(edge_scaler.fit_transform(edge_feats))

    # 构建异质图
    g = dgl.heterograph({
        ('node', 'tx', 'node'): (torch.tensor(src), torch.tensor(dst))
    })
    g.nodes['node'].data['feat'] = node_features
    g.edges['tx'].data['feat'] = edge_features
    g.edges['tx'].data['rel_type'] = torch.zeros(len(src), dtype=torch.int64)  # 单一边类型

    labels = torch.LongTensor([f if f in {0, 1} else -1 for f in flags])
    labeled_idx = torch.where(labels != -1)[0].tolist()
    lbls = labels[labeled_idx].numpy()

    train_idx, test_idx = train_test_split(labeled_idx, test_size=0.2, stratify=lbls)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=lbls[train_idx])

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return g, labels, train_mask, val_mask, test_mask


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_rels):
        super().__init__()
        self.conv1 = dgl.nn.RelGraphConv(in_feats, hid_feats, num_rels, regularizer='basis')
        self.conv2 = dgl.nn.RelGraphConv(hid_feats, out_feats, num_rels, regularizer='basis')

    def forward(self, g, x):
        # 获取边类型数据
        etype = g.edges['tx'].data['rel_type']
        h = self.conv1(g, x, etype)
        h = F.relu(h)
        h = self.conv2(g, h, etype)
        return h


def main():
    g, labels, train_mask, val_mask, test_mask = load_and_process_data()

    in_feats = g.nodes['node'].data['feat'].shape[1]
    hid_feats = 64
    out_feats = 2
    num_rels = 1  # 单一边类型

    model = RGCN(in_feats, hid_feats, out_feats, num_rels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    node_features = g.nodes['node'].data['feat']

    for epoch in range(100):
        model.train()
        logits = model(g, node_features)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(g, node_features)
            _, preds = torch.max(logits, 1)
            train_acc = (preds[train_mask] == labels[train_mask]).float().mean()
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean()

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
