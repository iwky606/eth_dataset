import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
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
                if isinstance(val, str):
                    val = float(val) if 'E' not in val else float(val)
                else:
                    val = float(val)
            except:
                val = 0.0
            features.append(val)
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

        # 处理边特征
        value = float(edge.get('value', 0))
        gas = float(edge.get('gas', 0))
        timeStamp = float(edge.get('timeStamp', 0))
        edge_feats.append([value, gas, timeStamp])

    edge_scaler = StandardScaler()
    edge_features = torch.FloatTensor(edge_scaler.fit_transform(edge_feats))
    # 构建异构图
    g = dgl.heterograph({
        ('node', 'tx', 'node'): (torch.tensor(src), torch.tensor(dst))
    })

    # 分配节点特征
    g.nodes['node'].data['feat'] = node_features

    # 分配边特征
    g.edges['tx'].data['feat'] = edge_features

    # 转换为同构图
    g = dgl.to_homogeneous(g, ndata=['feat'], edata=['feat'])
    # 处理标签
    labels = torch.LongTensor([f if f in {0, 1} else -1 for f in flags])
    labeled_idx = torch.where(labels != -1)[0].tolist()
    lbls = labels[labeled_idx].numpy()

    # 划分训练集/验证集/测试集
    train_idx, test_idx = train_test_split(labeled_idx, test_size=0.2, stratify=lbls)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=lbls[train_idx])

    # 创建掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return g, labels, train_mask, val_mask, test_mask


class EdgeEnhancedGATConv(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, num_heads):
        super().__init__()
        # 添加维度校验
        assert out_feats % num_heads == 0, f"输出特征数{out_feats}必须能被注意力头数{num_heads}整除"
        self.num_heads = num_heads
        self.out_feats_per_head = out_feats // num_heads

        # 节点特征变换
        self.node_fc = nn.Linear(in_feats, out_feats)
        # 边特征变换
        self.edge_fc = nn.Linear(edge_feats, out_feats)

        # 注意力参数（调整为2D张量）
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 3 * self.out_feats_per_head)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.node_fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn, gain=gain)

    def edge_attention(self, edges):
        # 特征变换
        Wh_src = edges.src['Wh']  # (E, H, D)
        Wh_dst = edges.dst['Wh']  # (E, H, D)
        We = edges.data['We']  # (E, H, D)

        # 拼接特征并计算注意力
        combined = torch.cat([Wh_src, Wh_dst, We], dim=-1)
        e = (combined * self.attn).sum(dim=-1).unsqueeze(-1)
        return {'e': F.leaky_relu(e, 0.2)}

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # 节点特征投影
            Wh = self.node_fc(node_feats)
            Wh = Wh.view(-1, self.num_heads, self.out_feats_per_head)
            g.ndata['Wh'] = Wh

            # 边特征投影
            We = self.edge_fc(edge_feats)
            We = We.view(-1, self.num_heads, self.out_feats_per_head)
            g.edata['We'] = We

            # 计算注意力分数
            g.apply_edges(self.edge_attention)
            g.edata['a'] = dgl.nn.functional.edge_softmax(g, g.edata['e'])

            # 消息传递
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata['h']

    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        Wh = nodes.mailbox['Wh']  # (N, E, H, D)
        a = nodes.mailbox['a']  # (N, E, H, 1)
        h = torch.sum(a * Wh, dim=1)
        return {'h': h.reshape(-1, self.num_heads * self.out_feats_per_head)}


class GAT(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_size, num_classes):
        super().__init__()
        # 第一层使用4个注意力头
        self.conv1 = EdgeEnhancedGATConv(in_feats, hidden_size, edge_feats, num_heads=4)
        # 第二层使用1个注意力头以适应分类维度
        self.conv2 = EdgeEnhancedGATConv(hidden_size, num_classes, edge_feats, num_heads=1)

    def forward(self, g, node_feats, edge_feats):
        h = self.conv1(g, node_feats, edge_feats)
        h = F.elu(h)
        h = self.conv2(g, h, edge_feats)
        return h


def main():
    g, labels, train_mask, val_mask, test_mask = load_and_process_data()

    # 模型参数调整
    in_feats = g.ndata['feat'].shape[1]
    edge_feats = g.edata['feat'].shape[1]
    hidden_size = 64
    num_classes = 2

    # 初始化模型
    model = GAT(in_feats, edge_feats, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环（添加早停机制）
    best_val_acc = 0
    patience = 10
    for epoch in range(200):
        model.train()
        logits = model(g, g.ndata['feat'], g.edata['feat'])
        loss = criterion(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            logits = model(g, g.ndata['feat'], g.edata['feat'])
            _, preds = torch.max(logits, 1)
            train_acc = (preds[train_mask] == labels[train_mask]).float().mean()
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean()

            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
