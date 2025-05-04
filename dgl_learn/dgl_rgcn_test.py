import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from sklearn.metrics import f1_score, roc_auc_score
from dgl_learn.dgl_graph_test_from_mongo import data_graph, edge_src, edge_dst

N = data_graph.num_nodes()
E = data_graph.num_edges()
# 假设已有以下数据（需替换为实际数据）
node_features = data_graph.ndata['feat']
edge_type = data_graph.edata['etype']
labels = data_graph.ndata['label']
train_mask = data_graph.ndata['train_mask']

# 创建图结构
g = dgl.graph((edge_src, edge_dst), num_nodes=N)
g.ndata['feat'] = node_features
g.edata['etype'] = edge_type  # 边类型作为RGCN的输入
g.ndata['label'] = labels
g.ndata['train_mask'] = train_mask

print(f"节点数量: {g.num_nodes()}")
print(f"边数量: {g.num_edges()}")
print(f"点特征维度: {g.ndata['feat'].shape[1]}")
print(f"边特征维度: {g.edata['etype'].shape[1] if len(g.edata['etype'].shape) > 1 else 1}")
num_rels = int(g.edata['etype'].max().item()) + 1

print("节点特征是否有NaN:", torch.isnan(node_features).any())
print("节点特征最大值:", node_features.max())
print("边类型是否有NaN:", torch.isnan(edge_type).any())
print("边类型最大值:", edge_type.max())




class RGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_rels):
        super().__init__()
        self.conv1 = RelGraphConv(in_dim, hid_dim, num_rels, regularizer='basis', num_bases=4)
        self.conv2 = RelGraphConv(hid_dim, out_dim, num_rels, regularizer='basis', num_bases=4)

    def forward(self, g, feat, etype):
        h = self.conv1(g, feat, etype)
        h = F.relu(h)
        h = self.conv2(g, h, etype)
        return h


model = RGCN(in_dim=g.ndata['feat'].shape[1], hid_dim=16, out_dim=2, num_rels=num_rels)
print("模型参数范围:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.min():.4f} ~ {param.data.max():.4f}")

print("=====[MODEL FINISH]=====")


def train(model, g, epochs=100, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        logits = model(g, g.ndata['feat'], g.edata['etype'])
        loss = F.cross_entropy(logits[g.ndata['train_mask']],
                               g.ndata['label'][g.ndata['train_mask']])

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证评估
        with torch.no_grad():
            model.eval()
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(1)

            # 计算训练集指标
            train_f1 = f1_score(g.ndata['label'][train_mask].numpy(),
                                preds[train_mask].numpy())

            # 计算未标注节点的预测分布
            unknown_probs = probs[~train_mask]

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Train F1: {train_f1:.4f}")

    return model


print("=====[MODEL TRAIN]=====")
model = train(model, g)
print("=====[MODEL TRAIN DONE]=====")


def evaluate(model, g):
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'], g.edata['etype'])
        probs = F.softmax(logits, dim=1)

    # 获取所有预测结果
    pred_labels = probs.argmax(1).numpy()
    true_labels = g.ndata['label'].numpy()

    # 分离已知/未知节点
    known_idx = g.ndata['train_mask'].numpy()
    unknown_idx = ~known_idx

    # 计算指标
    print(f"Known Nodes AUC: {roc_auc_score(true_labels[known_idx], pred_labels[known_idx])}")
    print(f"Unknown Nodes Predictions:\n{probs[unknown_idx].numpy()[:5]}")


print("=====[MODEL EVALUATE]=====")
evaluate(model, g)
print("=====[MODEL EVALUATE DONE]=====")
