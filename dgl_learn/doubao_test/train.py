import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.data import CoraGraphDataset


# 定义 RGCN 模型
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_rels):
        super().__init__()
        self.conv1 = dglnn.RelGraphConv(in_feats, hid_feats, num_rels)
        self.conv2 = dglnn.RelGraphConv(hid_feats, out_feats, num_rels)

    def forward(self, g, inputs, etypes):
        h = self.conv1(g, inputs, etypes)
        h = F.relu(h)
        h = self.conv2(g, h, etypes)
        return h


# 训练函数
def train(model, g, node_features, edge_types, labels, train_mask, optimizer, loss_fn):
    model.train()
    logits = model(g, node_features, edge_types)
    logp = F.log_softmax(logits, 1)
    loss = loss_fn(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# 评估函数
def evaluate(model, g, node_features, edge_types, labels, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, node_features, edge_types)
        logp = F.log_softmax(logits, 1)
        _, indices = torch.max(logp[test_mask], dim=1)
        correct = torch.sum(indices == labels[test_mask])
        return correct.item() * 1.0 / len(labels[test_mask])


# 主函数
def main():
    # 加载 Cora 数据集
    dataset = CoraGraphDataset()
    g = dataset[0]

    # 假设节点特征和边类型已经定义
    node_features = g.ndata['feat']
    # 由于 Cora 是无向图，这里简单将边类型设为 0
    edge_types = torch.zeros(g.num_edges(), dtype=torch.long)
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']

    # 初始化模型
    in_feats = node_features.shape[1]
    hid_feats = 16
    out_feats = dataset.num_classes
    num_rels = 1  # 因为只有一种边类型
    model = RGCN(in_feats, hid_feats, out_feats, num_rels)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.NLLLoss()

    # 训练模型
    for epoch in range(100):
        loss = train(model, g, node_features, edge_types, labels, train_mask, optimizer, loss_fn)
        if epoch % 10 == 0:
            acc = evaluate(model, g, node_features, edge_types, labels, test_mask)
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')


if __name__ == "__main__":
    main()
