import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from pymongo import MongoClient
from tqdm import tqdm
import logging
import warnings
from pymongo.errors import PyMongoError

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)


def connect_to_mongodb():
    from dao.mongo_client import db as eth_db
    return eth_db


def load_train_nodes(db):
    """加载训练节点（带数据验证）"""
    try:
        col = db.get_collection('train_nodes_v2')
        if col is None:
            raise ValueError("训练节点集合不存在")

        addresses = []
        labels = []
        valid_count = 0

        for doc in col.find(projection={"_id": 0, "address": 1, "flag": 1}):
            addr = doc.get('address', '').strip().lower()
            flag = doc.get('flag', 2)

            if len(addr) == 42 and addr.startswith('0x'):
                addresses.append(addr)
                labels.append(int(flag))
                valid_count += 1

        logging.info(f"有效训练节点数量: {valid_count}")
        return addresses, np.array(labels)

    except PyMongoError as e:
        logging.error(f"加载训练节点失败: {str(e)}")
        exit(1)


def safe_float_convert(value):
    """增强型数值转换"""
    try:
        if isinstance(value, str):
            value = value.strip().lower()
            if not value or value in ['nan', 'null', 'none']:
                return 0.0
            if 'e' in value:
                base, exp = value.split('e', 1)
                return float(base) * 10 ** float(exp)
            return float(value)
        return float(value)
    except:
        return 0.0


def load_node_features(db, addresses):
    """加载节点特征（修复空特征问题）"""
    try:
        col = db.get_collection('node_features')
        if col is None:
            raise ValueError("节点特征集合不存在")

        # 获取有效特征字段
        sample = col.find_one(projection={"_id": 0, "Address": 1})
        if not sample:
            raise ValueError("特征集合为空")

        feature_keys = [k for k in sample.keys() if k not in ['Address'] and not k.startswith('_')]
        if not feature_keys:
            raise ValueError("没有找到有效特征字段")

        logging.info(f"识别到 {len(feature_keys)} 个特征字段")

        # 初始化特征矩阵
        address_map = {addr.lower(): idx for idx, addr in enumerate(addresses)}
        feature_matrix = np.zeros((len(addresses), len(feature_keys)), dtype=np.float32)
        matched_count = 0

        # 批量处理特征
        batch_size = 1000
        total_docs = col.estimated_document_count()

        with tqdm(total=total_docs, desc="加载节点特征", unit="doc") as pbar:
            for batch in col.find().batch_size(batch_size):
                for doc in batch:
                    addr = doc.get('Address', '').lower()
                    if addr not in address_map:
                        continue

                    idx = address_map[addr]
                    features = []
                    for key in feature_keys:
                        raw_value = doc.get(key, 0)
                        features.append(safe_float_convert(raw_value))

                    # 特征有效性检查
                    if not np.any(features):  # 全零特征跳过
                        continue

                    feature_matrix[idx] = np.clip(
                        np.nan_to_num(features, nan=0.0, posinf=1e30, neginf=-1e30),
                        -1e30, 1e30
                    )
                    matched_count += 1
                pbar.update(len(batch))

        logging.info(f"特征匹配成功数: {matched_count}/{len(addresses)}")
        return feature_matrix, feature_keys

    except PyMongoError as e:
        logging.error(f"加载特征失败: {str(e)}")
        exit(1)


def build_graph(db, addresses):
    """构建图结构（带特征验证）"""
    try:
        col = db.get_collection('all_edges_py')
        if col is None:
            raise ValueError("边集合不存在")

        address_map = {addr.lower(): idx for idx, addr in enumerate(addresses)}
        src_nodes = []
        dst_nodes = []
        edge_features = []
        valid_edges = 0

        EDGE_FEATURES = ['value', 'gas', 'timeStamp']

        with tqdm(desc="构建图结构", unit="edge") as pbar:
            for doc in col.find(projection=["from", "to"] + EDGE_FEATURES):
                from_addr = doc.get('from', '').lower()
                to_addr = doc.get('to', '').lower()

                if from_addr in address_map and to_addr in address_map:
                    src = address_map[from_addr]
                    dst = address_map[to_addr]

                    # 处理边特征
                    feat = [safe_float_convert(doc.get(k, 0)) for k in EDGE_FEATURES]
                    feat = np.clip(np.nan_to_num(feat), -1e30, 1e30)

                    src_nodes.append(src)
                    dst_nodes.append(dst)
                    edge_features.append(feat)
                    valid_edges += 1
                pbar.update(1)

        # 创建图结构
        g = dgl.graph((src_nodes, dst_nodes))
        g = dgl.add_self_loop(g)

        if edge_features:
            g.edata['feat'] = torch.FloatTensor(np.array(edge_features, dtype=np.float32))

        logging.info(f"图结构构建完成: {g.num_nodes()} 节点, {g.num_edges()} 边")
        return g

    except PyMongoError as e:
        logging.error(f"构建图失败: {str(e)}")
        exit(1)


class GATModel(nn.Module):
    """增强型GAT模型"""

    def __init__(self, in_feats, hidden_size=128, num_heads=4, num_classes=3):
        super().__init__()
        self.conv1 = GATConv(
            in_feats,
            hidden_size,
            num_heads=num_heads,
            feat_drop=0.4,
            attn_drop=0.4,
            residual=True,
            allow_zero_in_degree=True
        )
        self.conv2 = GATConv(
            hidden_size * num_heads,
            num_classes,
            num_heads=1,
            feat_drop=0.4,
            residual=True,
            allow_zero_in_degree=True
        )
        self.bn = nn.BatchNorm1d(hidden_size * num_heads)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = h.view(h.size(0), -1)
        h = self.bn(h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h.squeeze(1)


def main():
    # 初始化
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 数据库连接
        db = connect_to_mongodb()

        # 加载数据
        addresses, labels = load_train_nodes(db)
        features, feature_names = load_node_features(db, addresses)

        # 特征维度验证
        if features.shape[1] == 0:
            raise ValueError("特征矩阵为空，请检查特征集合数据")

        # 数据预处理
        logging.info("特征预处理中...")
        valid_mask = ~np.all(features == 0, axis=1)
        features = features[valid_mask]
        labels = labels[valid_mask]
        addresses = [addr for addr, mask in zip(addresses, valid_mask) if mask]

        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features)

        # 构建图
        g = build_graph(db, addresses)
        g.ndata['feat'] = torch.FloatTensor(scaled_features)
        g.ndata['label'] = torch.LongTensor(labels)

        # 划分数据集
        train_mask = (labels != 2)
        indices = np.where(train_mask)[0]
        if len(indices) == 0:
            raise ValueError("无有效训练样本")

        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels[indices],
            random_state=42
        )

        # 初始化模型
        model = GATModel(in_feats=scaled_features.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 训练循环
        best_f1 = 0
        for epoch in range(200):
            model.train()
            logits = model(g, g.ndata['feat'])
            loss = F.cross_entropy(logits[train_idx], g.ndata['label'][train_idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # 验证
            if epoch % 5 == 0:
                with torch.no_grad():
                    model.eval()
                    pred = logits.argmax(1)
                    train_acc = accuracy_score(labels[train_idx], pred[train_idx].numpy())
                    val_acc = accuracy_score(labels[val_idx], pred[val_idx].numpy())
                    val_f1 = f1_score(labels[val_idx], pred[val_idx].numpy(), average='macro')

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        torch.save(model.state_dict(), 'best_model.pth')

                    logging.info(
                        f"Epoch {epoch:03d} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Train Acc: {train_acc:.4f} | "
                        f"Val F1: {val_f1:.4f}"
                    )

        logging.info(f"训练完成，最佳验证F1: {best_f1:.4f}")

    except Exception as e:
        logging.error(f"运行失败: {str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()