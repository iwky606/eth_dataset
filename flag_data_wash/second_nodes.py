import pymongo
import numpy as np
from tqdm import tqdm
from dao.mongo_client import client, ETH_DATASET


class Trans2VecNeighborGenerator:
    def __init__(self, db_name=ETH_DATASET, alpha=0.5):
        self.client = client
        self.db = self.client[db_name]
        self.alpha = alpha
        self.graph = {}

    def build_transaction_graph(self):
        """构建交易图结构并计算转移概率"""
        # 获取所有有效地址
        valid_addrs = {doc['address'] for doc in self.db.valid_address.find({'flag': {'$in': [1, 2]}})}

        # 临时存储交易数据 {from: {to: (total_amount, max_timestamp)}}
        adj_dict = {}

        # 遍历所有交易
        for tx in tqdm(self.db.flag_transaction.find(), desc="Processing transactions"):
            from_addr = tx.get('from')
            to_addr = tx.get('to')

            if not from_addr or not to_addr or from_addr not in valid_addrs:
                continue

            value = int(tx.get('value', 0))
            timestamp = int(tx.get('timeStamp', 0))

            # 更新邻接表
            if from_addr not in adj_dict:
                adj_dict[from_addr] = {}

            if to_addr not in adj_dict[from_addr]:
                adj_dict[from_addr][to_addr] = {'amount': 0, 'timestamp': 0}

            adj_dict[from_addr][to_addr]['amount'] += value
            adj_dict[from_addr][to_addr]['timestamp'] = max(
                adj_dict[from_addr][to_addr]['timestamp'], timestamp
            )

        # 计算转移概率
        for from_addr in tqdm(adj_dict, desc="Calculating probabilities"):
            edges = adj_dict[from_addr]
            to_nodes = list(edges.keys())

            # 提取金额和时间戳
            amounts = [edges[to]['amount'] for to in to_nodes]
            timestamps = [edges[to]['timestamp'] for to in to_nodes]

            # 处理时间步
            sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
            time_steps = {i: idx + 1 for idx, i in enumerate(sorted_indices)}

            # 计算概率分量
            total_amount = sum(amounts)
            pa = [a / total_amount if total_amount != 0 else 1 / len(amounts) for a in amounts]
            pt = [time_steps[i] / sum(time_steps.values()) for i in range(len(to_nodes))]

            # 合并概率
            probabilities = [
                (pa[i] ** self.alpha) * (pt[i] ** (1 - self.alpha))
                for i in range(len(to_nodes))
            ]

            # 归一化
            prob_sum = sum(probabilities)
            self.graph[from_addr] = {
                'nodes': to_nodes,
                'probs': [p / prob_sum for p in probabilities] if prob_sum != 0 else [1 / len(to_nodes)] * len(to_nodes)
            }

    def random_walk(self, num_walks=5, walk_length=10):
        """执行带偏置的随机游走"""
        neighbors = set()

        # 获取所有种子节点
        seed_nodes = [doc['address'] for doc in self.db.valid_address.find()]

        for _ in tqdm(range(num_walks), desc="Performing walks"):
            for node in seed_nodes:
                if node not in self.graph:
                    continue

                current = node
                walk = [current]

                for _ in range(walk_length - 1):
                    if current not in self.graph:
                        break

                    # 选择下一个节点
                    next_node = np.random.choice(
                        self.graph[current]['nodes'],
                        p=self.graph[current]['probs']
                    )

                    walk.append(next_node)
                    current = next_node

                # 添加邻居节点（排除种子节点）
                neighbors.update(walk[1:])

        return neighbors

    def save_neighbors(self, neighbors):
        """保存邻居节点到数据库"""
        existing = {doc['address'] for doc in self.db.second_address_v2.find()}

        batch = []
        for addr in tqdm(neighbors, desc="Saving neighbors"):
            if addr not in existing:
                # 检查是否为已知有效地址
                doc = self.db.valid_address.find_one({'address': addr})
                if doc:
                    continue
                doc = self.db.invalid_address.find_one({'address': addr})
                if doc:
                    continue
                batch.append({
                    'address': addr,
                    'flag': 0
                })

            if len(batch) >= 1000:
                self.db.second_address_v2.insert_many(batch)
                batch = []

        if batch:
            self.db.second_address_v2.insert_many(batch)

    def run(self):
        """执行完整流程"""
        self.build_transaction_graph()
        neighbors = self.random_walk()
        self.save_neighbors(neighbors)
        self.client.close()


if __name__ == "__main__":
    processor = Trans2VecNeighborGenerator(alpha=0.5)
    processor.run()
