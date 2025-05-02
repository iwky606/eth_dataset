import pymongo
import numpy as np
from tqdm import tqdm
from dao.mongo_client import client, ETH_DATASET


class Trans2VecNeighborGenerator:
    def __init__(self, collection_name, db_name=ETH_DATASET, alpha=0.5):
        self.client = client
        self.db = self.client[db_name]
        self.transactions = self.db[collection_name]
        self.alpha = alpha
        self.graph = {}

    def build_transaction_graph(self):
        """构建交易图结构并计算转移概率，支持双向关系"""
        valid_addrs = {doc['address'] for doc in self.db.valid_address.find({'flag': {'$in': [1, 2]}})}
        adj_dict = {}

        for tx in tqdm(self.transactions.find(), desc="Processing transactions"):
            from_addr = tx.get('from')
            to_addr = tx.get('to')

            if not from_addr or not to_addr:
                continue

            value = int(tx.get('value', 0))
            timestamp = int(tx.get('timeStamp', 0))

            # 处理正向关系（from是valid）
            if from_addr in valid_addrs:
                if from_addr not in adj_dict:
                    adj_dict[from_addr] = {}
                if to_addr not in adj_dict[from_addr]:
                    adj_dict[from_addr][to_addr] = {'amount': 0, 'timestamp': 0}
                adj_dict[from_addr][to_addr]['amount'] += value
                adj_dict[from_addr][to_addr]['timestamp'] = max(
                    adj_dict[from_addr][to_addr]['timestamp'], timestamp
                )

            # 处理逆向关系（to是valid）
            if to_addr in valid_addrs:
                if to_addr not in adj_dict:
                    adj_dict[to_addr] = {}
                if from_addr not in adj_dict[to_addr]:
                    adj_dict[to_addr][from_addr] = {'amount': 0, 'timestamp': 0}
                adj_dict[to_addr][from_addr]['amount'] += value
                adj_dict[to_addr][from_addr]['timestamp'] = max(
                    adj_dict[to_addr][from_addr]['timestamp'], timestamp
                )

        # 计算转移概率
        for node in tqdm(adj_dict, desc="Calculating probabilities"):
            edges = adj_dict[node]
            neighbors = list(edges.keys())

            amounts = [edges[n]['amount'] for n in neighbors]
            timestamps = [edges[n]['timestamp'] for n in neighbors]

            # 时间步长排序逻辑
            sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
            time_steps = {i: idx + 1 for idx, i in enumerate(sorted_indices)}

            # 计算混合概率
            total_amount = sum(amounts)
            pa = [a / total_amount if total_amount != 0 else 1 / len(amounts) for a in amounts]
            pt = [time_steps[i] / sum(time_steps.values()) for i in range(len(neighbors))]

            probabilities = [
                (pa[i] ** self.alpha) * (pt[i] ** (1 - self.alpha))
                for i in range(len(neighbors))
            ]

            # 归一化处理
            prob_sum = sum(probabilities)
            self.graph[node] = {
                'nodes': neighbors,
                'probs': [p / prob_sum for p in probabilities] if prob_sum != 0 else [1 / len(neighbors)] * len(
                    neighbors)
            }

    def random_walk(self, num_walks=5, walk_length=10):
        """增强的随机游走，支持双向遍历"""
        neighbors = set()
        seed_nodes = [doc['address'] for doc in self.db.valid_address.find()]

        for _ in tqdm(range(num_walks), desc="Performing walks"):
            for seed in seed_nodes:
                if seed not in self.graph:
                    continue

                current = seed
                walk = [current]

                for _ in range(walk_length - 1):
                    if current not in self.graph:
                        break

                    # 带概率选择下一个节点
                    next_node = np.random.choice(
                        self.graph[current]['nodes'],
                        p=self.graph[current]['probs']
                    )

                    walk.append(next_node)
                    current = next_node

                # 收集非种子节点作为邻居
                neighbors.update(walk[1:])

        return neighbors

    def save_neighbors(self, neighbors):
        """批量保存邻居节点，包含去重逻辑"""
        existing_valid = {doc['address'] for doc in self.db.valid_address.find()}
        existing_invalid = {doc['address'] for doc in self.db.invalid_address.find()}
        existing_second = {doc['address'] for doc in self.db.second_address_v3.find()}

        batch = []
        for addr in tqdm(neighbors, desc="Saving neighbors"):
            if addr not in existing_valid and addr not in existing_invalid and addr not in existing_second:
                batch.append({'address': addr, 'flag': 0})

            if len(batch) >= 1000:
                self.db.second_address_v3.insert_many(batch)
                batch = []

        if batch:
            self.db.second_address_v3.insert_many(batch)

    def run(self):
        """执行完整流程"""
        self.build_transaction_graph()
        neighbors = self.random_walk()
        self.save_neighbors(neighbors)
        self.client.close()


if __name__ == "__main__":
    # processor = Trans2VecNeighborGenerator(collection_name='flag_transaction', alpha=0.5)
    processor = Trans2VecNeighborGenerator(collection_name='flag_erc20_transfer', alpha=0.5)

    processor.run()
