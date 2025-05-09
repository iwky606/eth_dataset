import traceback

from pymongo import MongoClient
import atexit

# 常量定义
ETH_DATASET = 'eth_dataset'
VALID_ADDRESS = 'valid_address'
INVALID_ADDRESS = 'invalid_address'
TRANSACTION = 'transaction'
ERC20_TRANSFER = 'erc20_transfer'

# 二阶节点相关的文档
TEST = ''
SECOND_ADDRESS = 'second_address_v3'
INVALID_SECOND_ADDRESS = 'invalid_second_address' + TEST
VALID_SECOND_ADDRESS = 'valid_second_address' + TEST
SECOND_TRANSACTION = 'second_transaction' + TEST
SECOND_ERC20_TRANSFER = 'second_erc20_transfer' + TEST

NODE_FEATURES = 'node_features'

# 初始化连接
client = MongoClient("mongodb://root:123456@localhost:27017/")
db = client[ETH_DATASET]  # 获取指定数据库


def close_client():
    if client:
        client.close()
        print("MongoDB 客户端连接已关闭")


# 注册退出时关闭mongodb连接
atexit.register(close_client)


def save_to_eth_dataset(data, collection_name):
    try:
        collection = db[collection_name]
        result = collection.insert_one(data)
        print(f"数据保存成功，文档ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"保存到集合 {collection_name} 失败: {str(e)}")
        return None


def query_address(address, collection_name):
    try:
        collection = db[collection_name]
        return collection.find_one({'address': address})
    except Exception as e:
        print(f"查询集合 {collection_name} 失败: {str(e)}")
        return None


def query_second_address():
    try:
        collection = db[SECOND_ADDRESS]
        return [doc['address'] for doc in collection.find()]
    except Exception as e:
        print(f"查询集合 {SECOND_ADDRESS} 失败: {str(e)}")
        return []


def save_address(address, flag, collection_name):
    data = {'address': address, 'flag': flag}
    return save_to_eth_dataset(data, collection_name)


def save_batch_eth_dataset(
        data_list,
        collection_name,
):
    try:
        if not data_list or len(data_list) == 0:
            print("空列表，跳过写入操作")
            return []
        collection = db[collection_name]
        result = collection.insert_many(data_list)
        inserted_ids = [str(id) for id in result.inserted_ids]
        return inserted_ids
    except Exception as e:
        traceback.print_exc()
