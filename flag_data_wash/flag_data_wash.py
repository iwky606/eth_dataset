import pandas as pd
from api import etherscan_api
from dao import mongo_client
from dao.mongo_client import TRANSACTION, ERC20_TRANSFER

df = pd.read_csv('flag_data.csv')
# 提取两列数据
selected_columns = df[['Address', 'FLAG']]


def query_process_address(address):
    valid_result = mongo_client.query_valid_address(address)
    invalid_result = mongo_client.query_invalid_address(address)
    if valid_result or invalid_result:
        return True
    return False


cnt = 0
# 遍历每一行（仅在必要时使用，pandas通常用向量化操作）
for index, row in selected_columns.iterrows():
    address, flag = row['Address'], row['FLAG']
    if query_process_address(address):
        continue
    eth_tx, erc20_tx, status = etherscan_api.get_all_transactions(address)

    if not status:
        print(f'Address:{address} invalid')
        mongo_client.save_invalid_address(address, flag)
        continue
    else:
        print(f'success, Address:{address} valid')
        mongo_client.save_valid_address(address, flag)

    # 批量写入transaction
    mongo_client.save_batch_eth_dataset(eth_tx, TRANSACTION)

    # 批量写入erc20_transaction
    mongo_client.save_batch_eth_dataset(erc20_tx, ERC20_TRANSFER)

    # cnt += 1
    # if cnt > 10:
    #     break
