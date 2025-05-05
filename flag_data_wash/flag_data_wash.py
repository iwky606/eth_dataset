import pandas as pd
from data_collection import etherscan_api
from dao import mongo_client
from dao.mongo_client import TRANSACTION, ERC20_TRANSFER, INVALID_ADDRESS, VALID_ADDRESS, INVALID_SECOND_ADDRESS, \
    VALID_SECOND_ADDRESS, SECOND_TRANSACTION, SECOND_ERC20_TRANSFER


def query_process_address(address):
    valid_result = mongo_client.query_address(address, VALID_ADDRESS)
    invalid_result = mongo_client.query_address(address, INVALID_ADDRESS)
    invalid_second_result = mongo_client.query_address(address, INVALID_SECOND_ADDRESS)
    valid_second_result = mongo_client.query_address(address, VALID_SECOND_ADDRESS)
    if valid_result or invalid_result or valid_second_result or invalid_second_result:
        return True
    return False


def wash_data():
    df = pd.read_csv('address_2.csv')
    # 提取两列数据
    selected_columns = df[['Address', 'FLAG']]
    cnt = 0
    # 遍历每一行（仅在必要时使用，pandas通常用向量化操作）
    for index, row in selected_columns.iterrows():
        address, flag = row['Address'], row['FLAG']
        if query_process_address(address):
            print(f'address: {address}\n has been processed')
            continue
        print(f'address: {address}\n is processing')
        eth_tx, erc20_tx, status = etherscan_api.get_all_transactions(address)

        if not status:
            print(f'Address:{address} invalid')
            mongo_client.save_address(address, flag, INVALID_ADDRESS)
            continue
        else:
            print(f'success, Address:{address} valid')
            mongo_client.save_address(address, flag, VALID_ADDRESS)

        # 批量写入transaction
        mongo_client.save_batch_eth_dataset(eth_tx, TRANSACTION)

        # 批量写入erc20_transaction
        mongo_client.save_batch_eth_dataset(erc20_tx, ERC20_TRANSFER)

        print(f'address: {address}\n done')

        # cnt += 1
        # if cnt > 10:
        #     break


def wash_second_data(max_cnt=-1):
    second_address = mongo_client.query_second_address()
    cnt = 0
    for address in second_address:
        print('=======[START]=======')

        if query_process_address(address):
            print(f'address: {address}\n has been processed')
            continue

        print(f'address: {address}\n is processing')
        eth_tx, erc20_tx, status = etherscan_api.get_all_transactions(address)

        if not status:
            print(f'Address:{address} invalid')
            mongo_client.save_address(address, 0, INVALID_SECOND_ADDRESS)
            continue
        else:
            print(f'success, Address:{address} valid')
            mongo_client.save_address(address, 0, VALID_SECOND_ADDRESS)

        # 批量写入transaction
        mongo_client.save_batch_eth_dataset(eth_tx, SECOND_TRANSACTION)

        # 批量写入erc20_transaction
        mongo_client.save_batch_eth_dataset(erc20_tx, SECOND_ERC20_TRANSFER)

        print('=======[DONE]=======')

        if max_cnt != -1 and cnt > max_cnt:
            break
        cnt += 1


if __name__ == '__main__':
    wash_second_data()
