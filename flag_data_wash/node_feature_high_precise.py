from collections import defaultdict
from decimal import Decimal
import numpy as np
from dao.mongo_client import db, NODE_FEATURES
from dao import mongo_client


def get_address_features(address):
    # 连接MongoDB

    # 获取所有相关交易
    tx_query = {"$or": [{"from": address}, {"to": address}]}
    normal_txs = list(db.transaction_uk.find(tx_query))
    erc20_txs = list(db.erc20_transfer_uk.find(tx_query))

    # 初始化特征字典
    features = {"Address": address}

    # 处理普通交易特征
    process_normal_transactions(features, normal_txs, address)

    # 处理ERC20交易特征
    process_erc20_transactions(features, erc20_txs, address)

    # 计算最终衍生特征
    calculate_derived_features(features)

    return convert_decimals_to_strings(features)


def process_normal_transactions(features, txs, address):
    # 初始化数据结构
    sent, received = [], []
    contract_creations = 0
    values_sent = []
    values_received = []
    gas_fees = []
    unique_sent_to = set()
    unique_received_from = set()
    # 在原有代码基础上添加
    contract_values = []

    for tx in txs:
        # 时间处理
        timestamp = int(tx['timeStamp'])
        value = Decimal(tx['value']) / Decimal('1e18')  # 转换为ETH
        # 添加合约交易判断（假设合约交易有input数据）
        if tx['input'] not in ['0x', '']:
            contract_values.append(value)

        # 发送交易
        if tx['from'] == address:
            sent.append(timestamp)
            values_sent.append(value)
            unique_sent_to.add(tx['to'])

            # 合约创建判断
            if tx.get('contractAddress'):
                contract_creations += 1

        # 接收交易
        if tx['to'] == address:
            received.append(timestamp)
            values_received.append(value)
            unique_received_from.add(tx['from'])

        # Gas费用计算
        gas_used = Decimal(tx.get('gasUsed', 0))
        gas_price = Decimal(tx.get('gasPrice', 0))
        gas_fees.append(gas_used * gas_price / Decimal('1e18'))  # 转换为ETH

    # 存储基础特征
    def safe_decimal_op(values, op):
        if values:
            if op == 'min':
                return min(values)
            elif op == 'max':
                return max(values)
            elif op == 'mean':
                return sum(values) / len(values)
        return Decimal('0')

    features.update({
        "Sent_tnx": len(sent),
        "Received_tnx": len(received),
        "Number_of_Created_Contracts": contract_creations,
        "Unique_Sent_To_Addresses20": len(unique_sent_to),
        "Unique_Received_From_Addresses": len(unique_received_from),
        "Min_Val_Sent": safe_decimal_op(values_sent, 'min'),
        "Max_Val_Sent": safe_decimal_op(values_sent, 'max'),
        "Avg_Val_Sent": safe_decimal_op(values_sent, 'mean'),
        "Min_Value_Received": safe_decimal_op(values_received, 'min'),
        "Max_Value_Received": safe_decimal_op(values_received, 'max'),
        "Avg_Value_Received5": safe_decimal_op(values_received, 'mean'),
        "Avg_Gas_Fee (ETH)": safe_decimal_op(gas_fees, 'mean'),
        "Max_Gas_Fee (ETH)": safe_decimal_op(gas_fees, 'max'),
        "Min_Gas_Fee (ETH)": safe_decimal_op(gas_fees, 'min'),
        "Total_Transactions(Including_Tnx_to_Create_Contract)": len(txs),
        "Total_Ether_Sent": sum(values_sent),
        "Total_Ether_Received": sum(values_received),
        "Total_Ether_Sent_Contracts": sum(contract_values),
        "Min_Value_Sent_To_Contract": safe_decimal_op(contract_values, 'min'),
        "Max_Value_Sent_To_Contract": safe_decimal_op(contract_values, 'max'),
        "Avg_Value_Sent_To_Contract": safe_decimal_op(contract_values, 'mean'),
        "_sent_timestamps": sorted(sent),
        "_received_timestamps": sorted(received)
    })


def process_erc20_transactions(features, txs, address):
    # 初始化数据结构
    sent, received = [], []
    values_sent = []
    values_received = []
    gas_fees = []
    token_sent = defaultdict(Decimal)
    token_received = defaultdict(Decimal)
    unique_sent_to = set()
    unique_received_from = set()
    # 添加合约地址跟踪
    contract_sent = set()
    contract_values_sent = []

    for tx in txs:
        # ERC20数值转换
        decimals = int(tx['tokenDecimal'])
        value = Decimal(tx['value']) / (Decimal(10) ** decimals)
        timestamp = int(tx['timeStamp'])

        # Gas费用计算
        gas_used = Decimal(tx.get('gasUsed', 0))
        gas_price = Decimal(tx.get('gasPrice', 0))
        gas_fee = gas_used * gas_price / Decimal('1e18')

        # 发送交易
        if tx['from'] == address:
            sent.append(timestamp)
            values_sent.append(value)
            unique_sent_to.add(tx['to'])
            token_sent[tx['tokenSymbol']] += value
            if tx['to'].startswith('0x'):
                contract_sent.add(tx['to'])
                contract_values_sent.append(value)

        # 接收交易
        if tx['to'] == address:
            received.append(timestamp)
            values_received.append(value)
            unique_received_from.add(tx['from'])
            token_received[tx['tokenSymbol']] += value

        gas_fees.append(gas_fee)

    # 存储基础特征
    def safe_decimal_op(values, op):
        if values:
            if op == 'min':
                return min(values)
            elif op == 'max':
                return max(values)
            elif op == 'mean':
                return sum(values) / len(values)
        return Decimal('0')

    features.update({
        "Total_ERC20_Tnxs": len(txs),
        "ERC20_Total_Ether_Received": sum(values_received),
        "ERC20_Total_Ether_Sent": sum(values_sent),
        "ERC20_Uniq_Sent_Addr": len(unique_sent_to),
        "ERC20_Uniq_Rec_Addr": len(unique_received_from),
        "ERC20_Min_Val_Rec": safe_decimal_op(values_received, 'min'),
        "ERC20_Max_Val_Rec": safe_decimal_op(values_received, 'max'),
        "ERC20_Avg_Val_Rec": safe_decimal_op(values_received, 'mean'),
        "ERC20_Min_Val_Sent": safe_decimal_op(values_sent, 'min'),
        "ERC20_Max_Val_Sent": safe_decimal_op(values_sent, 'max'),
        "ERC20_Avg_Val_Sent": safe_decimal_op(values_sent, 'mean'),
        "ERC20_Uniq_Sent_Token_Name": len(token_sent),
        "ERC20_Uniq_Rec_Token_Name": len(token_received),
        "ERC20_Most_Sent_Token_Type": max(token_sent, key=token_sent.get, default=''),
        "ERC20_Most_Rec_Token_Type": max(token_received, key=token_received.get, default=''),
        "ERC20_Total_Ether_Sent_Contract": sum(contract_values_sent),
        "ERC20_Uniq_Rec_Contract_Addr": len(contract_sent),
        "ERC20_Avg_Time_Between_Contract_Tnx": safe_decimal_op([t for t, v in zip(sent, contract_values_sent) if v > 0],
                                                               'mean') // 60 if len(
            contract_values_sent) > 1 else 0,

        "_erc20_sent_ts": sorted(sent),
        "_erc20_rec_ts": sorted(received),
        "_erc20_gas_fees": [float(fee) for fee in gas_fees]
    })


def calculate_derived_features(features):
    # 时间差计算函数
    def time_diff(timestamps):
        if len(timestamps) < 2: return 0
        return (timestamps[-1] - timestamps[0]) // 60

    # 普通交易时间特征
    features["Time_Diff_between_first_and_last(Mins)"] = time_diff(
        features.get('_sent_timestamps', []) + features.get('_received_timestamps', [])
    )

    # 普通交易时间间隔
    for prefix in ['sent', 'received']:
        timestamps = features.get(f'_{prefix}_timestamps', [])
        diffs = np.diff(timestamps) // 60  # 转换为分钟
        avg = np.mean(diffs) if len(diffs) > 0 else 0
        features[f"Avg_min_between_{prefix}_tnx"] = avg

    # ERC20时间特征
    for tx_type in ['sent', 'rec']:
        timestamps = features.get(f'_erc20_{tx_type}_ts', [])
        diffs = np.diff(timestamps) // 60
        avg = np.mean(diffs) if len(diffs) > 0 else 0
        features[f"ERC20_Avg_Time_Between_{tx_type.title()}_Tnx"] = avg

    # Gas费用特征
    features["ERC20 Avg gas fee (ETH)"] = np.mean(features["_erc20_gas_fees"]) if features["_erc20_gas_fees"] else 0
    features["ERC20 Max gas fee (ETH)"] = np.max(features["_erc20_gas_fees"]) if features["_erc20_gas_fees"] else 0
    features["ERC20 Min gas fee (ETH)"] = np.min(features["_erc20_gas_fees"]) if features["_erc20_gas_fees"] else 0
    # features["Total_Ether_Balance"] = features["Total_Ether_Received"] - features["Total_Ether_Sent"]

    # 清理临时字段
    for key in ['_sent_timestamps', '_received_timestamps',
                '_erc20_sent_ts', '_erc20_rec_ts', '_erc20_gas_fees']:
        features.pop(key, None)


def get_csv_features(address):
    import pandas as pd

    # 读取CSV文件
    df = pd.read_csv('flag_data.csv')

    # 过滤符合条件的行
    filtered_df = df[df['Address'] == address]

    # 转换为JSON（orient参数控制格式）
    return convert_decimals_to_strings(filtered_df.to_dict('records')[0])


def convert_decimals_to_strings(data):
    if isinstance(data, dict):
        return {key: convert_decimals_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_decimals_to_strings(item) for item in data]
    elif isinstance(data, Decimal):
        return str(data)
    return data


def save_features(features):
    mongo_client.save_to_eth_dataset(features, NODE_FEATURES)


def solve_address_features(address):
    features = get_csv_features(address)
    save_features(features)


# 使用示例
if __name__ == "__main__":
    address = '0xb53e95a4b7c5e15a790df3a66709b2e9f1cf9e3f'
    features = get_address_features(address)
    import json

    print(json.dumps(features))
    print(json.dumps(get_csv_features(address)))
