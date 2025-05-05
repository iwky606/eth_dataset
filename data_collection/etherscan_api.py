import requests
import traceback
import time

# etherscan 接口密钥
ETHERSCAN_API_KEY = "IATSDMEKDDRDN6EKIW7JAGCX7BTHPP3CHP"

# etherscan 接口地址
BASE_URL = "https://api.etherscan.io/api"

# 最大重试次数
MAX_RETRIES = 5

# 分页获取数据工具
def get_page_data(params):
    transactions = []
    page = 1
    cnt = 0
    while True:
        params['page'] = page
        try:
            response = requests.get(BASE_URL, params=params, timeout=15)
            data = response.json()

            results = data["result"]
            if results is None:
                print(f"address: {params['address']}, 查询页数至: {page} 超过最大offset")
                return None, False

            transactions.extend(results)

            if len(results) < 100:
                break
            page += 1
            time.sleep(0.2)

            cnt = 0
        except Exception as e:
            traceback.print_exc()
            print(f"重试{params['address']}查询page:{page}任务")
            time.sleep(1)
            cnt += 1
            if cnt > MAX_RETRIES:
                print(f"重试{params['address']}查询page:{page}任务, 超过最大次数{MAX_RETRIES}, 手动查原因")
                return None, False
            continue

    return transactions, True


# 获取交易记录
def get_transactions(address: str):
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "offset": 200,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    return get_page_data(params)

# 获取代币转账
def get_erc20_transactions(address: str):
    params = {
        "module": "account",
        "action": "tokentx",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "offset": 100,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    return get_page_data(params)

# 获取以太坊上交易记录以及代币转账
def get_all_transactions(address: str):
    eth_tx, status = get_transactions(address)
    if not status:
        return None, None, False
    erc20_tx, status = get_erc20_transactions(address)
    if not status:
        return None, None, False
    if len(erc20_tx) == 0 and len(eth_tx) == 0:
        return None, None, False
    return eth_tx, erc20_tx, True


if __name__ == "__main__":
    # 调试代码，不在工程上直接运行
    import json
    result, result2, status = get_all_transactions('0xb53e95a4b7c5e15a790df3a66709b2e9f1cf9e3f')
    print(result)
    print(json.dumps(result2, indent=4))
