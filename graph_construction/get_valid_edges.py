# 获取交易中from和to节点都在train_nodes中的边。即构图需要的边
from db_connection import mongo_client


def process_transactions():
    # 配置数据库连接信息（根据实际情况修改）
    db = mongo_client.db

    # 步骤1：读取train_nodes中的所有address
    all_nodes = set()
    train_nodes = db.train_nodes.find({}, {"address": 1})

    for node in train_nodes:
        address = node.get('address')
        if address:
            all_nodes.add(address)

    print(f"已加载 {len(all_nodes)} 个节点地址")

    # 步骤2：分页处理transaction_uk
    page_size = 200
    page_number = 0
    total_processed = 0
    total_saved = 0

    while True:
        # 计算跳过的记录数
        skip = page_number * page_size

        # 获取当前页数据
        transactions = db.transaction_uk.find().skip(skip).limit(page_size)
        current_page = list(transactions)

        if not current_page:
            break  # 没有更多数据时退出循环

        # 处理当前页数据
        valid_transactions = []
        for tx in current_page:
            from_addr = tx.get('from')
            to_addr = tx.get('to')

            # 检查地址是否存在且都在all_nodes中
            if from_addr and to_addr:
                if from_addr in all_nodes and to_addr in all_nodes:
                    valid_transactions.append(tx)

        # 批量保存有效交易记录
        if valid_transactions:
            db.all_edges_py.insert_many(valid_transactions)
            total_saved += len(valid_transactions)

        total_processed += len(current_page)
        page_number += 1

        print(f"已处理 {total_processed} 条记录，已保存 {total_saved} 条有效边")

    print("处理完成")


if __name__ == "__main__":
    process_transactions()
