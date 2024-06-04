import os
import json


def check_empty_json_files(folder_path):
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)

    # 过滤出json文件
    json_files = [file for file in files if file.endswith('.json')]

    empty_json_files = []

    # 检查每个json文件是否为空
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)

        with open(file_path, 'r', encoding='gbk') as file:
            try:
                data = json.load(file)
                if not data:  # 如果json数据为空
                    empty_json_files.append(json_file)
            except json.JSONDecodeError:
                # 如果json文件不能被解析，视为不为空的文件
                continue

    if empty_json_files:
        print("空的JSON文件: ")
        for empty_file in empty_json_files:
            print(empty_file)
    else:
        print("没有空的JSON文件")


# 使用示例
folder_path = "C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\test"
check_empty_json_files(folder_path)
