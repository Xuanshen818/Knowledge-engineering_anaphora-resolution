import os
import json

# 输入文件夹路径，这里指向包含JSON文件的文件夹
folder_path = r'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train'
# 文本文件路径，这里指向包含语料的文本文件
text_file = r'C:\\Users\\86135\\Desktop\\知识工程\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\1998-01-2003data.txt'
# 输出文件路径，这里指向要写入语料的文本文件
output_file = r'text.txt'

# 用于保存每个 JSON 文件的数据
json_data = {}

# 用于保存已经加载的文件名
loaded_files = set()

# 用于保存加载失败的文件名
error_files = []

# 遍历文件夹中的所有JSON文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        # 检查文件名是否已经加载过，如果加载过，则跳过
        if filename in loaded_files:
            continue
        # 尝试打开JSON文件并加载数据到字典中
        try:
            with open(os.path.join(folder_path, filename), 'r', encoding='gbk') as json_file:
                data = json.load(json_file)
                # 获取JSON文件中的id并存储到字典中
                id = data['0']['id']
                json_data[id] = data
            # 将文件名添加到已加载的文件集合中
            loaded_files.add(filename)
            print(f"处理文件: {filename}, ID: {id}")
        except Exception as e:
            # 如果加载失败，则记录文件名，并输出错误信息
            error_files.append(filename)
            print(f"加载文件 {filename} 失败: {e}")

# 用于保存当前正在处理的语句的变量
current_sentence = ''

# 打开文本文件，逐行搜索
with open(text_file, 'r', encoding='gbk') as txt_file:
    for line in txt_file:
        # 检查当前行是否以id开头
        for id, data in json_data.items():
            if line.startswith(id):
                # 如果不是第一行，则先将之前的语句写入输出文件
                if current_sentence:
                    with open(output_file, 'a', encoding='utf-8') as out_file:
                        out_file.write(current_sentence.strip() + '\n')
                    # 将当前语句清空，准备处理下一个语句
                    current_sentence = ''
                # 将当前行添加到当前语句中
                current_sentence += line
                break
        # 如果当前行不是以任何id开头，说明当前语句还未结束
        else:
            # 将当前行添加到当前语句中
            current_sentence += line

# 处理完所有文本文件后，将最后一个语句写入输出文件
if current_sentence:
    with open(output_file, 'a', encoding='utf-8') as out_file:
        out_file.write(current_sentence.strip() + '\n')

# 输出加载失败的文件名
if error_files:
    print("加载失败的文件:")
    for filename in error_files:
        print(filename)