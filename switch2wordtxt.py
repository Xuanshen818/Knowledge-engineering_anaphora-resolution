import re

def extract_words_with_tags(input_file, output_file):
    with open(input_file, 'r', encoding='gbk') as f:  # 使用 gbk 编码打开 ANSI 文件
        lines = f.readlines()

    tagged_words = []
    for line in lines:
        matches = re.findall(r'(\S+/n\w*)', line)
        for match in matches:
            # 找到最后一个斜杠的位置
            last_slash_index = match.rfind('/')
            # 获取斜杠前面的部分作为词汇
            word = match[:last_slash_index]
            tagged_words.append(word)

    with open(output_file, 'w', encoding='utf-8') as f:  # 写入文件时最好使用 UTF-8 编码
        for word in tagged_words:
            f.write(word + '\n')

# 调用函数提取并保存词汇
input_file = 'C:/Users/86135/Desktop/知识工程/20180712165812468713/04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330/1998-01-2003data.txt'  # 输入文件名
output_file = 'n.txt'    # 输出文件名
extract_words_with_tags(input_file, output_file)
