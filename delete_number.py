                                 # 删除数字
def remove_space_and_numbers_in_file(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 使用正则表达式匹配中文后的空格及数字，并删除数字
    import re
    pattern = r'([\u4e00-\u9fff]+)(\s+)(\d+)'
    for i, line in enumerate(lines):
        lines[i] = re.sub(pattern, r'\1', line)

    # 将处理后的文本写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# 测试用例，假设文件名为 test.txt
remove_space_and_numbers_in_file('n_sort_delete_number.txt')
