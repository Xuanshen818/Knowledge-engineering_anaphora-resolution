def remove_bracket_from_start_of_lines(file_path):
    # 读取文件
    with open(file_path, 'r',encoding='utf-8') as file:
        lines = file.readlines()

    # 如果行以[开头，就删除[
    lines = [line[1:] if line.startswith('[') else line for line in lines]

    # 将修改后的行写回文件
    with open(file_path, 'w',encoding='utf-8') as file:
        file.writelines(lines)

# 调用函数，传入你的文件路径
remove_bracket_from_start_of_lines('n.txt')

# 注意: 请将 'your_file.txt' 替换为你的文件路径