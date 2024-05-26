# 删除重复行
def remove_duplicates(file_path):
    try:
        # 读取文件内容并去重
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            unique_lines = set(lines)

        # 写入去重后的内容到同一文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(unique_lines)

        print("去重完成！")

    except Exception as e:
        print("处理出错:", e)


# 文件路径
file_path = "vn_sort_nr.txt"  # 替换为你的文件路径

# 处理文件
remove_duplicates(file_path)

