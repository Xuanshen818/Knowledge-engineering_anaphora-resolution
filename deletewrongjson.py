import os

# 读取 "wrong.txt" 文件中的每一行，并删除对应的文件
with open("C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\test\\wrong.txt", "r",
          encoding="utf-8") as f:
    for line in f:
        filename = line.strip()  # 获取文件名
        file_path = os.path.join(
            "C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\test", filename)

        # 检查文件是否存在，如果存在，则删除
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("File not found:", file_path)