import os
import json

def read_all_json_files(json_dir):
    all_json_data = []
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, 'r', encoding='gbk') as file:
                try:
                    json_data = json.load(file)
                    all_json_data.append(json_data)
                except UnicodeDecodeError:
                    print(f"Error reading file {file_path}. Trying with 'gbk' encoding.")
                    file.seek(0)  # Reset file pointer to the beginning
                    json_data = json.load(file, encoding='gbk')
                    all_json_data.append(json_data)
    return all_json_data

json_dir = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train'
json_data = read_all_json_files(json_dir)
print(json_data)

