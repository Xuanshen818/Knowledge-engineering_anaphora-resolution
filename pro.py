import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re

class CoreferenceDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.read_dataset(file_path)

    def read_dataset(self, file_path):
        data = []

        with open(file_path, 'r', encoding='gbk') as file:
            for line in file:
                line = line.strip()
                if line:
                    # 使用空格分割每个词语和其对应的标签
                    parts = line.split()
                    words = [part.split('/')[0] for part in parts]
                    labels = [part.split('/')[1] for part in parts]
                    data.append((words, labels))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, labels = self.data[idx]
        return words, labels

def extract_pronouns(words, labels):
    pronouns = []
    for i, label in enumerate(labels):
        if label == 'r':  # 代词的标签为'r'
            pronouns.append(words[i])
    return pronouns

def word_to_vector(word):
    # 这里假设您有一个将词语转换为词向量的函数word_embedding
    # word_embedding函数的输入是词语，输出是词向量
    # 这里我们简单地假设词向量的维度为300
    return np.random.rand(300)

def sentence_to_matrix(sentence, max_length=50):
    matrix = np.zeros((max_length, 300))  # 假设词向量的维度为300
    for i, word in enumerate(sentence[:max_length]):
        matrix[i] = word_to_vector(word)
    return matrix

class GLM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLM, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 准备数据集
file_path = "C:\\Users\\86135\\Desktop\\知识工程\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\1998-01-2003版-带音.txt"
dataset = CoreferenceDataset(file_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 检查数据集中是否有代词
has_pronouns = False
for words, labels in dataset:
    if extract_pronouns(words, labels):
        has_pronouns = True
        break

if not has_pronouns:
    print("数据集中没有代词，请检查数据集是否正确。")
else:
    print("数据集中有代词。")

# 初始化模型
input_dim = 300  # 输入特征维度
output_dim = 1   # 输出维度
model = GLM(input_dim, output_dim)

# 检查模型参数是否正确初始化
if model(torch.randn(1, input_dim)).item() == 0.5:
    print("模型参数正确初始化。")
else:
    print("模型参数未正确初始化，请检查模型结构和参数初始化方式。")
