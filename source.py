import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
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

# 测试数据集读取
file_path = "C:\\Users\\86135\\Desktop\\知识工程\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\1998-01-2003版-带音.txt"
dataset = CoreferenceDataset(file_path)

# 打印前5个样本
for i in range(99999):
    words, labels = dataset[i]
    print("词语:", words)
    print("标签:", labels)
    print()

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

# 测试数据集处理
file_path = "C:\\Users\\86135\\Desktop\\知识工程\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\1998-01-2003版-带音.txt"
dataset = CoreferenceDataset(file_path)

pronouns = []
word_matrices = []

for words, labels in dataset:
    pronouns.extend(extract_pronouns(words, labels))

for pronoun in pronouns:
    word_matrices.append(word_to_vector(pronoun))

word_matrices = np.array(word_matrices)

print("代词数量:", len(pronouns))
print("代词矩阵形状:", word_matrices.shape)