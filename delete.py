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

# 初始化模型
input_dim = 300  # 输入特征维度
output_dim = 1   # 输出维度
model = GLM(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for words, labels in dataloader:
        pronouns = extract_pronouns(words[0], labels[0])
        if len(pronouns) == 0:
            continue

        # 将代词转换为词向量
        pronoun_vectors = []
        for pronoun in pronouns:
            pronoun_vectors.append(word_to_vector(pronoun))
        pronoun_vectors = torch.tensor(pronoun_vectors, dtype=torch.float32)

        # 将数据传递给模型
        outputs = model(pronoun_vectors)

        # 构造标签
        targets = torch.ones(len(pronouns)).unsqueeze(1)

        # 计算损失
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # 计算准确率
        predicted = (outputs > 0.5).type(torch.float32)
        correct += (predicted == targets).sum().item()
        total += len(pronouns)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = correct / total if total != 0 else 0
    print(f'Epoch {epoch+1}, Loss: {total_loss}, Accuracy: {accuracy}')
