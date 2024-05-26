import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

# 数据集类
class CoreferenceDataset(Dataset):
    def __init__(self, json_dir, corpus_file, noun_vocab_file):
        self.data = self.read_dataset(json_dir)
        self.corpus = self.read_corpus(corpus_file)
        self.noun_vocab = self.read_noun_vocab(noun_vocab_file)

    def read_dataset(self, json_dir):
        all_json_data = []
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(json_dir, filename)
                with open(file_path, 'r', encoding='gbk') as file:
                    json_data = json.load(file)
                    all_json_data.extend(json_data)
        return all_json_data

    def read_corpus(self, corpus_file):
        with open(corpus_file, 'r', encoding='utf-8') as file:
            corpus = file.read().split()
        return corpus

    def read_noun_vocab(self, noun_vocab_file):
        with open(noun_vocab_file, 'r', encoding='utf-8') as file:
            noun_vocab = [line.strip() for line in file]
        return noun_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sentence = sample['sentence']
        pronoun_idx = sample['pronoun']['indexFront']
        pronoun = sentence[pronoun_idx]

        # 提取特征
        features = self.extract_features(sentence, pronoun_idx)

        # 获取代词的先行词
        antecedent = sample['antecedentNum']

        return features, antecedent

    def extract_features(self, sentence, pronoun_idx):
        # 初始化特征向量
        max_length = 50
        feature = np.zeros((max_length, 300))

        # 提取代词之前的词语作为特征
        for i in range(pronoun_idx):
            word = sentence[i]
            if word in self.noun_vocab:
                feature[i] = self.word_to_vector(word)

        return feature

    def word_to_vector(self, word):
        # 假设这是一个将词语转换为词向量的函数
        return np.random.rand(300)

# 定义模型
class CoreferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CoreferenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# 准备数据集
json_dir = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train'
corpus_file = 'C:\\Users\\86135\\Desktop\\知识工程\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\1998-01-2003data.txt'
noun_vocab_file = 'D:\\Python 3.11\\a PyCharm\\pythonProject3\\n_sort_delete_number.txt'
dataset = CoreferenceDataset(json_dir, corpus_file, noun_vocab_file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 初始化模型
input_dim = 300
hidden_dim = 128
output_dim = 1
model = CoreferenceModel(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for features, antecedent in dataloader:
        features = torch.tensor(features, dtype=torch.float32)
        antecedent = torch.tensor(antecedent, dtype=torch.float32).unsqueeze(1)

        # 将数据传递给模型
        outputs = model(features)

        # 计算损失
        loss = criterion(outputs, antecedent)
        total_loss += loss.item()

        # 计算准确率
        predicted = (torch.sigmoid(outputs) > 0.5).type(torch.float32)
        correct += (predicted == antecedent).sum().item()
        total += antecedent.size(0)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = correct / total if total != 0 else 0
    print(f'Epoch {epoch+1}, Loss: {total_loss}, Accuracy: {accuracy}')
