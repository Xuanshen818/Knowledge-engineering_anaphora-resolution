import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class CoreferenceDataset:
    def __init__(self, json_dir, corpus_file, noun_vocab_file):
        self.encoder = self.initialize_encoder(noun_vocab_file)
        self.data = self.read_dataset(json_dir, corpus_file, noun_vocab_file)

    def initialize_encoder(self, noun_vocab_file):
        # 读取词库文件
        with open(noun_vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().splitlines()

        vocab = np.array(vocab).reshape(-1, 1)

        encoder = OneHotEncoder()
        encoder.fit(vocab)

        return encoder

    def read_dataset(self, json_dir, corpus_file, noun_vocab_file):
        # 读取语料库
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.readlines()

        data = []

        # 读取 JSON 文件
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(json_dir, filename)
                with open(file_path, 'r', encoding='gbk') as file:
                    try:
                        json_data = json.load(file)
                    except UnicodeDecodeError:
                        print("UnicodeDecodeError in file:", file_path)
                        continue
                    if not isinstance(json_data, dict):  # 检查是否为字典
                        continue
                    task_id = json_data['taskID'].split()[0]  # 获取任务 ID
                    pronoun_index = json_data['pronoun']['indexFront']
                    antecedent_index = json_data['0']['indexFront']
                    antecedent_num = json_data['antecedentNum']

                    # 提取代词和先行词所在句子
                    pronoun_sentence_id = None
                    antecedent_sentence_id = None
                    for i, line in enumerate(corpus):
                        words = line.strip().split()
                        if pronoun_sentence_id is None and pronoun_index < len(words):
                            pronoun_sentence_id = i
                        if antecedent_sentence_id is None and antecedent_index < len(words):
                            antecedent_sentence_id = i
                        if pronoun_sentence_id is not None and antecedent_sentence_id is not None:
                            break

                    if pronoun_sentence_id is None or antecedent_sentence_id is None:
                        continue  # 跳过未找到句子的情况

                    pronoun_sentence = corpus[pronoun_sentence_id].strip().split()
                    antecedent_sentence = corpus[antecedent_sentence_id].strip().split()

                    # 提取代词和先行词
                    pronoun = pronoun_sentence[pronoun_index]
                    antecedent = antecedent_sentence[antecedent_index]

                    # one-hot 编码代词和先行词
                    pronoun_vector = self.word_to_onehot(pronoun)
                    antecedent_vector = self.word_to_onehot(antecedent)

                    # 添加数据
                    pronoun_number = 1 if antecedent_num == 1 else 0
                    pronoun_sentence = [self.encoder.categories_[0].tolist().index(word) for word in pronoun_sentence if word in self.encoder.categories_[0]]
                    previous_word_vector = np.zeros(len(self.encoder.categories_[0]), dtype=int)
                    if pronoun_sentence:
                        previous_word_vector[pronoun_sentence[-1]] = 1
                    data.append({
                        "pronoun_vector": pronoun_vector,
                        "antecedent_vector": antecedent_vector,
                        "previous_word_vector": previous_word_vector,
                        "pronoun_number": pronoun_number,
                        "pronoun_sentence": pronoun_sentence
                    })

        return data

    def word_to_onehot(self, word):
        if word in self.encoder.categories_[0]:
            one_hot_encoded = self.encoder.transform([[word]]).toarray()[0]
        else:
            # 如果词汇不在词库中，则使用对应维度的全零向量
            one_hot_encoded = np.zeros(len(self.encoder.categories_[0]), dtype=np.float32)
        return one_hot_encoded

    def __getitem__(self, index):
        item = self.data[index]
        pronoun_vector = torch.tensor(item["pronoun_vector"], dtype=torch.float32)
        antecedent_vector = torch.tensor(item["antecedent_vector"], dtype=torch.float32)
        previous_word_vector = torch.tensor(item["previous_word_vector"], dtype=torch.float32)
        pronoun_number = torch.tensor(item["pronoun_number"], dtype=torch.float32)
        return pronoun_vector, antecedent_vector, previous_word_vector, pronoun_number

    def __len__(self):
        return len(self.data)


# 读取数据集文件
corpus_file = 'text.txt'

# 读取词库文件
noun_vocab_file = 'n_sort_delete_number.txt'

# 生成特征向量和标签向量，并计算最长句子长度
max_sentence_length = 0

with open(corpus_file, 'r', encoding='utf-8') as f:
    corpus = f.readlines()

    for line in corpus:
        sentence_length = len(line.strip().split())
        max_sentence_length = max(max_sentence_length, sentence_length)

# 生成特征向量和标签向量，并进行填充
features = []
labels = []

json_folder = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train'

dataset = CoreferenceDataset(json_folder, corpus_file, noun_vocab_file)

for i in range(len(dataset)):
    pronoun_vector, antecedent_vector, previous_word_vector, pronoun_number = dataset[i]
    features.append(torch.cat((pronoun_vector, antecedent_vector, previous_word_vector)))
    labels.append(pronoun_number)

# 找到最长的标签数组长度
max_label_length = max(len(label) for label in labels)

# 填充所有标签数组
padded_labels = []
for label in labels:
    padded_label = label.unsqueeze(0)
    padded_label = torch.zeros(max_label_length).copy_(padded_label)
    padded_labels.append(padded_label)

labels_tensor = torch.stack(padded_labels)
features_tensor = torch.stack(features)

# 将数据划分为训练集和测试集
split = int(0.8 * len(features))
train_features, test_features = features_tensor[:split], features_tensor[split:]
train_labels, test_labels = labels_tensor[:split], labels_tensor[split:]

# 将数据转换为 PyTorch 张量
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义神经网络模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(-1, 1)  # 调整输出形状


input_size = len(features_tensor[0])
hidden_size = 64
model = SimpleModel(input_size, hidden_size)

# 损失函数和优化器
class_weight = torch.tensor([1, (len(labels_tensor) - sum(sum(labels_tensor))) / sum(sum(labels_tensor))], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型移动到 GPU 上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# 训练模型
num_epochs = 10
train_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # 输出训练损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 绘制训练损失曲线
plt.plot(train_losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
