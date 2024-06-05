import os
import json
import nltk
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 读取 JSON 文件内容
def read_json_files(json_folder):
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
    json_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='gbk') as f:
            json_content = json.load(f)
            json_data.append(json_content)
    return json_data

train_json_folder = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train'
test_json_folder = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\test'
validation_json_folder = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\validation'

train_json_data = read_json_files(train_json_folder)
test_json_data = read_json_files(test_json_folder)
validation_json_data = read_json_files(validation_json_folder)

# 读取数据集文件
with open('text.txt', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

# 计算最大句子长度和词性标签集
pos_tags = set()
max_sentence_length = 0
for data in train_json_data + test_json_data + validation_json_data:
    json_id = data["0"]["id"]
    pronoun_index_front = data['pronoun']['indexFront']
    pronoun_index_behind = data['pronoun']['indexBehind']

    for line in corpus:
        if json_id in line:
            sentence = line.strip().split(" ", 1)[1]
            break

    words_before_pronoun = sentence[:pronoun_index_front].split()
    words_after_pronoun = sentence[pronoun_index_behind:].split()
    sentence_length = len(words_before_pronoun) + len(words_after_pronoun)
    max_sentence_length = max(max_sentence_length, sentence_length)

    words_before_pronoun_pos = nltk.pos_tag(words_before_pronoun)
    words_after_pronoun_pos = nltk.pos_tag(words_after_pronoun)

    for _, pos in words_before_pronoun_pos:
        pos_tags.add(pos)
    for _, pos in words_after_pronoun_pos:
        pos_tags.add(pos)

pos_tags = {tag: idx for idx, tag in enumerate(pos_tags)}

# 生成特征向量和标签向量的函数
def process_data(json_data, corpus, pos_tags, max_sentence_length):
    features = []
    labels = []
    for data in json_data:
        json_id = data["0"]["id"]
        pronoun_index_front = data['pronoun']['indexFront']
        pronoun_index_behind = data['pronoun']['indexBehind']
        antecedent_index_front = data['0']['indexFront']
        antecedent_index_behind = data['0']['indexBehind']

        # 获取句子并提取词汇
        for line in corpus:
            if json_id in line:
                sentence = line.strip().split(" ", 1)[1]
                break

        words_before_pronoun = sentence[:pronoun_index_front].split()
        words_after_pronoun = sentence[pronoun_index_behind:].split()

        # 构造句子的特征向量
        sentence_features = []

        # 构造词性特征向量
        for word, pos in nltk.pos_tag(words_before_pronoun):
            pos_idx = pos_tags[pos]
            sentence_features.append(pos_idx)

        for word, pos in nltk.pos_tag(words_after_pronoun):
            pos_idx = pos_tags[pos]
            sentence_features.append(pos_idx)

        # 补全特征向量使其与最长的句子等长
        padding_length = max_sentence_length - len(sentence_features)
        sentence_features += [0] * padding_length

        features.append(sentence_features)

        # 使用每个句子的最后一个标签作为整个句子的标签
        sentence_label = 1 if (antecedent_index_front < len(words_before_pronoun) and
                               antecedent_index_behind <= len(words_before_pronoun)) or (
                                      pronoun_index_front < len(words_before_pronoun) and
                                      pronoun_index_behind <= len(words_before_pronoun)) else 0

        # 将每个句子的标签添加到标签列表中
        labels.append(sentence_label)

    return np.array(features), np.array(labels)

# 处理训练集、测试集和验证集数据
train_features, train_labels = process_data(train_json_data, corpus, pos_tags, max_sentence_length)
test_features, test_labels = process_data(test_json_data, corpus, pos_tags, max_sentence_length)
validation_features, validation_labels = process_data(validation_json_data, corpus, pos_tags, max_sentence_length)

# 将特征和标签转换为PyTorch张量
train_features_tensor = torch.tensor(train_features, dtype=torch.long)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
validation_features_tensor = torch.tensor(validation_features, dtype=torch.long)
validation_labels_tensor = torch.tensor(validation_labels, dtype=torch.float32)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

validation_dataset = TensorDataset(validation_features_tensor, validation_labels_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


# 定义神经网络模型
class ComplexModel(nn.Module):
    def __init__(self, input_size, hidden_size, max_sentence_length):
        super(ComplexModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * max_sentence_length, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入词性特征
        x = x.view(x.size(0), -1)  # 将嵌入的词性特征展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return torch.squeeze(x, -1)

input_size = len(pos_tags)  # 嵌入层输入尺寸等于词性标签数量
hidden_size = 256
model = ComplexModel(input_size, hidden_size, max_sentence_length)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 将模型移动到 GPU 上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
num_epochs = 100
train_losses = []
validation_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item() * inputs.size(0)
    validation_epoch_loss = validation_loss / len(validation_loader.dataset)
    validation_losses.append(validation_epoch_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {validation_epoch_loss:.4f}")

# 绘制训练和验证损失曲线
plt.plot(train_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 在测试集上评估模型
model.eval()
correct = 0
total = 0
true_positive = 0
false_positive = 0
false_negative = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        true_positive += ((predicted == 1) & (labels == 1)).sum().item()
        false_positive += ((predicted == 1) & (labels == 0)).sum().item()
        false_negative += ((predicted == 0) & (labels == 1)).sum().item()

accuracy = 100 * correct / total
precision = 0 if (true_positive + false_positive) == 0 else true_positive / (true_positive + false_positive)
recall = 0 if (true_positive + false_negative) == 0 else true_positive / (true_positive + false_negative)
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

print(f"Accuracy on the test set: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
