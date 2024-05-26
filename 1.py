import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix

# 读取数据集文件
with open('text.txt', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

# 读取词库文件
with open('onehot_encoded_words.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().splitlines()

vocab = np.array(vocab).reshape(-1, 1)

encoder = OneHotEncoder()
encoder.fit(vocab)

json_folder_train = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train'
json_files_train = [os.path.join(json_folder_train, f) for f in os.listdir(json_folder_train) if f.endswith('.json')]

json_data_train = []
for i, json_file in enumerate(json_files_train):
    with open(json_file, 'r', encoding='gbk') as f:
        try:
            json_content = json.load(f)
            json_data_train.append(json_content)
        except json.JSONDecodeError:
            print(f"Error reading JSON file: {json_file}")

# 生成特征向量和标签向量，并计算最长句子长度
max_sentence_length = 0

for i, data in enumerate(json_data_train):
    if data and "0" in data and "id" in data["0"]:
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
        sentence_length = len(words_before_pronoun) + len(words_after_pronoun)
        max_sentence_length = max(max_sentence_length, sentence_length)

# 生成特征向量和标签向量，并进行填充
features_train = []
labels_train = []

for i, data in enumerate(json_data_train):
    if data and "0" in data and "id" in data["0"]:
        json_id = data["0"]["id"].replace('-', '')
        pronoun_index_front = data['pronoun']['indexFront']
        pronoun_index_behind = data['pronoun']['indexBehind']
        antecedent_index_front = data['0']['indexFront']
        antecedent_index_behind = data['0']['indexBehind']

        # 初始化句子的特征向量和标签向量
        sentence_features = []
        sentence_labels = []

        # 获取句子并提取词汇
        for line in corpus:
            if json_id in line:
                sentence = line.strip().split(" ", 1)[1]
                break

        words_before_pronoun = sentence[:pronoun_index_front].split()
        words_after_pronoun = sentence[pronoun_index_behind:].split()

        # 构造句子的特征向量
        sentence_feature = []
        for word in words_before_pronoun:
            if word in encoder.categories_[0]:
                one_hot_encoded = encoder.transform([[word]]).toarray()[0]
            else:
                # 如果词汇不在词库中，则使用对应维度的全零向量
                one_hot_encoded = np.zeros(encoder.categories_[0].shape[0], dtype=np.float32)
            sentence_feature.append(one_hot_encoded)
        features_train.append(sentence_feature)

        # 构造句子的标签向量
        sentence_label = []
        for j in range(len(words_before_pronoun) + len(words_after_pronoun)):
            if (j >= antecedent_index_front and j < antecedent_index_behind) or (
                    j >= pronoun_index_front and j < pronoun_index_behind):
                sentence_label.append(1)
            else:
                sentence_label.append(0)
        labels_train.append(sentence_label)

# 转换为稀疏矩阵
features_train = coo_matrix(np.array(features_train))

# 找到最长的标签数组长度
max_label_length_train = max(len(label) for label in labels_train)

# 填充所有标签数组
padded_labels_train = []
for label in labels_train:
    padded_label = np.zeros(max_label_length_train)
    padded_label[:len(label)] = label
    padded_labels_train.append(padded_label)

labels_train = np.array(padded_labels_train)
features_train = features_train.toarray()

# 将数据划分为训练集和验证集
split = int(0.8 * len(features_train))
train_features, val_features = features_train[:split], features_train[split:]
train_labels, val_labels = labels_train[:split], labels_train[split:]

# 将数据转换为 PyTorch 张量
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 定义神经网络模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return torch.squeeze(x, -1)


input_size = train_features_tensor.shape[1]
hidden_size = 64
model = SimpleModel(input_size, hidden_size)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型移动到 GPU 上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
num_epochs = 500
train_losses = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1_scores = []
for epoch in range(num_epochs):
    # 训练模型
    model.train()  # 将模型设置为训练模式
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

    # 在验证集上评估模型
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item() * inputs.size(0)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_epoch_loss)

    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    precision = 0 if (true_positive + false_positive) == 0 else true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    val_precisions.append(precision)
    val_recalls.append(recall)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    val_f1_scores.append(f1_score)

    # 输出训练损失和验证集性能指标
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

# 绘制训练损失曲线和验证集性能指标曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Accuracy')
plt.plot(val_precisions, label='Precision')
plt.plot(val_recalls, label='Recall')
plt.plot(val_f1_scores, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Percentage')
plt.title('Validation Metrics')
plt.legend()

plt.show()

# 保存模型
torch.save(model.state_dict(), 'model.pth')
