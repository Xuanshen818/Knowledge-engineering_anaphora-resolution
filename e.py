import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from sklearn.preprocessing import OneHotEncoder

# 读取数据集文件
with open('text.txt', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

# 读取词库文件
with open('onehot_encoded_words.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().splitlines()

vocab = np.array(vocab).reshape(-1, 1)

encoder = OneHotEncoder()
encoder.fit(vocab)

json_folder = 'C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train'
json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]

json_data = []
for i, json_file in enumerate(json_files):
    with open(json_file, 'r', encoding='gbk') as f:
        json_content = json.load(f)
        json_data.append(json_content)

# 生成特征向量和标签向量，并计算最长句子长度
max_sentence_length = 0

for i, data in enumerate(json_data):
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
features = []
labels = []

for i, data in enumerate(json_data):
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
    for word in words_before_pronoun:
        if word in encoder.categories_[0]:
            one_hot_encoded = encoder.transform([[word]]).toarray()[0]
        else:
            # 如果词汇不在词库中，则使用对应维度的全零向量
            one_hot_encoded = np.zeros(encoder.categories_[0].shape[0], dtype=np.float32)
        sentence_features.append(one_hot_encoded)

    # 补全特征向量使其与最长的句子等长
    padding_length = max_sentence_length - len(sentence_features)
    for _ in range(padding_length):
        sentence_features.append(np.zeros_like(encoder.categories_[0][0]))

    # 构造句子的标签向量
    for j in range(len(words_before_pronoun) + len(words_after_pronoun)):
        if (j >= antecedent_index_front and j < antecedent_index_behind) or (
                j >= pronoun_index_front and j < pronoun_index_behind):
            sentence_labels.append(1)
        else:
            sentence_labels.append(0)

    padding_length = max_sentence_length - len(sentence_labels)
    for _ in range(padding_length):
        sentence_labels.append(0)  # 对于填充部分，我们可以添加一个表示“非指代”的标签，即0

    # 将句子的特征向量和标签向量添加到总的特征和标签列表中
    features.append(sentence_features)
    labels.append(sentence_labels)

# 将特征向量和标签向量转换为 numpy 数组
features = np.array(features)
labels = np.array(labels)

# 检查特征向量和标签向量的维度是否匹配
print("特征向量维度:", features.shape)
print("标签向量维度:", labels.shape)

# 将数据划分为训练集和测试集
split = int(0.8 * len(features))
train_features, test_features = features[:split], features[split:]
train_labels, test_labels = labels[:split], labels[split:]

# 将数据转换为 PyTorch 张量
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
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
        return torch.squeeze(x, -1)


input_size = train_features_tensor.shape[1]  # 修改这里
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
for epoch in range(num_epochs):
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

    # 在测试集上评估模型
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100 * correct / total
    precision = 0 if (true_positive + false_positive) == 0 else true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    # 输出训练损失、测试集准确率、precision、recall和F1 score
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy on the test set: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

# 绘制训练损失曲线
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()


# 保存模型
torch.save(model.state_dict(), 'model.pth')
