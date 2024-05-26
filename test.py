import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 读取词汇表文件
def read_vocabulary(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        vocabulary = [line.strip() for line in file]
    return vocabulary

# 对词语进行 one-hot 编码
def one_hot_encode(word, vocab):
    word_to_index = {word: i for i, word in enumerate(vocab)}
    one_hot = np.zeros(len(vocab), dtype=int)
    if word in word_to_index:
        one_hot[word_to_index[word]] = 1
    return one_hot

# 保存编码后的词语到文件
def save_encoded_words(encoded_words, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word in encoded_words:
            file.write(' '.join(map(str, word)) + '\n')

# 文件路径
input_filename = "n_sort_delete_number.txt"  # 输入文件名
output_filename = "onehot_encoded_words.txt"  # 输出文件名

# 读取词汇表
vocab = read_vocabulary(input_filename)

# 读取词语并进行 one-hot 编码
encoded_words = []
with open(input_filename, 'r', encoding='utf-8') as file:
    for line in file:
        word = line.strip()
        encoded_word = one_hot_encode(word, vocab)
        encoded_words.append(encoded_word)

# 保存编码后的词语到文件
save_encoded_words(encoded_words, output_filename)

print("One-hot 编码完成并已保存到文件:", output_filename)
















# 示例数据
class PronounDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

example_data = [
    {
        "pronoun_vector": np.array([1, 0, 0, 0, 0]),  # 代词的 one-hot 编码向量
        "antecedent_vector": np.array([0, 1, 0, 0, 0]),  # 先行词的 one-hot 编码向量
        "previous_word_vector": np.array([0, 0, 1, 0, 0]),  # 上一个词的 one-hot 编码向量
        "pronoun_number": np.array([1], dtype=np.float32),  # 代词数量的标签
        "pronoun_sentence": np.array([0, 1, 0, 0, 0])  # 代词所在句子的词汇表索引序列
    },
    {
        "pronoun_vector": np.array([0, 1, 0, 0, 0]),
        "antecedent_vector": np.array([0, 0, 1, 0, 0]),
        "previous_word_vector": np.array([0, 0, 0, 1, 0]),
        "pronoun_number": np.array([0], dtype=np.float32),
        "pronoun_sentence": np.array([0, 0, 1, 0, 0])
    },
    # 添加更多示例数据...
]

class PronounModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PronounModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, pronoun_vector, antecedent_vector, previous_word_vector, pronoun_sentence):
        x = torch.cat((pronoun_vector, antecedent_vector, previous_word_vector, pronoun_sentence), dim=1)  # 正确的连接方式
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def train_model(model, criterion, optimizer, dataloader, num_epochs=10000):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for data in dataloader:
            pronoun_vector = torch.tensor(data["pronoun_vector"], dtype=torch.float32, requires_grad=True)
            antecedent_vector = torch.tensor(data["antecedent_vector"], dtype=torch.float32, requires_grad=True)
            previous_word_vector = torch.tensor(data["previous_word_vector"], dtype=torch.float32, requires_grad=True)
            pronoun_number = torch.tensor(data["pronoun_number"], dtype=torch.float32, requires_grad=True)
            pronoun_sentence = torch.tensor(data["pronoun_sentence"], dtype=torch.float32)

            optimizer.zero_grad()

            outputs = model(pronoun_vector, antecedent_vector, previous_word_vector, pronoun_sentence)
            loss = criterion(outputs.view(-1), pronoun_number.view(-1))  # 调整目标维度与模型输出相匹配
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.round(outputs)
            total += pronoun_number.size(0)
            correct += (predicted == pronoun_number.view(-1)).sum().item()  # 调整正确预测的计算方式
            print(running_loss)

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, epoch_loss, epoch_accuracy))

# 创建数据加载器
dataset = PronounDataset(example_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 定义模型参数
input_size = 20  # 输入向量的长度
hidden_size = 10  # 隐藏层大小
output_size = 1  # 输出大小（代表是否为单数）

# 实例化模型
model = PronounModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.05)  # Adam优化器，学习率为0.001

# 训练模型
train_model(model, criterion, optimizer, dataloader, num_epochs=50)
