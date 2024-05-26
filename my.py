import os
import json
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

# 定义数据集类
class CoreferenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pronoun_vector, antecedent_vector, previous_word_vector, pronoun_number, pronoun_sentence = self.data[idx]
        return {
            "pronoun_vector": torch.tensor(pronoun_vector, dtype=torch.float32),
            "antecedent_vector": torch.tensor(antecedent_vector, dtype=torch.float32),
            "previous_word_vector": torch.tensor(previous_word_vector, dtype=torch.float32),
            "pronoun_number": torch.tensor(pronoun_number, dtype=torch.float32),
            "pronoun_sentence": torch.tensor(pronoun_sentence, dtype=torch.long)
        }

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

# 定义神经网络模型
class CoreferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CoreferenceModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)  # Change the output dimension of the first linear layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Change the input dimension of the second linear layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pronoun_vector, antecedent_vector, previous_word_vector, pronoun_sentence):
        # Embedding the pronoun sentence
        pronoun_sentence = self.embedding(pronoun_sentence)

        # Take the mean of embeddings to get a fixed size vector
        pronoun_sentence = torch.mean(pronoun_sentence, dim=1)

        # Concatenate all vectors
        x = torch.cat((pronoun_vector, antecedent_vector, previous_word_vector, pronoun_sentence), dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# 定义数据集类
class CoreferenceDataset(Dataset):
    def __init__(self, json_dir, corpus_file, noun_vocab_file):
        self.data = self.read_dataset(json_dir, corpus_file, noun_vocab_file)

    def read_dataset(self, json_dir, corpus_file, noun_vocab_file):
        data = []

        # 读取名词词汇表
        with open(noun_vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.readlines()
            vocab = [word.strip() for word in vocab]

        # 读取语料库
        with open(corpus_file, 'r', encoding='gbk') as f:
            corpus = f.readlines()

        # 读取 JSON 文件
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(json_dir, filename)
                with open(file_path, 'r', encoding='gbk') as file:
                    json_data = json.load(file)
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

                    pronoun_sentence = corpus[pronoun_sentence_id].strip().split()
                    antecedent_sentence = corpus[antecedent_sentence_id].strip().split()

                    # 提取代词和先行词
                    pronoun = pronoun_sentence[pronoun_index]
                    antecedent = antecedent_sentence[antecedent_index]

                    # one-hot 编码代词和先行词
                    pronoun_vector = self.word_to_onehot(pronoun, vocab)
                    antecedent_vector = self.word_to_onehot(antecedent, vocab)

                    # 添加数据
                    pronoun_number = 1 if antecedent_num == 1 else 0
                    pronoun_sentence = [vocab.index(word) for word in pronoun_sentence if word in vocab]
                    previous_word_vector = np.zeros(len(vocab), dtype=int)
                    if pronoun_sentence:
                        previous_word_vector[pronoun_sentence[-1]] = 1
                    data.append((pronoun_vector, antecedent_vector, previous_word_vector, pronoun_number, pronoun_sentence))

        return data

    def word_to_onehot(self, word, vocab):
        onehot = np.zeros(len(vocab), dtype=int)
        if word in vocab:
            onehot[vocab.index(word)] = 1
        return onehot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pronoun_vector, antecedent_vector, previous_word_vector, pronoun_number, pronoun_sentence = self.data[idx]
        return {
            "pronoun_vector": torch.tensor(pronoun_vector, dtype=torch.float32),
            "antecedent_vector": torch.tensor(antecedent_vector, dtype=torch.float32),
            "previous_word_vector": torch.tensor(previous_word_vector, dtype=torch.float32),
            "pronoun_number": torch.tensor(pronoun_number, dtype=torch.float32),
            "pronoun_sentence": torch.tensor(pronoun_sentence, dtype=torch.long)
        }


# 文件路径
json_dir = "C:\\Users\\86135\\Desktop\\知识工程\\coref-dataset\\coref-dataset\\train"  # JSON 文件夹路径
corpus_file = "C:\\Users\\86135\\Desktop\\知识工程\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\1998-01-2003data.txt"   # 语料库文件路径
noun_vocab_file = "D:\\Python 3.11\\a PyCharm\\pythonProject3\\n_sort_delete_number.txt"  # 名词词汇表文件路径

# 准备数据集
dataset = CoreferenceDataset(json_dir, corpus_file, noun_vocab_file)
print("Number of samples in dataset:", len(dataset))
print("data" , dataset)

if len(dataset) == 0:
    print("Error: Dataset is empty. Check your data loading process.")
else:
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    # 定义神经网络模型
    class CoreferenceModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(CoreferenceModel, self).__init__()
            self.embedding = nn.Embedding(input_dim, hidden_dim)
            self.fc1 = nn.Linear(hidden_dim * 3 + hidden_dim, hidden_dim)  # 修改输入维度
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, pronoun_vector, antecedent_vector, previous_word_vector, pronoun_sentence):
            pronoun_sentence = self.embedding(pronoun_sentence).mean(dim=1)
            x = torch.cat((pronoun_vector, antecedent_vector, previous_word_vector), dim=1)
            x = torch.cat((x, pronoun_sentence), dim=1)
            x = self.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x.squeeze()


    # 初始化模型
    input_dim = len(vocab)
    hidden_dim = 128
    output_dim = 1
    model = CoreferenceModel(input_dim, hidden_dim, output_dim)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            pronoun_vector = batch["pronoun_vector"]
            antecedent_vector = batch["antecedent_vector"]
            previous_word_vector = batch["previous_word_vector"]
            pronoun_number = batch["pronoun_number"]
            pronoun_sentence = batch["pronoun_sentence"]

            optimizer.zero_grad()
            output = model(pronoun_vector, antecedent_vector, previous_word_vector, pronoun_sentence)
            loss = criterion(output, pronoun_number)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.round(output)
            correct += (predicted == pronoun_number).sum().item()
            total += pronoun_number.size(0)

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, total_loss / total,
                                                                      100 * correct / total))
