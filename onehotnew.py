import numpy as np

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
            print(word)

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

print("One-hot 编码完成并已保存到文件:  ", output_filename)
