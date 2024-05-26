                                   #
def count_word_occurrences(input_file):
    word_counts = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    return word_counts

def sort_and_write_counts(counts, output_file):
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in sorted_counts:
            f.write(f"{word} {count}\n")

# 统计词的出现次数
input_file = 'n.txt'
word_counts = count_word_occurrences(input_file)

# 对出现次数进行排序并写入文件
output_file = 'n_sort.txt'
sort_and_write_counts(word_counts, output_file)

