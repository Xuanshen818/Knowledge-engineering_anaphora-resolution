首先先解释主要部分（以词性为特征）
main2_1是主要代码，delete[ 是删除名词前的括号方便合并。
delete_number是删除词频的数量，方便进行独热编码。
deletewrongjson是为了删除空json文件
onehotnew是onehot编码所用，有个不太好用的老版本，所以这个叫new
remove是去重所用，提取名词之后合并名词
sortandout是统计词频
switch2txt是用来寻找有问题的json文件的
switch2wordtxt是用来提取名词的

上述包含了词频所用预处理，词性用不到这么多

我的词频代码单独放到了一个文件夹里main.py（不是这次作业所用，只是另一个思路，不太好我觉得）
这里赘述一下，其实词频的我已经写完了，不过运行需要20多个G的内存，电脑根本跑不动，我曾经想过限制向量补充0的数量，或者随机下采样，但是都不太理想
所以就换成了词性的
