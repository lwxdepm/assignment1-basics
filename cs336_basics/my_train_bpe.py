# ==========================================
# 阶段 1：初始化
# ==========================================
# 假设我们已经统计好了文本里的词频，并且把每个词拆成了单个字符（加上空格隔开）
# 结尾加上 </w> 表示单词结束符，防止 cross-word 合并（比如把a的结尾和b的开头合并）

import collections


# 你的初始字典
vocab = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3
}

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        word = word.split()
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pairs[pair] += freq
    return pairs

# 看看哪两个字符挨在一起最频繁？
pairs = get_stats(vocab)
best_pair = max(pairs, key=pairs.get)
print(f"最频繁的字符对是: {best_pair}, 出现次数: {pairs[best_pair]}")


def merge_vocab(best_pair, vocab):
    # 这里用普通的 dict 即可，因为我们是直接赋值新词，不需要累加
    new_vocab = {}

    for word_str, freq in vocab.items():
        # 最好换个名字，不然会覆盖原来的字符串 word_str
        word = word_str.split()
        
        # 【修正 1】定义临时的 new_tokens 列表
        new_tokens = []
        
        # 【修正 2】使用 while 循环来手动控制指针 i
        i = 0
        while i < len(word):
            # 【修正 3】把边界检查 i < len(word) - 1 放在最前面，防止越界
            if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                # 匹配上了！塞入 new_tokens 列表
                new_tokens.append(best_pair[0] + best_pair[1])
                # 指针往前跳 2 步
                i += 2
            else:
                # 没匹配上，原样塞入
                new_tokens.append(word[i])
                # 指针往前跳 1 步
                i += 1

        # 把新列表拼成字符串，存入字典
        new_word = " ".join(new_tokens)
        new_vocab[new_word] = freq
        
    return new_vocab


print(merge_vocab(best_pair, vocab))


# ==========================================
# 阶段 2：BPE 核心循环
# ==========================================

num_merges = 10


for i in range(num_merges):
    
    # 动作 A：统计当前词典里，所有相邻字符对的出现频率
    # 预期输出例子：{ ('e', 's'): 9, ('s', 't'): 9, ('l', 'o'): 7 ... }
    pair_freqs = get_stats(vocab) 
    
    # 边界情况：如果找不到相邻对了（都合并成一坨了），就提前结束
    if not pair_freqs:
        break
        
    # 动作 B：找出频率最高的那个相邻对
    # 比如找到了 ('e', 's') 出现了 9 次
    best_pair = max(pair_freqs, key=pair_freqs.get)
    
    # 动作 C：在整个词典中，把这个 best_pair 合并成一个新词
    # 比如把所有的 "e s" 替换成 "es"
    vocab = merge_vocab(best_pair, vocab)
    
    # 记录下这条合并规则（面试点：推理部署的时候，遇到新词也要按这个规则合并）
    print(f"Merge {i}: {best_pair[0]} + {best_pair[1]} -> {best_pair[0]}{best_pair[1]}")

print("最终词典:", vocab)

