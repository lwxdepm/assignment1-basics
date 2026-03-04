# cs336_basics/train_bpe.py 
from collections import defaultdict

import regex as re
    # GPT-2 风格的 pre-tokenization 正则：
    # - 英语缩写片段（'s, 're, ...）
    # - 可选前导空格 + 字母串
    # - 可选前导空格 + 数字串
    # - 可选前导空格 + 其它符号串
    # - 各类空白
    GPT2_PRETOKEN_PATTERN = re.compile(
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = None):
    
    # 假设我们要把读到的文本赋值给变量 corpus
    with open(input_path, mode="r", encoding="utf-8") as f:
    # f.read() 会一次性把整个文件读成一个完整的字符串
        corpus = f.read()
    # 出了 with 的缩进块后，f 会被自动关闭
    # 你可以接着处理 corpus 了

    pretoken_freq: dict[bytes, int] = {}

    if special_tokens:
        # 第一步：用 special_tokens 构建正则 special_splitter
        # 核心技巧 1：用 re.escape 防止 '<', '|', '>' 被当成正则特殊符号
        # 核心技巧 2：用 '|' 将它们拼接成一个 "或" 的匹配模式
        special_splitter = "|".join(re.escape(token) for token in special_tokens)
        
        # 第二步：对 corpus 做 split
        # 核心技巧 3：不在正则外层加括号 ()，这样 split 的结果就不会包含 special token 本体
        chunks = re.split(special_splitter, corpus)
        
        # 可选优化：过滤掉切分后可能产生的空字符串 ""
        chunks = [chunk for chunk in chunks if chunk]
    else:
        # 否则直接放入列表
        chunks = [corpus]

    # 第一层循环：处理被特殊 Token（如 <|endoftext|>）切开后的文本块
    for chunk in chunks:
        
        # 第二层循环：利用 GPT-2 的正则表达式模式，在当前文本块中寻找匹配项
        # finditer 是高效的选择，它返回一个迭代器，避免一次性在内存中创建巨大的列表
        for match in GPT2_PRETOKEN_PATTERN.finditer(chunk):
            
            # 获取匹配到的字符串片段（例如：" Hello", "'s", " 123", "!" 等）
            token = match.group(0)
            
            # 健壮性检查：确保匹配到的内容不为空字符串
            if token:
                # --- 关键工程步骤：Byte-level 转换 ---
                # 动机 1：GPT-2 采用 Byte-level BPE。通过 encode("utf-8") 将字符串转为原始字节序列。
                # 动机 2：解决 Out-of-Vocabulary (OOV) 问题。Unicode 字符成千上万，
                #        但 UTF-8 编码后的字节只有 256 种可能（0-255），
                #        这保证了词表的基础单元是有限且完备的。
                
                # 使用 bytes 对象作为字典的 Key，记录该片段在整个语料库中出现的次数
                # pretoken_freq 的结构示例：{ b' Hello': 1500, b' world': 800, b'!': 300 }
                pretoken_freq[token.encode("utf-8")] += 1


    # 初始 token 集合：256 个单字节 token（id 0~255）
    token_bytes: list[bytes] = [bytes([i]) for i in range(256)]

    words = []      # 每个元素是一个“词”的 token-id 序列
    words_freqs = [] # 与 words 同步下标，记录该词在语料中的频次

    # 遍历字典中的每一个预分词片段（bytes）及其出现频次（int）
    for pretoken, freq in pretoken_freq.items(): # 【修正点】：加上了 ()
        
        # 防御性编程：过滤掉可能为空的字节序列（b''），防止后续合并逻辑报错
        if pretoken:
            
            # --- 核心数据转换：Bytes -> Token IDs ---
            # pretoken 是 bytes 类型，例如 b'Hello'。
            # 调用 list(pretoken) 会自动将其中的每个字节转换为对应的 0~255 的整型（Integer）。
            # 这一步极其关键，它把底层的“字节”映射到了初始的 256 个“Token ID”上。
            # 举例：b' Hi' -> 转换后变成 -> [32, 72, 105] (分别是 空格, H, i 的 ASCII/Byte 值)
            word_ids = list(pretoken)
            
            # 将转换好的 Token ID 列表放入 words 中备用
            words.append(word_ids)
            
            # 将该词的频次放入 words_freqs 相同下标的位置
            words_freqs.append(freq)


    pair_freq = defaultdict(int)

    pair_to_word_indices = defaultdict(set)

    # 遍历词表中的每一个“词”（这里的 word 是一个 Token ID 序列，例如 [72, 101, 108...]）
    for word_idx, word in enumerate(words):
        
        # 剪枝优化：如果这个词的长度不到 2，根本凑不出一个 Pair，直接跳过
        if len(word) < 2:
            continue
            
        else:
            # collect_pair_counts 是一个辅助函数，用于统计【当前这一个词】内部，各个 pair 出现了几次。
            # 例如 word 是 [108, 108, 108]，它会返回 {(108, 108): 2}
            temp_counts = collect_pair_counts(word)
            
            # 获取这个词在整个语料库中的出现频次
            freq = word_freqs[word_idx]

            # 将当前词内部的 pair 统计结果，累加到全局统计表中
            for pair, num in temp_counts.items():
                
                # 核心数学逻辑：全局频次 = (该 pair 在本词中出现的次数) * (本词在语料中出现的总次数)
                pair_freq[pair] += num * freq
                
                # 建立反向索引：将当前词的下标 (word_idx) 加入到该 pair 的集合中
                pair_to_word_indices[pair].add(word_idx)


    # 非 special token 可分配容量，至少是 0
    target_non_special_vocab_size = max(vocab_size - len(special_tokens), 0)
    # 已有 256 个 byte token，因此最多还能做这么多次 merge
    max_merges = max(target_non_special_vocab_size - 256, 0)
    new_token_id = 256  # 新 Token 的 ID 从 256 开始分配
    merges = []  # 记录合并历史

    for i in range(max_merges):
        if not pair_freq:
            break
        else:
            best_pair = max(pair_freq, key=pair_freq.get)

            # 记录本次合并
            merges.append(best_pair)
            # 记录新 token 的字节内容（将两个旧 token 的 bytes 拼起来）
            new_token_bytes = token_bytes[best_pair[0]] + token_bytes[best_pair[1]]
            token_bytes.append(new_token_bytes)

            # 2. 【性能核心】通过反向索引，获取所有包含这个 best_pair 的词的下标
            # 如果不这样做，你就得遍历所有十几万个词，速度会慢几百倍
            affected_words = pair_to_word_indices[best_pair]

            # 3. 只对这些包含 best_pair 的词进行更新
            for word_idx in words_to_update:
                old_word = words[word_idx]
                freq = words_freqs[word_idx]
                
                # --- 步骤 3.1：扣除旧词对全局频次的贡献 ---
                old_pairs = collect_pair_counts(old_word)
                for pair, count in old_pairs.items():
                    pair_freq[pair] -= count * freq
                    # 注意：为了性能，我们通常不在此刻去清理 pair_to_word_indices 里的无用索引，
                    # 因为即使索引存在，只要 pair_freq 为 0，它就不会被选为 best_pair。

                # --- 步骤 3.2：执行合并，生成新词 ---
                new_word = merge(old_word, best_pair, new_token_id)
                words[word_idx] = new_word  # 原地更新词表

                # --- 步骤 3.3：加上新词对全局频次的贡献 ---
                new_pairs = collect_pair_counts(new_word)
                for pair, count in new_pairs.items():
                    pair_freq[pair] += count * freq
                    pair_to_word_indices[pair].add(word_idx)  # 更新新 pair 的反向索引

            # 4. 这个 best_pair 已经被合并完了，从频率表中彻底删除它
            del pair_freq[best_pair]
            
            # 准备下一个新 Token 的 ID
            new_token_id += 1

    vocab: dict[int, bytes] = {}
    token_id = 0

    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1

    for byte_value in range(256):
        vocab[token_id] = bytes([byte_value])
        token_id += 1

    for left, right in merges:
        vocab[token_id] = left + right
        token_id += 1

    return vocab, merges


def merge(word, best_pair, new_token_id):
    """
    在单个 Token ID 序列中执行合并操作。
    例如: word=[72, 101, 108], best_pair=(72, 101), new_token_id=256
    返回: [256, 108]
    """
    new_word = []
    i = 0
    while i < len(word):
        # 检查是否匹配到了 best_pair
        if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
            new_word.append(new_token_id)
            i += 2  # 匹配成功，指针跳过这两个元素
        else:
            new_word.append(word[i])
            i += 1  # 没匹配上，保留原 ID，指针走一步
    return new_word

def collect_pair_counts(word):
    pair_counts = dict()
    
    i = 0
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        if pair in pair_counts:
            pair_counts[pair] += 1
        else:
            pair_counts[pair] = 1
    return pair_counts