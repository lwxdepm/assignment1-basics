# train_bpe 伪代码（对应当前 byte-level BPE 实现）

> 对应代码：`cs336_basics/train_bpe.py`
> 目标：给定语料、词表大小和 special tokens，训练出 `vocab` 与 `merges`

---

## 1. 辅助函数：统计一个词内的相邻 pair

```text
函数 COLLECT_PAIR_COUNTS(word_token_ids):
    输入: word_token_ids  (例如 [99, 97, 116])
    输出: pair_counts     (例如 {(99,97):1, (97,116):1})

    创建空字典 pair_counts, 默认值为 0
    对 i 从 0 到 len(word_token_ids)-2:
        pair = (word_token_ids[i], word_token_ids[i+1])
        pair_counts[pair] += 1
    返回 pair_counts
```

---

## 2. 主函数：TRAIN_BPE

```text
函数 TRAIN_BPE(input_path, vocab_size, special_tokens):
    # ---------------------------------------------------
    # A. 读取语料
    # ---------------------------------------------------
    corpus = 读取 input_path 的全部文本(utf-8)

    # ---------------------------------------------------
    # B. 预分词并统计 pretoken 频次
    # ---------------------------------------------------
    pretoken_freq = Counter()  # 键: bytes 形式 pretoken, 值: 频次

    如果 special_tokens 非空:
        用 special_tokens 构建正则 special_splitter
        chunks = 用 special_splitter 对 corpus 做 split
        # 注意: split 结果不包含 special token 本体，避免 special 被继续拆分
    否则:
        chunks = [corpus]

    对每个 chunk in chunks:
        如果 chunk 为空: continue

        用 GPT2_PRETOKEN_PATTERN 在 chunk 中 finditer:
            token_text = 当前匹配字符串
            如果 token_text 非空:
                token_bytes = token_text.encode("utf-8")
                pretoken_freq[token_bytes] += 1

    # ---------------------------------------------------
    # C. 初始化 token 集合与“词”表示
    # ---------------------------------------------------
    token_bytes = [bytes([0]), bytes([1]), ..., bytes([255])]
    # 含义: 初始词表只有 256 个单字节 token

    words = []      # 每个元素是一个“词”的 token-id 序列
    word_freqs = [] # 与 words 同步下标，记录该词在语料中的频次

    对每个 (pretoken, freq) in pretoken_freq:
        如果 pretoken 非空:
            word_ids = list(pretoken)
            # 例如 b"cat" -> [99, 97, 116]
            words.append(word_ids)
            word_freqs.append(freq)

    # ---------------------------------------------------
    # D. 初始化全局 pair 统计与倒排索引
    # ---------------------------------------------------
    pair_freq = 默认值为 0 的字典
    # pair_freq[(a,b)] 表示 pair(a,b) 在整个语料中的加权频次

    pair_to_word_indices = 默认值为空集合的字典
    # pair_to_word_indices[(a,b)] = {出现该 pair 的 word 下标集合}

    对每个 word_idx, word in enumerate(words):
        如果 len(word) < 2: continue

        local_counts = COLLECT_PAIR_COUNTS(word)
        freq = word_freqs[word_idx]

        对每个 (pair, count_in_word) in local_counts:
            pair_freq[pair] += count_in_word * freq
            pair_to_word_indices[pair].add(word_idx)

    # ---------------------------------------------------
    # E. 计算最多可 merge 的次数
    # ---------------------------------------------------
    target_non_special_vocab_size = max(vocab_size - len(special_tokens), 0)
    max_merges = max(target_non_special_vocab_size - 256, 0)

    merges = []  # 顺序记录每一步 merge 的 (left_bytes, right_bytes)

    # ---------------------------------------------------
    # F. 迭代训练：每轮合并一个最高频 pair
    # ---------------------------------------------------
    重复 max_merges 轮:
        如果 pair_freq 为空:
            break

        best_pair = 选择 pair_freq 中“频次最大”的 pair
        若频次并列:
            使用 (left_bytes, right_bytes) 的字典序做稳定 tie-break

        (left_id, right_id) = best_pair

        merged_token = token_bytes[left_id] + token_bytes[right_id]
        new_token_id = len(token_bytes)
        token_bytes.append(merged_token)

        merges.append((token_bytes[left_id], token_bytes[right_id]))

        # 只更新受影响的词（包含 best_pair 的词）
        affected_words = pair_to_word_indices.get(best_pair, 空集合)

        对每个 word_idx in affected_words:
            old_word = words[word_idx]
            如果 len(old_word) < 2: continue
            freq = word_freqs[word_idx]

            # F1. 先移除 old_word 对全局统计的贡献
            old_pairs = COLLECT_PAIR_COUNTS(old_word)
            对每个 (pair, count_in_word) in old_pairs:
                pair_freq[pair] -= count_in_word * freq
                如果 pair_freq[pair] 变为 0:
                    删除 pair_freq[pair]

                从 pair_to_word_indices[pair] 中移除 word_idx
                若该集合为空:
                    删除 pair_to_word_indices[pair]

            # F2. 在 old_word 中执行 pair 替换，得到 new_word
            # 规则: 从左到右贪心，不重叠匹配 (left_id, right_id)
            new_word = []
            i = 0
            while i < len(old_word):
                如果 i+1 < len(old_word) 且 old_word[i]==left_id 且 old_word[i+1]==right_id:
                    new_word.append(new_token_id)
                    i += 2
                否则:
                    new_word.append(old_word[i])
                    i += 1

            words[word_idx] = new_word

            # F3. 把 new_word 对全局统计的贡献加回去
            如果 len(new_word) >= 2:
                new_pairs = COLLECT_PAIR_COUNTS(new_word)
                对每个 (pair, count_in_word) in new_pairs:
                    pair_freq[pair] += count_in_word * freq
                    pair_to_word_indices[pair].add(word_idx)

    # ---------------------------------------------------
    # G. 构建最终 vocab
    # ---------------------------------------------------
    vocab = 空字典
    token_id = 0

    # G1. 先放 special tokens
    对 token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1

    # G2. 再放 256 个单字节 token
    对 byte_value 从 0 到 255:
        vocab[token_id] = bytes([byte_value])
        token_id += 1

    # G3. 最后按训练顺序放 merges 生成的新 token
    对每个 (left_bytes, right_bytes) in merges:
        vocab[token_id] = left_bytes + right_bytes
        token_id += 1

    返回 (vocab, merges)
```

---

## 3. 核心思想（一句话版）

每一轮都找“在语料中最常一起出现”的相邻 token 对，把它们合并成一个新 token，并只对受影响的词做增量更新，直到达到词表容量上限或没有可合并 pair 为止。
