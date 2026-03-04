from __future__ import annotations



from collections import Counter, defaultdict

from pathlib import Path



# 优先使用第三方 `regex` 库，是因为它支持 `\p{L}` / `\p{N}` 这种 Unicode 属性写法。

# 这和 GPT-2 常见实现保持一致，能够更准确地处理多语言文本。

try:

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

except ModuleNotFoundError:

    import re



    # 当环境中没有 `regex` 库时，退化到标准库 `re`。

    # 这里用替代写法模拟“字母/数字/符号/空白”分组逻辑。

    GPT2_PRETOKEN_PATTERN = re.compile(

        r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",

        flags=re.UNICODE,

    )





def _collect_pair_counts(word_token_ids: list[int]) -> dict[tuple[int, int], int]:

    """统计一个“词”（token id 序列）内部所有相邻 pair 的出现次数。



    例如：word_token_ids = [10, 20, 10, 20]

    相邻 pair 依次是：(10,20), (20,10), (10,20)

    返回：{(10,20): 2, (20,10): 1}

    """

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    # 相邻扫描：i 与 i+1 组成一个 pair

    for i in range(len(word_token_ids) - 1):

        pair_counts[(word_token_ids[i], word_token_ids[i + 1])] += 1

    return pair_counts





def train_bpe(

    input_path: str | Path,

    vocab_size: int,

    special_tokens: list[str],

) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    """训练一个 byte-level BPE 分词器。



    Args:

        input_path: 纯文本训练语料路径。

        vocab_size: 最终词表大小（包含 special tokens）。

        special_tokens: 特殊 token 列表。

            它们会被直接加入词表，且在 pre-tokenization 时会被“隔离”，

            防止其内容被继续拆分。



    Returns:

        vocab: token_id -> token_bytes 的映射。

        merges: 按训练顺序记录的 merge 操作列表，元素为

            (left_token_bytes, right_token_bytes)。



    训练流程概览：

    1) 读取语料并做 pre-tokenization，统计每个 pretoken 出现频次；

    2) 用 0~255 单字节作为初始 token；

    3) 反复找到全局最高频相邻 pair 并合并；

    4) 根据 special + bytes + merges 构建最终词表。

    """

    # 读入完整语料（一次性读入，便于离线作业场景实现简单直观）

    with open(input_path, "r", encoding="utf-8") as f:

        corpus = f.read()



    # pretoken_freq: 统计“预分词后子串（以 bytes 保存） -> 出现次数”

    pretoken_freq: Counter[bytes] = Counter()



    # 如果定义了 special tokens，先按 special token 进行切分。

    # split 后的 chunks 不包含 special token 本身，这样可确保 special token 不参与训练拆分。

    if special_tokens:

        special_splitter = re.compile("|".join(re.escape(token) for token in special_tokens))

        chunks = special_splitter.split(corpus)

    else:

        chunks = [corpus]



    # 对每个 chunk 做 GPT-2 风格 pre-tokenization，并统计频次。

    for chunk in chunks:

        if not chunk:

            continue

        for match in GPT2_PRETOKEN_PATTERN.finditer(chunk):

            token = match.group(0)

            if token:

                # BPE 的实现是 byte-level，因此统一转为 utf-8 bytes 再处理。

                pretoken_freq[token.encode("utf-8")] += 1



    # 初始 token 集合：256 个单字节 token（id 0~255）

    token_bytes: list[bytes] = [bytes([i]) for i in range(256)]



    # 把每个 pretoken 表示为“字节 id 列表”，并记录该词频次。

    # 例如 pretoken=b"cat" -> [99, 97, 116]

    words: list[list[int]] = []

    word_freqs: list[int] = []

    for pretoken, freq in pretoken_freq.items():

        if pretoken:

            words.append(list(pretoken))

            word_freqs.append(freq)

    # pair_freq: 全局 pair 频次（考虑每个词在语料中的出现次数）

    pair_freq: dict[tuple[int, int], int] = defaultdict(int)

    # pair_to_word_indices: 记录某个 pair 出现在哪些 word 中，用于增量更新。

    pair_to_word_indices: dict[tuple[int, int], set[int]] = defaultdict(set)

    # 初始化全局 pair 统计。

    for word_idx, word in enumerate(words):

        if len(word) < 2:

            continue

        local_pair_counts = _collect_pair_counts(word)

        freq = word_freqs[word_idx]

        for pair, count_in_word in local_pair_counts.items():

            # 词内次数 * 该词出现频次 = 对全局的贡献

            pair_freq[pair] += count_in_word * freq

            pair_to_word_indices[pair].add(word_idx)



    # 非 special token 可分配容量，至少是 0

    target_non_special_vocab_size = max(vocab_size - len(special_tokens), 0)

    # 已有 256 个 byte token，因此最多还能做这么多次 merge

    max_merges = max(target_non_special_vocab_size - 256, 0)



    # merges 按顺序记录每次合并的左右 bytes，这个顺序在推理阶段很重要。

    merges: list[tuple[bytes, bytes]] = []



    # 主循环：每次选择“最高频 pair”进行合并。

    for _ in range(max_merges):

        if not pair_freq:

            break



        # 选频次最大的 pair。

        # 频次相同时，用 pair 对应的 bytes 字典序做稳定 tie-break，保证结果可复现。

        best_pair = max(

            pair_freq.items(),

            key=lambda kv: (kv[1], token_bytes[kv[0][0]], token_bytes[kv[0][1]]),

        )[0]



        left_id, right_id = best_pair

        # 新 token 的 bytes 就是左右 token bytes 直接拼接

        merged_token = token_bytes[left_id] + token_bytes[right_id]

        new_token_id = len(token_bytes)

        token_bytes.append(merged_token)

        merges.append((token_bytes[left_id], token_bytes[right_id]))



        # 只重算受影响的词（包含 best_pair 的词），避免每轮全量扫描。

        affected_words = list(pair_to_word_indices.get(best_pair, set()))



        for word_idx in affected_words:

            old_word = words[word_idx]

            if len(old_word) < 2:

                continue



            # 先移除旧词对全局 pair 统计的贡献。

            old_pairs = _collect_pair_counts(old_word)

            freq = word_freqs[word_idx]



            for pair, count_in_word in old_pairs.items():

                new_count = pair_freq[pair] - count_in_word * freq

                if new_count:

                    pair_freq[pair] = new_count

                else:

                    pair_freq.pop(pair, None)



                # 维护 pair -> words 的倒排索引

                word_set = pair_to_word_indices.get(pair)

                if word_set is not None:

                    word_set.discard(word_idx)

                    if not word_set:

                        pair_to_word_indices.pop(pair, None)



            # 在该词中执行“左+右 -> 新 token”的贪心替换（从左到右，不重叠匹配）。

            new_word: list[int] = []

            i = 0

            n = len(old_word)

            while i < n:

                if i + 1 < n and old_word[i] == left_id and old_word[i + 1] == right_id:

                    new_word.append(new_token_id)

                    i += 2

                else:

                    new_word.append(old_word[i])

                    i += 1



            words[word_idx] = new_word



            # 再把新词对全局统计的贡献加回去（增量更新完成）。

            if len(new_word) >= 2:

                new_pairs = _collect_pair_counts(new_word)

                for pair, count_in_word in new_pairs.items():

                    pair_freq[pair] += count_in_word * freq

                    pair_to_word_indices[pair].add(word_idx)



    # 构建最终词表：

    # 1) special tokens（排在最前）

    # 2) 256 个 byte tokens

    # 3) 按 merge 顺序加入的新 token

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

