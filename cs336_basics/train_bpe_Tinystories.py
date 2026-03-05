import time
import tracemalloc
import json
import os
from pathlib import Path

# 从你之前实现的模块中导入核心训练函数
from train_bpe import train_bpe

def main():
    # --- 1. 配置参数 ---
    # 替换为你实际下载的 TinyStories 数据集路径
    input_file = "/root/autodl-tmp/data/TinyStoriesV2-GPT4-train.txt" 
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # 防御性检查
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 {input_file}。请先准备好语料！")
        return

    # --- 2. 启动监控 (Profiling) ---
    print(f"🚀 开始训练 BPE 词表 (目标大小: {vocab_size})...")
    tracemalloc.start() # 开启内存分配追踪
    start_time = time.perf_counter() # 高精度计时起点

    # --- 3. 执行核心算法 ---
    # 调用 train_bpe.py 中的逻辑，返回 vocab 和 merges
    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)

    # --- 4. 结束监控与指标计算 ---
    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_time = end_time - start_time
    peak_mem_gb = peak_mem / (1024 ** 3) # 转换为 GB

    print("\n" + "="*40)
    print("📊 训练指标汇报")
    print("="*40)
    print(f"⏱️ 训练耗时: {elapsed_time:.2f} 秒")
    print(f"💾 峰值内存占用: {peak_mem_gb:.4f} GB")

    # --- 5. 提取最长 Token ---
    # 核心逻辑：遍历 vocab 的 values，找出字节长度最长的那一个
    longest_token_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    
    # 踩坑预警：由于 BPE 是基于字节的，某些 token 可能不是合法的 UTF-8 字符串（比如被截断的中文/Emoji）。
    # 使用 errors="replace" 可以在遇到非法字节时用 '' 替代，防止程序崩溃。
    longest_token_str = longest_token_bytes.decode("utf-8", errors="replace")
    print(f"📏 最长 Token 内容: {longest_token_str!r}")
    print(f"📏 最长 Token 长度: {len(longest_token_bytes)} bytes")

    # --- 6. 序列化与持久化 ---
    print("\n💾 正在序列化 vocab 和 merges...")
    
    # 6.1 保存 Vocab 为 JSON
    # JSON 的 key 必须是字符串，value 我们也将 bytes 解码为字符串
    vocab_serializable = {
        str(k): v.decode("utf-8", errors="replace") 
        for k, v in vocab.items()
    }
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)
        
    # 6.2 保存 Merges 为 TXT
    # 格式通常是 "左token 右token" 按行分隔
    with open("merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = left.decode("utf-8", errors="replace")
            right_str = right.decode("utf-8", errors="replace")
            f.write(f"{left_str} {right_str}\n")
            
    print("✅ 序列化完成！已生成 vocab.json 和 merges.txt")
    print("="*40)

if __name__ == "__main__":
    main()