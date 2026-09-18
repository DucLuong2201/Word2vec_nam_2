import re
from collections import Counter
import numpy as np


def preprocess(text, min_count=5):
    """
    Tiền xử lý văn bản thô cho pipeline huấn luyện Word2Vec.

    Các bước thực hiện:
        1. Chuyển chữ thường, tách từ (chỉ giữ chữ cái a-z).
        2. Đếm tần suất toàn bộ từ.
        3. Xây dựng vocab: giữ từ có tần suất >= min_count, thêm <UNK> ở đầu.
        4. Chuyển danh sách từ thành danh sách index, từ ngoài vocab -> <UNK>.
        5. Xây dựng freq_dict cho NegativeSampler (không gồm <UNK>).

    Args:
        text      (str): Văn bản thô đầu vào.
        min_count (int): Ngưỡng tần suất tối thiểu để giữ từ trong vocab.

    Returns:
        token_indices (list[int]) : Danh sách index của các token sau khi ánh xạ.
        word_to_idx   (dict)      : Ánh xạ word -> index.
        idx_to_word   (dict)      : Ánh xạ index -> word.
        freq_dict     (dict)      : Tần suất các từ trong vocab, dạng {index: count},
                                    dùng cho NegativeSampler (không chứa <UNK>).
        word_counts   (Counter)   : Tần suất gốc của toàn bộ từ trước khi lọc vocab.
    """
    # Bước 1: Chuyển chữ thường, tách từ (chỉ giữ chữ cái)
    text = text.lower()
    words = re.findall(r'[a-z]+', text)

    # Bước 2: Đếm tần suất toàn bộ
    word_counts = Counter(words)

    # Bước 3: Xây dựng vocab — chỉ giữ từ >= min_count, thêm <UNK> ở đầu
    vocab = ["<UNK>"]
    vocab.extend(sorted([w for w, c in word_counts.items() if c >= min_count]))

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # Bước 4: Chuyển words -> token indices, từ ngoài vocab -> <UNK>
    unk_idx = word_to_idx["<UNK>"]
    token_indices = [word_to_idx.get(w, unk_idx) for w in words]

    # Bước 5: freq_dict chỉ chứa từ trong vocab (trừ <UNK>)
    # Dùng sau này cho NegativeSampler
    freq_dict = {word_to_idx[w]: word_counts[w] for w in vocab if w != "<UNK>"}

    return token_indices, word_to_idx, idx_to_word, freq_dict, word_counts


def generate_training_pairs(tokens, window_size=2):
    """
    Sinh tất cả cặp (center_idx, context_idx) từ danh sách token indices.

    Với mỗi token tại vị trí i, lấy tất cả token trong cửa sổ
    [i - window_size, i + window_size] (trừ chính i) làm context.

    Args:
        tokens      (list[int]): Danh sách index token sau preprocess.
        window_size (int)      : Bán kính cửa sổ ngữ cảnh (mỗi phía).

    Returns:
        pairs (list[tuple[int, int]]): Danh sách cặp (center_idx, context_idx).
    """
    pairs = []
    n = len(tokens)

    for i in range(n):
        center = tokens[i]
        start = max(0, i - window_size)         # Giới hạn trái của cửa sổ
        end = min(n, i + window_size + 1)       # Giới hạn phải của cửa sổ
        for j in range(start, end):
            if i != j:                          # Bỏ qua chính center word
                pairs.append((center, tokens[j]))
    return pairs


if __name__ == "__main__":
    # Đọc file
    print("Đang đọc file")
    with open("../data/text8", encoding="utf-8") as f:
        text = f.read()

    token_indices, word_to_idx, idx_to_word, freq_dict, word_counts = preprocess(text, min_count=5)
    pairs = generate_training_pairs(token_indices, window_size=2)

    # B1.2: Báo cáo corpus
    print("\n" + "=" * 50)
    print("  BÁO CÁO CORPUS")
    print("=" * 50)
    print(f"\n  Số từ trong từ điển   : {len(freq_dict):,} từ")
    print(f"  Tổng số cặp huấn luyện: {len(pairs):,} cặp")

    print(f"\n  Top 5 từ phổ biến nhất:")
    print(f"  {'Từ':<15} {'Tần suất':>10}")
    print("  " + "-" * 28)
    count = 0
    for w, c in word_counts.most_common():
        if w in word_to_idx and w != "<UNK>":
            print(f"  {w:<15} {c:>10,}")
            count += 1
            if count == 5:
                break

    print("=" * 50)