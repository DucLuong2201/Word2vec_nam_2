import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

# =============================================================================
# Yêu cầu C2.1, C2.2, C2.3 (PDF trang 7)
# Các hàm đánh giá intrinsic cho Word2Vec
# Vector từ BẮT BUỘC lấy từ ma trận W (center word) của SkipGramModel
# =============================================================================

def get_vector(word, word_to_idx, model):
    """
    Lấy vector embedding của một từ từ ma trận W (center word matrix).

    Args:
        word        (str)          : Từ cần lấy vector.
        word_to_idx (dict)         : Ánh xạ word -> index.
        model       (SkipGramModel): Mô hình đã huấn luyện.

    Returns:
        np.ndarray shape (embed_size,) nếu từ tồn tại, None nếu không.
    """
    if word not in word_to_idx:
        print(f"  [!] Từ '{word}' không có trong từ điển.")
        return None
    idx = word_to_idx[word]
    return model.W[idx]  # Lấy từ W (center embeddings)


def cosine_similarity(vec1, vec2):
    """
    Tính cosine similarity giữa 2 vector.

    Args:
        vec1, vec2 (np.ndarray): Hai vector cần so sánh.

    Returns:
        float: Giá trị cosine similarity trong [-1, 1].
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def word_analogy(a, b, c, word_to_idx, idx_to_word, model):
    """
    Giải bài toán analogy: a - b + c ≈ d  (VD: king - man + woman ≈ queen)
    Yêu cầu C2.2 (PDF trang 7): v_a - v_b + v_c ≈ v_d

    Args:
        a, b, c     (str)          : Ba từ đầu vào.
        word_to_idx (dict)         : Ánh xạ word -> index.
        idx_to_word (dict)         : Ánh xạ index -> word.
        model       (SkipGramModel): Mô hình đã huấn luyện.

    Returns:
        list of (word, score): Top 5 từ gần nhất (loại trừ a, b, c).
    """
    for word in [a, b, c]:
        if word not in word_to_idx:
            print(f"  [!] Từ '{word}' không có trong từ điển.")
            return []

    v_a = model.W[word_to_idx[a]]
    v_b = model.W[word_to_idx[b]]
    v_c = model.W[word_to_idx[c]]

    # Vector mục tiêu: v_a - v_b + v_c
    target = v_a - v_b + v_c

    # Tính cosine similarity với toàn bộ từ điển (dùng W)
    # Chuẩn hóa để tính nhanh
    norms = np.linalg.norm(model.W, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    W_norm = model.W / norms

    target_norm = target / (np.linalg.norm(target) + 1e-10)
    scores = W_norm @ target_norm  # (vocab_size,)

    # Loại trừ a, b, c khỏi kết quả
    exclude_ids = {word_to_idx[a], word_to_idx[b], word_to_idx[c]}
    for idx in exclude_ids:
        scores[idx] = -np.inf

    # Lấy top 5
    top_ids = np.argsort(scores)[::-1][:5]
    return [(idx_to_word[i], float(scores[i])) for i in top_ids]


def plot_tsne(words, word_clusters, word_to_idx, model):
    """
    Dùng t-SNE giảm xuống 2D, vẽ scatter plot tô màu theo nhóm từ.
    Yêu cầu C2.3 (PDF trang 7): ít nhất 50 từ, tô màu theo nhóm ngữ nghĩa.

    Args:
        words         (list[str])        : Danh sách từ cần visualize.
        word_clusters (dict[str, str])   : Ánh xạ word -> tên nhóm (cluster).
        word_to_idx   (dict)             : Ánh xạ word -> index.
        model         (SkipGramModel)    : Mô hình đã huấn luyện.
    """
    # Lọc các từ có trong từ điển
    valid_words = [w for w in words if w in word_to_idx]
    if len(valid_words) < 2:
        print("  [!] Không đủ từ hợp lệ để vẽ t-SNE.")
        return

    # Lấy vectors
    vectors = np.array([model.W[word_to_idx[w]] for w in valid_words])

    # t-SNE giảm xuống 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_words)-1))
    vectors_2d = tsne.fit_transform(vectors)

    # Tô màu theo nhóm
    cluster_names = sorted(set(word_clusters.values()))
    colors = cm.get_cmap('tab10', len(cluster_names))
    cluster_to_color = {name: colors(i) for i, name in enumerate(cluster_names)}

    plt.figure(figsize=(14, 10))
    # Vẽ từng nhóm
    for cluster_name in cluster_names:
        idxs = [i for i, w in enumerate(valid_words)
                if word_clusters.get(w) == cluster_name]
        plt.scatter(
            vectors_2d[idxs, 0], vectors_2d[idxs, 1],
            c=[cluster_to_color[cluster_name]],
            label=cluster_name, s=80, alpha=0.8
        )
        # Gán nhãn từng từ
        for i in idxs:
            plt.annotate(
                valid_words[i],
                (vectors_2d[i, 0], vectors_2d[i, 1]),
                fontsize=8, alpha=0.85,
                xytext=(4, 4), textcoords='offset points'
            )

    plt.title('t-SNE Visualization of Word Embeddings (Word2Vec)', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Nhóm từ', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    print(f"  [✓] t-SNE vẽ {len(valid_words)} từ thuộc {len(cluster_names)} nhóm.")


# =============================================================================
# Khối __main__: Chạy độc lập để giáo viên kiểm tra
# (Khi import vào demo.ipynb sẽ không chạy đoạn này)
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  KIỂM TRA ĐỘC LẬP CÁC HÀM ĐÁNH GIÁ (evaluate.py)")
    print("=" * 55)

    # --- Tạo mock model để test độc lập (không cần load data thật) ---
    class MockModel:
        def __init__(self, vocab_size, embed_size):
            np.random.seed(42)
            self.W = np.random.randn(vocab_size, embed_size)

    mock_vocab = ["king", "queen", "man", "woman", "paris",
                  "france", "italy", "rome", "dog", "cat"]
    mock_word_to_idx = {w: i for i, w in enumerate(mock_vocab)}
    mock_idx_to_word = {i: w for i, w in enumerate(mock_vocab)}
    mock_model = MockModel(len(mock_vocab), 10)

    # --- Test 1: get_vector ---
    print("\n[Test 1] get_vector('king')")
    vec = get_vector("king", mock_word_to_idx, mock_model)
    print(f"  Vector 'king' (10 chiều): {vec}")

    vec_missing = get_vector("xyz_missing", mock_word_to_idx, mock_model)
    print(f"  Vector 'xyz_missing': {vec_missing}")

    # --- Test 2: cosine_similarity ---
    print("\n[Test 2] cosine_similarity")
    v1 = get_vector("king", mock_word_to_idx, mock_model)
    v2 = get_vector("queen", mock_word_to_idx, mock_model)
    sim = cosine_similarity(v1, v2)
    print(f"  cosine_similarity(king, queen) = {sim:.4f}")

    # --- Test 3: word_analogy ---
    print("\n[Test 3] word_analogy: king - man + woman = ?")
    results = word_analogy("king", "man", "woman",
                           mock_word_to_idx, mock_idx_to_word, mock_model)
    for word, score in results:
        print(f"  {word:<12} score={score:.4f}")

    # --- Test 4: cosine_similarity edge cases ---
    print("\n[Test 4] cosine_similarity với vector zero")
    zero_vec = np.zeros(10)
    sim_zero = cosine_similarity(zero_vec, v1)
    print(f"  cosine_similarity(zero, king) = {sim_zero:.4f}  (expected: 0.0)")

    print("\n" + "=" * 55)
    print("  Tất cả hàm chạy OK!")
    print("=" * 55)
