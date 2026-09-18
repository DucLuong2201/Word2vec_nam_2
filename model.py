import numpy as np


def sigmoid(x):
    """
    Tính giá trị hàm sigmoid, có clipping để tránh overflow.

    Args:
        x (float hoặc np.ndarray): Giá trị đầu vào.

    Returns:
        float hoặc np.ndarray: Giá trị sigmoid trong khoảng (0, 1).
    """
    x = np.clip(x, -500, 500)  # Giới hạn để tránh overflow khi tính exp
    return 1.0 / (1.0 + np.exp(-x))


class SkipGramModel:
    """
    Mô hình Skip-Gram với Negative Sampling cho Word2Vec.

    Duy trì hai ma trận embedding:
        - W       : Ma trận embedding của center word,  shape (vocab_size, embed_size).
        - W_prime : Ma trận embedding của context word, shape (vocab_size, embed_size).

    Attributes:
        vocab_size (int): Kích thước từ điển.
        embed_size (int): Số chiều của vector embedding.
        W          (np.ndarray): Center word embeddings, khởi tạo Uniform(-0.5/d, 0.5/d).
        W_prime    (np.ndarray): Context word embeddings, khởi tạo bằng 0.
    """

    def __init__(self, vocab_size, embed_size):
        """
        Khởi tạo SkipGramModel với hai ma trận embedding W và W_prime.

        Args:
            vocab_size (int): Kích thước từ điển.
            embed_size (int): Số chiều vector embedding.
        """
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Khởi tạo W (center) theo Uniform(-0.5/d, 0.5/d) để giá trị nhỏ, đối xứng quanh 0
        limit = 0.5 / embed_size
        self.W = np.random.uniform(-limit, limit, (vocab_size, embed_size))

        # Khởi tạo W_prime (context) bằng 0 theo quy ước chuẩn của Word2Vec
        # Mỗi hàng i là vector context của từ thứ i => truy cập W_prime[i] trực tiếp
        self.W_prime = np.zeros((vocab_size, embed_size))

    def forward(self, center_idx):
        """
        Tính phân phối xác suất P(o | c) cho toàn bộ từ điển bằng softmax đầy đủ.

        Args:
            center_idx (int): Chỉ số của center word trong từ điển.

        Returns:
            np.ndarray shape (vocab_size,): Xác suất softmax P(o | c) cho mỗi từ.
        """
        v_c = self.W[center_idx]        # Vector center word, shape (embed_size,)
        scores = self.W_prime @ v_c     # Điểm số với toàn bộ context, shape (vocab_size,)

        # Trừ max trước khi exp để ổn định số học (numerically stable softmax)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        softmax_probs = exp_scores / exp_scores.sum()
        return softmax_probs

    def negative_sampling_loss(self, center_idx, context_idx, neg_samples):
        """
        Tính loss J_NEG và gradient theo công thức Negative Sampling.

        Công thức loss:
            J = -log σ(u_o · v_c) - Σ_k log σ(-u_k · v_c)

        Args:
            center_idx  (int)         : Chỉ số center word.
            context_idx (int)         : Chỉ số context word (positive sample).
            neg_samples (np.ndarray)  : Mảng shape (K,) chứa chỉ số các negative samples.

        Returns:
            loss     (float)      : Giá trị hàm loss J_NEG.
            grad_v_c (np.ndarray) : Gradient theo vector center word v_c, shape (embed_size,).
            grad_u_o (np.ndarray) : Gradient theo vector context word u_o, shape (embed_size,).
            grad_u_w (np.ndarray) : Gradient theo các negative word vectors, shape (K, embed_size).
        """
        v_c = self.W[center_idx]            # Vector center word,   shape (embed_size,)
        u_o = self.W_prime[context_idx]     # Vector context word,  shape (embed_size,)
        u_w = self.W_prime[neg_samples]     # Vector negative words, shape (K, embed_size)

        # Tính điểm số dot product cho positive và negative samples
        score_pos = np.dot(v_c, u_o)        # Scalar: độ tương đồng center - context
        score_neg = u_w @ v_c               # Shape (K,): độ tương đồng center - negative

        # Áp dụng sigmoid để tính xác suất
        pos_prob = sigmoid(score_pos)       # Xác suất positive pair, scalar
        neg_probs = sigmoid(-score_neg)     # Xác suất negative pairs, shape (K,)

        # Tính loss: -log P(positive) - Σ log P(negative)
        loss = -np.log(pos_prob + 1e-10) - np.sum(np.log(neg_probs + 1e-10))

        # --- Tính Gradients ---
        grad_score_pos = pos_prob - 1                               # Scalar
        grad_score_neg = 1 - neg_probs                              # Shape (K,)

        # Gradient theo v_c: tổng đóng góp từ positive và tất cả negative
        grad_v_c = grad_score_pos * u_o + u_w.T @ grad_score_neg   # Shape (embed_size,)

        # Gradient theo u_o (context word)
        grad_u_o = grad_score_pos * v_c                             # Shape (embed_size,)

        # Gradient theo u_w: mỗi negative word k có gradient riêng = scalar_k * v_c
        grad_u_w = grad_score_neg[:, None] * v_c[None, :]          # Shape (K, embed_size)

        return loss, grad_v_c, grad_u_o, grad_u_w


class NegativeSampler:
    """
    Lấy mẫu negative theo phân phối Unigram lũy thừa 3/4 bằng lookup table.

    Phân phối lấy mẫu:
        P(w) ∝ count(w)^0.75

    Lookup table kích thước lớn (mặc định 1e6) giúp lấy mẫu O(1).

    Attributes:
        table (np.ndarray): Lookup table chứa chỉ số từ, shape (table_size,).
    """

    def __init__(self, word_counts, vocab_size, power=0.75, table_size=int(1e6)):
        """
        Xây dựng lookup table theo phân phối Unigram^0.75.

        Args:
            word_counts (dict): Ánh xạ word_index -> số lần xuất hiện.
            vocab_size  (int) : Kích thước từ điển.
            power       (float): Số mũ của phân phối (mặc định 0.75).
            table_size  (int) : Kích thước lookup table (mặc định 1_000_000).
        """
        self.table = np.zeros(table_size, dtype=np.int32)

        # Lấy tần suất xuất hiện của từng từ theo thứ tự index
        freqs = np.array([word_counts.get(i, 0) for i in range(vocab_size)],
                         dtype=np.float64)

        # Tính xác suất lấy mẫu theo công thức P(w) ∝ count(w)^power
        probs = freqs ** power
        if probs.sum() == 0:
            # Trường hợp degenerate: chia đều cho tất cả từ
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()     # Chuẩn hóa thành phân phối xác suất

        # Điền lookup table: từ có xác suất cao sẽ chiếm nhiều ô hơn => O(1) khi sample
        idx, fill = 0, 0.0
        for word_idx, p in enumerate(probs):
            fill += p * table_size          # Số ô mà từ này được phân bổ
            while idx < fill and idx < table_size:
                self.table[idx] = word_idx
                idx += 1

    def sample(self, n, exclude=None):
        """
        Lấy n negative samples từ lookup table.

        Args:
            n       (int)      : Số lượng negative samples cần lấy.
            exclude (int|None) : Chỉ số từ cần loại trừ (thường là context word đúng).
                                 Nếu None, lấy mẫu nhanh không kiểm tra.

        Returns:
            np.ndarray shape (n,): Mảng chỉ số các negative samples.
        """
        if exclude is None:
            # Không cần loại trừ => vectorized, lấy n mẫu một lần cho nhanh
            indices = np.random.randint(0, len(self.table), size=n)
            return self.table[indices]

        # Cần loại trừ context word đúng => lấy từng mẫu và kiểm tra
        samples = []
        while len(samples) < n:
            idx = self.table[np.random.randint(0, len(self.table))]
            if idx != exclude:
                samples.append(idx)
        return np.array(samples, dtype=np.int32)