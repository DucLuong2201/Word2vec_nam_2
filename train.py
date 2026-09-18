import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocess import preprocess, generate_training_pairs
from model import SkipGramModel, NegativeSampler


def train(model, pairs, sampler, epochs, lr, k_neg, batch_size=512):
    """
    Huấn luyện mô hình Skip-gram bằng SGD với Negative Sampling.

    Mỗi bước cập nhật:
        1. Lấy k_neg negative samples (loại trừ context word đúng).
        2. Tính loss J_NEG và gradient qua negative_sampling_loss.
        3. Cập nhật W (center) và W_prime (context) theo SGD.
        4. Giảm learning rate tuyến tính theo số bước đã chạy.

    Args:
        model      (SkipGramModel)  : Mô hình Skip-gram từ B2.
        pairs      (np.ndarray)     : Mảng các cặp (center_idx, context_idx).
        sampler    (NegativeSampler): Bộ lấy mẫu âm từ B2.3.
        epochs     (int)            : Số epoch huấn luyện.
        lr         (float)          : Learning rate ban đầu.
        k_neg      (int)            : Số negative samples mỗi bước.
        batch_size (int)            : Kích thước batch (mặc định 512).

    Returns:
        losses_history (list[float]): Lịch sử loss trung bình mỗi 10,000 bước.
    """
    losses_history = []

    # Lưu lr ban đầu để tính giảm tuyến tính đúng cách
    lr_init = lr
    total_steps = len(pairs) * epochs
    current_step = 0

    for epoch in range(epochs):
        np.random.shuffle(pairs)    # Trộn dữ liệu sau mỗi epoch để tránh học theo thứ tự cố định
        total_loss = 0
        loss_tracker = 0            # Tích lũy loss trong 10,000 bước để tính trung bình

        print(f"Epoch {epoch+1}/{epochs}")
        for step in tqdm(range(len(pairs))):
            center_idx, context_idx = pairs[step]

            # Lấy mẫu âm, loại trừ context word đúng để tránh nhiễu gradient
            neg_samples = sampler.sample(k_neg, exclude=context_idx)

            # Tính loss và gradient qua Negative Sampling
            loss, g_vc, g_uo, g_uw = model.negative_sampling_loss(
                center_idx, context_idx, neg_samples
            )

            total_loss += loss
            loss_tracker += loss

            # Cập nhật trọng số theo SGD
            # W_prime shape (vocab, embed) => update theo HÀNG tương ứng
            model.W[center_idx]        -= lr * g_vc   # Cập nhật center word vector
            model.W_prime[context_idx] -= lr * g_uo   # Cập nhật context word vector

            # Dùng np.subtract.at để cập nhật đúng khi neg_samples có chỉ số trùng nhau
            np.subtract.at(model.W_prime, neg_samples, lr * g_uw)

            # Giảm lr tuyến tính: lr = lr_init * (1 - step / total_steps)
            current_step += 1
            lr = lr_init * (1 - current_step / total_steps)
            lr = max(lr, lr_init * 0.0001)  # Giữ lr tối thiểu để tránh bước cập nhật quá nhỏ

            # In trung bình loss mỗi 10,000 bước để theo dõi quá trình hội tụ
            if (step + 1) % 10000 == 0:
                avg_loss = loss_tracker / 10000
                losses_history.append(avg_loss)
                print(f"  Step {step+1} | Avg Loss: {avg_loss:.4f} | LR: {lr:.6f}")
                loss_tracker = 0    # Reset bộ đếm sau mỗi 10,000 bước

        # Lưu checkpoint sau mỗi epoch để có thể resume nếu bị gián đoạn
        with open(f"word2vec_epoch_{epoch+1}.pkl", "wb") as f:
            pickle.dump(model, f)

    return losses_history


def subsample_corpus(token_indices, word_counts, t=1e-5):
    """
    Lọc bớt các từ phổ biến theo xác suất để giảm nhiễu và tăng cường học từ hiếm.

    Công thức xác suất loại bỏ từ w:
        P(discard | w) = 1 - sqrt(t / f(w))
    Trong đó f(w) = tần suất tương đối của w trong corpus.

    Args:
        token_indices (list[int]): Danh sách index token sau preprocess.
        word_counts   (dict)     : Tần suất từ trong vocab, dạng {index: count}.
        t             (float)    : Ngưỡng subsampling (mặc định 1e-5).
                                   Từ có f(w) >> t sẽ bị loại bỏ với xác suất cao.

    Returns:
        subsampled_tokens (list[int]): Danh sách token sau khi lọc từ phổ biến.
    """
    total_words = sum(word_counts.values())     # Tổng số từ để tính tần suất tương đối
    subsampled_tokens = []

    for word_idx in token_indices:
        # Dùng .get() để tránh KeyError nếu word_idx không có trong word_counts
        f_w = word_counts.get(word_idx, 0) / total_words

        # Bỏ qua từ không xuất hiện trong word_counts (tần suất = 0)
        if f_w == 0:
            continue

        # Tính xác suất loại bỏ: từ càng phổ biến (f_w lớn) thì p_discard càng cao
        p_discard = 1 - np.sqrt(t / f_w)

        if np.random.rand() > p_discard:    # Giữ lại từ nếu không bị discard
            subsampled_tokens.append(word_idx)

    return subsampled_tokens


def plot_loss(losses):
    """
    Vẽ đồ thị loss theo số bước huấn luyện.

    Args:
        losses (list[float]): Danh sách loss trung bình mỗi 10,000 bước,
                              lấy từ giá trị trả về của hàm train().
    """
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title('Training Loss over steps (10k step interval)')
    plt.xlabel('Intervals (x10,000 steps)')
    plt.ylabel('Negative Sampling Loss')

    # Thêm grid để dễ quan sát xu hướng hội tụ
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load và xử lý dữ liệu
    print("Đang load dữ liệu...")
    with open("data/text8", encoding="utf-8") as f:
        raw_text = f.read()

    token_indices, word_to_idx, idx_to_word, freq_dict, _ = preprocess(raw_text, min_count=5)

    print("Đang subsampling...")
    token_indices = subsample_corpus(token_indices, freq_dict)

    # Subsampling và tạo pairs
    print("Đang tạo training pairs...")
    pairs = generate_training_pairs(token_indices, window_size=2)
    pairs = np.array(pairs)

    # Khởi tạo model
    vocab_size = len(word_to_idx)
    embed_size = 100
    model = SkipGramModel(vocab_size, embed_size)
    sampler = NegativeSampler(freq_dict, vocab_size, power=0.75)
    print("Model ready!")

    # Huấn luyện model
    losses = train(
        model=model,
        pairs=pairs,
        sampler=sampler,
        epochs=5,
        lr=0.025,
        k_neg=5
    )

    np.save("losses_history.npy", np.array(losses))
    print("✅ Đã lưu mảng loss vào losses_history.npy")

    # Vẽ đồ thị loss sau khi huấn luyện xong
    plot_loss(losses)