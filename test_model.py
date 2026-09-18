import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import SkipGramModel, NegativeSampler

# =============================================================================
# Yêu cầu B4.1: Thực hiện gradient check (numerical vs analytical)
# =============================================================================

def test_gradient_check_negative_sampling():
    """
    Kiểm tra đạo hàm (Gradient Check) cho hàm loss Negative Sampling.

    Quy trình kiểm tra:
        1. Tính đạo hàm giải tích (Analytical) từ công thức toán học.
        2. Tính đạo hàm số học (Numerical) bằng xấp xỉ vi phân trung tâm.
        3. Tính sai số tương đối (Relative Error) giữa 2 ma trận.
        4. Đảm bảo sai số < 1e-5 (Yêu cầu B4.1).
    """
    vocab_size, embed_size = 10, 5
    model = SkipGramModel(vocab_size, embed_size)
    center_idx, context_idx = 1, 2
    neg_samples = np.array([3, 4])
    
    eps = 1e-4      # Bước nhảy vi phân cực nhỏ
    tol = 1e-5      # Yêu cầu sai số < 10^-5
    
    # 1. Tính analytical gradients (gradient bằng công thức)
    _, grad_vc_a, grad_uo_a, grad_uw_a = model.negative_sampling_loss(
        center_idx, context_idx, neg_samples
    )
    
    # 2. Tính numerical gradients cho v_c (gradient xấp xỉ số học)
    grad_vc_n = np.zeros_like(model.W[center_idx])
    for i in range(embed_size):
        model.W[center_idx, i] += eps       # f(x + eps)
        loss_plus, _, _, _ = model.negative_sampling_loss(
            center_idx, context_idx, neg_samples
        )
        
        model.W[center_idx, i] -= 2 * eps   # f(x - eps)
        loss_minus, _, _, _ = model.negative_sampling_loss(
            center_idx, context_idx, neg_samples
        )
        
        # Công thức vi phân trung tâm
        grad_vc_n[i] = (loss_plus - loss_minus) / (2 * eps)
        model.W[center_idx, i] += eps       # Trả lại giá trị gốc
        
    # Tính sai số tương đối (thêm 1e-8 để tránh chia cho 0)
    rel_error = np.linalg.norm(grad_vc_a - grad_vc_n) / (
        np.linalg.norm(grad_vc_a) + np.linalg.norm(grad_vc_n) + 1e-8
    )
    print(f"\n[Gradient Check - Negative Sampling] Relative Error: {rel_error:.5e}")
    assert rel_error < tol, f"Gradient check thất bại với error: {rel_error}"

def test_gradient_check_softmax():
    """
    Kiểm tra đạo hàm (Gradient Check) cho Skip-gram Softmax loss.

    Quy trình kiểm tra:
        1. Tính đạo hàm giải tích dJ/dv_c = -u_o + sum(P(w|c) * u_w).
        2. Tính đạo hàm số học từ hàm J = -log P(o|c).
        3. So sánh và đảm bảo sai số tương đối < 1e-5.
    """
    vocab_size, embed_size = 10, 5
    model = SkipGramModel(vocab_size, embed_size)
    center_idx = 1
    context_idx = 2

    eps = 1e-4
    tol = 1e-5
    
    # 1. Tính analytical gradient cho v_c
    probs = model.forward(center_idx)       # Xác suất softmax: P(w|c)
    u_o = model.W_prime[context_idx]        # Vector ngữ cảnh đúng
    
    # sum(P(w|c) * u_w) tính nhanh bằng phép nhân ma trận (dot product)
    expected_context = np.dot(probs, model.W_prime) 
    grad_vc_a = -u_o + expected_context
    
    # 2. Tính numerical gradient cho v_c
    grad_vc_n = np.zeros_like(model.W[center_idx])
    for i in range(embed_size):
        model.W[center_idx, i] += eps       # f(x + eps)
        probs_plus = model.forward(center_idx)
        loss_plus = -np.log(probs_plus[context_idx] + 1e-10) 
        
        model.W[center_idx, i] -= 2 * eps   # f(x - eps)
        probs_minus = model.forward(center_idx)
        loss_minus = -np.log(probs_minus[context_idx] + 1e-10)
        
        grad_vc_n[i] = (loss_plus - loss_minus) / (2 * eps)
        model.W[center_idx, i] += eps       # Trả lại giá trị gốc
        
    # Tính sai số tương đối
    rel_error = np.linalg.norm(grad_vc_a - grad_vc_n) / (
        np.linalg.norm(grad_vc_a) + np.linalg.norm(grad_vc_n) + 1e-8
    )
    print(f"\n[Gradient Check - Softmax] Relative Error: {rel_error:.5e}")
    assert rel_error < tol, f"Softmax gradient check thất bại với error: {rel_error}"

# =============================================================================
# Yêu cầu B4.2: 5 Unit tests
# =============================================================================

def test_tensor_shapes():
    """
    1. Kiểm tra kích thước tensor đầu ra của mô hình.
    
    Đảm bảo:
        - W có shape (vocab_size, embed_size).
        - W_prime cũng có shape (vocab_size, embed_size) để tối ưu truy xuất.
    """
    model = SkipGramModel(10, 3)
    assert model.W.shape == (10, 3)
    assert model.W_prime.shape == (10, 3)   # W_prime đã đổi shape sang (vocab, embed)

def test_softmax_sum():
    """
    2. Kiểm tra tính hợp lệ của hàm Softmax.
    Đảm bảo tổng phân phối xác suất P(w|c) luôn xấp xỉ bằng 1.0.
    """
    model = SkipGramModel(10, 3)
    probs = model.forward(0)
    assert np.isclose(np.sum(probs), 1.0)

# 3. Gradient check pass
# (Đã được thực hiện chi tiết thông qua 2 hàm test_gradient_check phía trên)

def test_negative_sampler_no_context():
    """
    4. Kiểm tra tính loại trừ của bộ lấy mẫu âm (Negative Sampler).
    Đảm bảo samples trả về KHÔNG bao gồm context word đúng để tránh nhiễu gradient.
    """
    word_counts = {0: 100, 1: 10, 2: 5}
    sampler = NegativeSampler(word_counts, 3, table_size=100)
    
    # Lấy 10 mẫu, yêu cầu loại bỏ từ đích (exclude=1)
    samples = sampler.sample(10, exclude=1)
    assert len(samples) == 10
    assert 1 not in samples, "Negative sampler vẫn trả về context word đúng!"

def test_loss_decreases_toy_dataset():
    """
    5. Kiểm tra tính hội tụ (Convergence) của mô hình.
    
    Quy trình:
        1. Khởi tạo mô hình và toy dataset.
        2. Huấn luyện 1 epoch bằng SGD.
        3. Đảm bảo tổng loss lúc sau < tổng loss ban đầu.
    """
    model = SkipGramModel(5, 2)
    word_counts = {0: 10, 1: 5, 2: 2, 3: 2, 4: 1}
    sampler = NegativeSampler(word_counts, 5)
    
    pairs = [(0, 1), (0, 2), (1, 0), (2, 0)]    # Toy pairs
    
    # Tính tổng loss ban đầu
    initial_loss = sum([
        model.negative_sampling_loss(c, o, sampler.sample(2))[0]
        for c, o in pairs
    ])
    
    # Huấn luyện 1 epoch
    for c, o in pairs:
        neg = sampler.sample(2, exclude=o)
        loss, g_vc, g_uo, g_uw = model.negative_sampling_loss(c, o, neg)
        
        # Cập nhật theo HÀNG vì W_prime shape là (vocab_size, embed_size)
        model.W[c]         -= 0.1 * g_vc
        model.W_prime[o]   -= 0.1 * g_uo
        model.W_prime[neg] -= 0.1 * g_uw
        
    # Tính tổng loss lúc sau
    final_loss = sum([
        model.negative_sampling_loss(c, o, sampler.sample(2))[0]
        for c, o in pairs
    ])
    
    assert final_loss < initial_loss, "Mô hình không hội tụ, loss không giảm"