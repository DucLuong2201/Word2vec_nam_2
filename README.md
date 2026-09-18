# Word2Vec: Skip-gram & Negative Sampling

Đồ án cuối kỳ môn Xử lý Ngôn ngữ Tự nhiên. Cài đặt mô hình Word2Vec (Skip-gram) kết hợp kỹ thuật Negative Sampling hoàn toàn từ đầu bằng NumPy.

## 📁 Cấu trúc thư mục
* `data/`: Chứa corpus (`text8`)
* `src/`: Chứa mã nguồn (`preprocess.py`, `model.py`, `train.py`, `evaluate.py`).
* `notebooks/`: File `demo.ipynb` minh họa kết quả thực nghiệm.
* `tests/`: File `test_model.py` chạy Gradient Check và Unit test.

## 🚀 Hướng dẫn chạy (Quick Start)
Chạy tuần tự 5 lệnh sau trên Terminal để tái tạo toàn bộ kết quả:

**1. Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

**2. Kiểm tra đạo hàm & Unit test:**
```bash
pytest tests/ -v
```

**3. Tiền xử lý & Huấn luyện (Train):**
```bash
python src/train.py
```

**4. Đánh giá mô hình (Evaluate):**
```bash
python src/evaluate.py
```

**5. Mở Notebook xem trực tiếp báo cáo & biểu đồ:**
```bash
jupyter notebook notebooks/demo.ipynb
```

## 🛠 Yêu cầu hệ thống
- Python 3.8+
- Các thư viện: numpy, matplotlib, scikit-learn, tqdm, gensim, pytest, jupyter, ipykernel
- Tải data để có thể chạy code tại link: http://mattmahoney.net/dc/text8.zip (copy link và paste vào trang mới của ứng dụng tìm kiếm, nhóm cam kết link hoàn toàn an toàn)
