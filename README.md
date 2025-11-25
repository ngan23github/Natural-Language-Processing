# Neural Machine Translation (English → French)
## Model: Encoder–Decoder LSTM (Fixed Context Vector)

#### 🧠 Mục tiêu
Xây dựng mô hình Seq2Seq cơ bản bằng LSTM *không dùng thư viện seq2seq có sẵn*, thực hiện dịch máy Anh–Pháp.

### 🛠️ Hướng dẫn cài đặt

```bash
#!/bin/bash

# 1️⃣ Tạo môi trường mới và kích hoạt
conda create -n nlp_env python=3.10 -y

### Kích hoạt môi trường
conda activate nlp_env

# 2️⃣ Cài PyTorch + Torchtext (CPU)
pip install torch==1.13.1 torchtext==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# Hoặc nếu dùng GPU với CUDA 11.7:
pip install torch==1.13.1+cu117 torchtext==0.14.1 --index-url https://download.pytorch.org/whl/cu117

# 3️⃣ Cài SpaCy 3.7.2
pip install spacy==3.7.2

# 4️⃣ Cài model SpaCy tiếng Anh
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# 5️⃣ Cài model SpaCy tiếng Pháp
pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl

# 6️⃣ Cài Jupyter Notebook
conda install jupyter -y
```

### 🧩 Kiến trúc
- **Encoder:** 2-layer LSTM (embedding 256, hidden 512)
- **Decoder:** 2-layer LSTM + Linear projection
- **Loss:** CrossEntropyLoss (ignore PAD)
- **Optimizer:** Adam (lr=1e-3)
- **Teacher forcing:** 0.5
- **BLEU:** khoảng 25–35 (Multi30k subset)

### 📈 Quy trình huấn luyện
1. Tiền xử lý dữ liệu (token hóa, từ điển)
2. Huấn luyện với teacher forcing
3. Lưu checkpoint tốt nhất (`best_model.pth`)
4. Đánh giá BLEU score
5. Dịch thử câu tiếng Anh → tiếng Pháp

### 🧮 Kết quả mẫu
| Epoch | Train Loss | Val Loss | BLEU |
|:------|:-----------:|:--------:|:----:|
| 1 | 3.85 | 3.62 | 18.4 |
| 2 | 3.12 | 2.98 | 23.5 |
| 3 | 2.70 | 2.60 | 27.1 |

### 📚 Tài liệu tham khảo
- Sutskever et al., *Sequence to Sequence Learning with Neural Networks*, 2014.
- PyTorch Tutorials: NLP Sequence Models
- François Chollet, *Deep Learning with Python* (Chap 6.2)
