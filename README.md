# Hệ Thống Đề Xuất Lộ Trình Học Tập Cá Nhân Hóa Dựa Trên Đồ Thị Tri Thức và Mô Hình Ngôn Ngữ Lớn

Dự án nghiên cứu và xây dựng một hệ thống mẫu (prototype) nhằm đề xuất các lộ trình học tập được cá nhân hóa, dựa trên sự kết hợp giữa **Đồ Thị Tri Thức (Knowledge Graph - KG)** và **Mô Hình Ngôn Ngữ Lớn (Large Language Model - LLM)**. Hệ thống hướng đến việc cung cấp trải nghiệm học tập linh hoạt, thích ứng với trình độ, mục tiêu và phong cách học của từng cá nhân, giải quyết những hạn chế của các nền tảng học tập trực tuyến hiện tại.

---

## 🚀 Các Tính Năng Chính

* **Biểu Diễn Tri Thức Bằng Đồ Thị:** Sử dụng đồ thị tri thức (KG) được xây dựng trên **Neo4j** để mô hình hóa các khái niệm giáo dục và mối quan hệ phức tạp giữa chúng (như `REQUIRES`, `NEXT`, `IS_SUBCONCEPT_OF`).
* **Tương Tác Bằng Ngôn Ngữ Tự Nhiên:** Tận dụng sức mạnh của **Mô hình Ngôn ngữ Lớn (LLM)** và **LlamaIndex** để hiểu các yêu cầu học tập của người dùng (ví dụ: "Tôi muốn học SQL cơ bản") và chuyển chúng thành các truy vấn trên đồ thị.
* **Tạo Lộ Trình Tối Ưu:** Áp dụng thuật toán **A\*** tùy chỉnh để tìm kiếm và tạo ra các lộ trình học tập tối ưu, logic và phù hợp nhất với hồ sơ của từng học viên[cite: 538, 1457].
* **Sinh Nội Dung Cá Nhân Hóa:** Tự động tạo ra nội dung bài giảng và bài kiểm tra trắc nghiệm được cá nhân hóa theo phong cách học tập (VARK), trình độ và mục tiêu của người học.
* **Quy Trình Dữ Liệu Tự Động:** Giới thiệu quy trình tiền xử lý dữ liệu tự động hóa bằng cách sử dụng các bộ prompt `SPR Generator` và `SPR Validation` để trích xuất và chuẩn hóa kiến thức từ tài liệu gốc.

---

## 🏛️ Kiến Trúc Hệ Thống

[cite_start]Kiến trúc tổng thể của hệ thống được thiết kế theo mô hình 3 lớp và được minh họa chi tiết bằng **mô hình C4**, giúp làm rõ sự tương tác giữa các thành phần từ cấp độ tổng quan (Context) đến chi tiết (Component).

* **Tầng Giao Diện Người Dùng (UI Layer):** Giao diện dòng lệnh (và Streamlit trong tương lai) để người dùng tương tác.
* **Tầng Xử Lý Logic (Logic Layer):** "Bộ não" của hệ thống, được viết bằng **Python**, chứa các module xử lý yêu cầu, tạo lộ trình (thuật toán A\*), và sinh nội dung (gọi API LLM).
* **Tầng Dữ Liệu (Data Layer):** Bao gồm cơ sở dữ liệu đồ thị **Neo4j AuraDB** để lưu trữ đồ thị tri thức và **LlamaIndex** để tạo các chỉ mục (index) cho việc truy vấn ngữ nghĩa và đồ thị.


*Sơ đồ kiến trúc hệ thống 3 lớp*

---

## 🛠️ Công Nghệ Sử Dụng

* **Ngôn ngữ:** Python 3.x
* **Cơ sở dữ liệu Đồ thị:** Neo4j AuraDB (Cloud)
* **LLM & Indexing:** OpenAI (GPT-3.5-Turbo), LlamaIndex
* **Thư viện Python chính:** `neo4j`, `llama-index`, `pandas`, `scikit-learn`, `mlxtend`, `python-dotenv`.

---

## 📂 Cấu Trúc Dự Án

```
/Personalized-Learning-Path-KG-LLM
|
├── 📂 data/
|   ├── 📂 input/
|   └── 📂 output/
├── 📂 notebooks/
├── 📂 prompts/
├── 📂 src/
|   ├── - config.py
|   ├── - data_loader.py
|   ├── - content_generator.py
|   ├── - path_generator.py
|   ├── - recommendations.py
|   ├── - session_manager.py
|   └── - main.py
├── 📄 .env
├── 📄 .gitignore
├── 📄 README.md
└── 📄 requirements.txt
```

---

## ⚙️ Hướng Dẫn Cài Đặt và Chạy

#### **1. Chuẩn Bị Môi Trường**

* Đã cài đặt [Python 3.8+](https://www.python.org/downloads/).
* Đã cài đặt [Git](https://git-scm.com/downloads).
* Có tài khoản [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-database/) (có gói miễn phí).
* Có API Key từ [OpenAI](https://platform.openai.com/).

#### **2. Cài Đặt**

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/your-username/Personalized-Learning-Path-KG-LLM.git](https://github.com/your-username/Personalized-Learning-Path-KG-LLM.git)
    cd Personalized-Learning-Path-KG-LLM
    ```

2.  **Tạo và kích hoạt môi trường ảo (khuyến khích):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Thiết lập biến môi trường:**
    * Tạo một file tên là `.env` ở thư mục gốc của dự án.
    * Sao chép nội dung dưới đây vào file `.env` và thay thế bằng thông tin của bạn:
        ```env
        # Biến môi trường cho dự án
        # --- Gemini / Google Generative AI ---
        # The code in this project uses Google/Generative AI (Gemini) as the
        # primary LLM. Set your key here. In some places the runtime may also
        # read GOOGLE_API_KEY as a fallback.
        GEMINI_API_KEY="ya29.your_gemini_api_key_here"

        # --- Neo4j AuraDB ---
        NEO4J_URL="neo4j+s://xxxxxxxx.databases.neo4j.io"
        NEO4J_USER="neo4j"
        NEO4J_PASSWORD="your_strong_auradb_password"

        # --- GitHub (optional, for public/private repo fetches) ---
        GITHUB_TOKEN="ghp_xxx"
        ```
    * **Quan trọng:** Thêm file `.env` vào `.gitignore` để không đưa thông tin nhạy cảm lên GitHub.

#### **5. Tải Dữ Liệu Lên Neo4j**

* Hệ thống sử dụng các file `nodes.csv` và `relationships.csv` để xây dựng đồ thị. Các file này được tạo ra từ quy trình tiền xử lý (xem bên dưới).
* Bạn cần đặt các file CSV này vào thư mục `import` của cơ sở dữ liệu Neo4j của bạn, hoặc điều chỉnh hàm `check_and_load_kg` để tải từ một đường dẫn khác.

#### **6. Chạy Ứng Dụng (Main flow)**

Ứng dụng chính được triển khai dưới dạng một FastAPI app trong `backend/src/api.py`.
Để chạy API server (tức là main flow), dùng `uvicorn` và chạy bằng Python trong virtualenv
để đảm bảo các package được lấy từ môi trường ảo của dự án.

Ví dụ (PowerShell / Windows):

```powershell
& 'venv\Scripts\python.exe' -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000
```

Hoặc nếu bạn đang dùng virtualenv nằm trong `.venv` (the workspace default used here):

```powershell
& '.venv\Scripts\python.exe' -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000
```

Ghi chú vận hành:
- Khi server khởi động, việc khởi tạo các kết nối tới Neo4j và cấu hình LLM có thể chạy
    trong background (một thread) để tránh chặn quá trình khởi động của ASGI server.
- Nếu Neo4j hoặc Gemini chưa sẵn sàng, API vẫn cung cấp các endpoint demo (ví dụ
    `/api/generate_path_demo`) để frontend hoặc trình duyệt kiểm tra giao diện.
- Nếu bạn gặp lỗi liên quan tới `lifespan` hoặc thấy server dừng tự động khi khởi động,
    chạy `uvicorn` ở foreground (như lệnh trên) để xem log chi tiết và xác định nguyên nhân.

Sau khi server chạy, truy cập:

- Health / status: `http://127.0.0.1:8000/api/status`
- Demo path: gửi POST tới `http://127.0.0.1:8000/api/generate_path_demo` với body JSON:

```json
{
    "student_id": "demo",
    "level": "beginner",
    "context": "test",
    "student_goal": "learn SQL",
    "use_llm": false
}
```

---

## 🔄 Quy Trình Xử Lý Dữ Liệu

Hệ thống hoạt động qua 2 giai đoạn chính:

1.  **Giai đoạn 1: Tiền Xử Lý & Xây Dựng Knowledge Graph**
    * **Trích xuất Dữ liệu:** Sử dụng prompt `SPR Generator`để LLM đọc một tài liệu học thuật (PDF, DOCX) và tự động trích xuất các khái niệm, thuộc tính, mối quan hệ thành 2 file `nodes.csv` và `relationships.csv`.
    * **Xác thực Dữ liệu:** Sử dụng prompt `SPR Validation`để một phiên LLM khác kiểm tra chéo, sửa lỗi, và chuẩn hóa dữ liệu đã trích xuất, đảm bảo chất lượng và tính nhất quán cho đồ thị tri thức.
    * **Tải vào Neo4j:** Dữ liệu đã được xác thực sẽ được tải vào Neo4j để hình thành đồ thị tri thức hoàn chỉnh.

2.  **Giai đoạn 2: Vận Hành Hệ Thống Đề Xuất **
    * Hệ thống tương tác với người dùng để lấy thông tin đầu vào.
    * `path_generator.py` sẽ phân tích yêu cầu, tìm điểm đầu/cuối và chạy thuật toán A* để tạo lộ trình.
    * `content_generator.py` sẽ tạo nội dung học tập và bài kiểm tra cho từng bước.
    * Hồ sơ người dùng được cập nhật liên tục để các đề xuất trong tương lai ngày càng chính xác hơn.

---

## 📈 Hướng Phát Triển

* **Nâng cao chất lượng KG:** Xây dựng cơ chế cho phép chuyên gia kiểm duyệt và tinh chỉnh đồ thị tri thức.
* **Xây dựng Giao diện người dùng:** Phát triển giao diện web thân thiện bằng **Streamlit** để nâng cao trải nghiệm người dùng.
* **Tối ưu hóa LLM:** Thử nghiệm với các mô hình nhỏ hơn (distilled models) hoặc các kỹ thuật caching để giảm chi phí và độ trễ.
* **Nghiên cứu dài hạn:** Thực hiện các thử nghiệm với người dùng thực tế để đánh giá tác động của hệ thống đến kết quả học tập.
