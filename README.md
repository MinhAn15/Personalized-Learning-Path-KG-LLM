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

* **Tầng Giao Diện Người Dùng (UI Layer):** Next.js (React) frontend located in the `frontend/` folder for browser-based interaction. There is also a minimal command-line interface for ad-hoc runs in `backend/src/main.py`.
* **Tầng Xử Lý Logic (Logic Layer):** "Bộ não" của hệ thống, được viết bằng **Python**, chứa các module xử lý yêu cầu, tạo lộ trình (thuật toán A\*), và sinh nội dung (gọi API LLM).
* **Tầng Dữ Liệu (Data Layer):** Bao gồm cơ sở dữ liệu đồ thị **Neo4j AuraDB** để lưu trữ đồ thị tri thức và **LlamaIndex** để tạo các chỉ mục (index) cho việc truy vấn ngữ nghĩa và đồ thị.


*Sơ đồ kiến trúc hệ thống 3 lớp*

---

## 🛠️ Công Nghệ Sử Dụng

* **Ngôn ngữ:** Python 3.x
* **Cơ sở dữ liệu Đồ thị:** Neo4j AuraDB (Cloud)
* **LLM & Indexing:** Google Generative AI (Gemini) is the primary LLM surface used by the backend (via a small compatibility wrapper). `llama-index` is referenced in some modules as an optional integration for indexing and adapters, but Gemini (google.generativeai) is the recommended runtime.
* **Thư viện Python chính:** `neo4j`, `llama-index`, `pandas`, `scikit-learn`, `mlxtend`, `python-dotenv`.

---

## 📂 Cấu Trúc Dự Án

```
/Personalized-Learning-Path-KG-LLM
|
├── 📂 backend/            # Python backend (FastAPI) and data processing
|   ├── 📂 data/
|   └── 📂 src/
|       ├── api.py
|       ├── main.py
|       └── ...
├── 📂 frontend/           # Next.js (React) frontend app
├── 📂 notebooks/
├── 📂 prompts/
├── 📄 .env
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

2.  **Tạo và kích hoạt môi trường ảo cho backend (khuyến khích):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  **Cài đặt các thư viện cần thiết cho backend:**
    ```bash
    # Trong thư mục gốc của repo
    & '(.venv)\Scripts\python.exe' -m pip install -r backend/requirements.txt
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

#### **4. Cài đặt và chạy frontend (Next.js)**

1. Chuyển vào thư mục frontend và cài dependencies:

```bash
cd frontend
npm install
```

2. Chạy frontend ở chế độ phát triển:

```bash
npm run dev
```

Frontend mặc định sẽ chạy trên `http://localhost:3000`. Đảm bảo backend API (`http://127.0.0.1:8000`) đang chạy hoặc cập nhật cấu hình API base URL trong frontend nếu cần.

#### **5. Tải Dữ Liệu Lên Neo4j**

* Hệ thống sử dụng các file `nodes.csv` và `relationships.csv` để xây dựng đồ thị. Các file này được tạo ra từ quy trình tiền xử lý (xem bên dưới).
* Bạn cần đặt các file CSV này vào thư mục `import` của cơ sở dữ liệu Neo4j của bạn, hoặc điều chỉnh hàm `check_and_load_kg` để tải từ một đường dẫn khác.

#### **6. Chạy Ứng Dụng (Main flow - backend)**

Ứng dụng chính backend được triển khai dưới dạng một FastAPI app trong `backend/src/api.py`.
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
    trong background (một thread) để tránh chặn quá trình khởi động của ASGI server. Code hiện đã
    triển khai một background init để giảm nguy cơ chặn startup.
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

### Neo4j schema setup (constraints & indexes)

Chạy script thiết lập schema để tạo các ràng buộc và chỉ mục quan trọng trong Neo4j:

```powershell
& '.venv\Scripts\python.exe' backend/scripts/neo4j_schema_setup.py
```

Script cố gắng tạo:
- UNIQUE constraint cho `KnowledgeNode.Node_ID` và `Student.StudentID`
- Indexes cho các thuộc tính truy vấn phổ biến: `Context`, `Skill_Level`, `Priority`, `Difficulty`, `Time_Estimate`, `Semantic_Tags`
- Indexes cho `LearningData (student_id, timestamp)` và `node_id`
- Index thuộc tính quan hệ `Weight`, `Dependency` (bỏ qua nếu Neo4j không hỗ trợ)

### New building blocks (optional to use now)

- `backend/src/learner_state.py`: mô hình hóa trạng thái người học (mastery, ZPD, lịch ôn)
    - `LearnerState.from_neo4j(driver, student_id)` để dựng trạng thái từ `LearningData`
    - `update_mastery`, `estimate_next_review`, `predict_mastery`
- `backend/src/adaptive_path_planner.py`: bộ lập kế hoạch đường đi, có thể tận dụng `a_star_custom` nếu đã có
    - `AdaptivePathPlanner.compute_dynamic_weights(learner)` và `plan_path(...)`
- `backend/src/hybrid_retriever.py`: Hybrid retriever (Graph RAG + tag-sim surrogate)
    - `retrieve(query, learner_id, context_type)` với router đơn giản giữa cấu trúc/semantic
- `backend/src/explainability.py`: giải thích đường đi và đề xuất thay thế
    - `explain_path(nodes, metrics, learner)` và `generate_counterfactuals(...)`

### Production-oriented helpers (optional)

- `backend/src/neo4j_manager.py`: Neo4j connection manager with pooling
    - `Neo4jManager().create_schema()` applies constraints/indexes (same as script)
    - `execute_read(query, params)`, `execute_write(query, params)` helpers
- `backend/src/learner_profile_manager.py`: Student profile management
    - `create_student(student_id, initial_profile)`
    - `update_profile_dimension(student_id, dimension, updates)`
    - `get_student_profile(student_id)` returns summary with derived metrics
- `backend/src/knowledge_tracing.py`: advanced knowledge tracing (decay + Bayesian-like update)
    - `compute_mastery_with_decay(student_id, node_id)`
    - `update_mastery_after_assessment(student_id, node_id, performance_score, assessment_method)`

## 🧰 Dev helper (start/stop logs)

There's a convenient PowerShell helper at `scripts/start_dev.ps1` to start/stop backend and frontend and capture logs. Example usage:

PowerShell:

```powershell
# Start both services (background)
.\scripts\start_dev.ps1

# Start backend in foreground (show logs directly)
.\scripts\start_dev.ps1 -Foreground

# Stop both services
.\scripts\start_dev.ps1 -Stop

# Tail logs
Get-Content -Path .\backend\logs\backend_stdout.log -Wait -Tail 200
Get-Content -Path .\frontend\logs\frontend_stdout.log -Wait -Tail 200
```
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

## � Tải dữ liệu & Upload vào Neo4j (chi tiết)

Dự án có sẵn các công cụ tiền xử lý và tải dữ liệu để xây dựng Knowledge Graph từ CSV. Dưới đây là các bước thường dùng:

- Các file CSV generated (ví dụ `master_nodes.csv` và `master_relationships.csv`) thường nằm trong `backend/data/github_import/`.
- Tạo/chuẩn hoá CSV từ thư mục input bằng script `prepare_data.py`:

```powershell
# Chạy prepare_data để tổng hợp CSV từ thư mục input
& '.venv\Scripts\python.exe' backend/src/prepare_data.py
```

- Tải CSV lên Neo4j bằng helper `check_and_load_kg` (hàm kiểm tra và tải dữ liệu):

```powershell
& '.venv\Scripts\python.exe' -c "from neo4j import GraphDatabase; from backend.src.config import NEO4J_CONFIG; from backend.src.data_loader import check_and_load_kg; drv=GraphDatabase.driver(NEO4J_CONFIG['url'], auth=(NEO4J_CONFIG['username'], NEO4J_CONFIG['password'])); print(check_and_load_kg(drv))"
```

- Hoặc import thủ công bằng Neo4j Browser / LOAD CSV: copy `master_nodes.csv` và `master_relationships.csv` vào thư mục import của Neo4j và chạy Cypher tương ứng.

> Lưu ý: đảm bảo biến môi trường `NEO4J_URL`, `NEO4J_USER`, `NEO4J_PASSWORD` trong `.env` là đúng.

## 🛠️ Troubleshooting (Vấn đề phổ biến và cách khắc phục)

1) Uvicorn startup hangs / asyncio.exceptions.CancelledError / lifespan errors

 - Triệu chứng: khi chạy `uvicorn backend.src.api:app` server in ra logs về `Waiting for application startup` rồi dừng với `CancelledError`.
 - Nguyên nhân phổ biến: khối lượng công việc trong sự kiện startup (lifespan) chặn tiến trình (ví dụ: chờ kết nối mạng tới Neo4j hoặc LLM) hoặc một exception xảy ra trong startup handler.
 - Cách kiểm tra & khắc phục nhanh:

```powershell
# Chạy uvicorn ở foreground để xem log chi tiết
& '.venv\Scripts\python.exe' -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000

# Nếu cần debug nhanh và bỏ qua sự kiện lifespan (không khuyến nghị cho production):
& '.venv\Scripts\python.exe' -m uvicorn backend.src.api:app --host 127.0.0.1 --port 8000 --lifespan off
```

 - Lưu ý: dự án hiện chạy khởi tạo Neo4j/LLM trong background thread để giảm chặn lúc startup, nhưng nếu cấu hình sai (missing env, network blocked), background init vẫn có thể fail — hãy xem logs.

2) GEMINI / LLM không hoạt động (không có API key hoặc lỗi model)

 - Triệu chứng: các endpoint LLM trả lỗi, hoặc `api/status` báo `gemini: false`.
 - Kiểm tra: đảm bảo bạn đã thêm `GEMINI_API_KEY` vào file `.env` (hoặc `GOOGLE_API_KEY` làm fallback).

```env
GEMINI_API_KEY="ya29.your_gemini_api_key_here"
```

 - Nếu backend sử dụng adapter `llama-index` ở một số phần, cài `llama-index` và adapter tương ứng.

3) Neo4j connectivity / authentication errors

 - Triệu chứng: `driver.verify_connectivity()` fails, or `check_and_load_kg` returns errors.
 - Kiểm tra:
   - Mở `NEO4J_URL` (ví dụ `neo4j+s://<id>.databases.neo4j.io`), user and password in `.env`.
   - Test via the Python snippet shown above or via Neo4j Browser.

4) Logs & nơi tìm log

 - Backend logs: `backend/src/main.py` configures logging to a file under `Config.LOG_DIR` (mặc định `backend/logs/learning_path_system.log` nếu cấu hình theo mặc định).
 - Tail logs in PowerShell:

```powershell
Get-Content -Path .\backend\logs\learning_path_system.log -Wait -Tail 200
```

## �📈 Hướng Phát Triển (gợi ý)

* **Nâng cao chất lượng KG:** Xây dựng cơ chế cho phép chuyên gia kiểm duyệt và tinh chỉnh đồ thị tri thức.
* **Phát triển giao diện người dùng:** Hoàn thiện và mở rộng frontend Next.js (React). Có thể giữ Streamlit cho các thử nghiệm nội bộ nếu cần, nhưng chính thức giao diện web sản phẩm là Next.js.
* **Tối ưu hóa LLM:** Thử nghiệm với các mô hình nhỏ hơn (distilled models), caching và batching để giảm chi phí và độ trễ.
* **Nghiên cứu dài hạn:** Thực hiện các thử nghiệm với người dùng thực tế để đánh giá tác động của hệ thống đến kết quả học tập.

