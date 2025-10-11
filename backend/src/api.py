from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Import các hàm logic chính từ các module của bạn
from .session_manager import run_learning_session
from .main import initialize_connections_and_settings # Tận dụng lại hàm khởi tạo

# --- Khởi tạo ứng dụng và các kết nối ---
app = FastAPI()
driver = None
logger = logging.getLogger(__name__)

# Cấu hình CORS để cho phép frontend (chạy ở port 3000) gọi tới
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Khởi tạo kết nối Neo4j khi server khởi động."""
    global driver
    try:
        driver = initialize_connections_and_settings()
        logger.info("Khởi tạo API thành công.")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi khởi tạo API: {e}")
        driver = None

@app.on_event("shutdown")
def shutdown_event():
    """Đóng kết nối khi server tắt."""
    if driver:
        driver.close()
        logger.info("Đã đóng kết nối Neo4j.")

# --- Định nghĩa model cho dữ liệu nhận vào ---
class LearningRequest(BaseModel):
    student_id: str
    level: str
    context: str
    student_goal: str
    use_llm: bool = False

# --- Định nghĩa các API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Personalized Learning Path API!"}

@app.post("/api/generate_path")
def generate_path_endpoint(request: LearningRequest):
    """
    Endpoint chính để nhận yêu cầu và tạo lộ trình học tập.
    """
    if not driver:
        return {"status": "error", "message": "Kết nối Database chưa sẵn sàng."}

    logger.info(f"Nhận yêu cầu tạo lộ trình cho: {request.student_id}")

    # Gọi hàm logic cốt lõi của bạn
    result = run_learning_session(
        student_id=request.student_id,
        level=request.level,
        context=request.context,
        student_goal=request.student_goal,
        driver=driver,
        use_llm=request.use_llm
    )

    return result