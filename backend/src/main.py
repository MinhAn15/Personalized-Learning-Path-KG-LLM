import os
import sys
import logging
import warnings
from dotenv import load_dotenv

# Load .env early and set GOOGLE_API_KEY from GEMINI_API_KEY if present to satisfy
# libraries that expect GOOGLE_API_KEY or ADC. Also suppress noisy dependency warnings.
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, message=".*validate_default.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Local imports
from .config import Config
from .data_loader import (
    execute_cypher_query, check_and_load_kg, verify_graph
)
from .path_generator import (
    generate_learning_path
)
from .content_generator import (
    generate_quiz, evaluate_quiz
)
from .api import app # Import FastAPI app

# External libs used for initialization
from neo4j import GraphDatabase
_LLAMA_INDEX_HAS_SETTINGS = False
Settings = None
Gemini = None
GeminiEmbedding = None
try:
    # Prefer our genai wrapper which uses google.generativeai directly
    from .genai_wrapper import GenAIWrapper
except Exception:
    GenAIWrapper = None

# ==============================================================================
# THIẾT LẬP BAN ĐẦU (SETUP)
# ==============================================================================

def setup_logging():
    """Cấu hình logging để ghi vào file và in ra console."""
    log_file = os.path.join(Config.LOG_DIR, 'learning_path_system.log')
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Xóa các handler cũ để tránh ghi log lặp lại
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("="*50)
    logging.info("Bắt đầu phiên làm việc mới. Logger đã được cấu hình.")

@app.on_event("startup")
def initialize_connections_and_settings():
    """Khởi tạo kết nối tới Neo4j và cấu hình LlamaIndex/Gemini."""
    logging.info("Đang khởi tạo các kết nối và thiết lập...")
    # 1. Kết nối tới Neo4j (ưu tiên để phát hiện lỗi DB sớm)
    from . import config as _config_module
    neo_cfg = _config_module.NEO4J_CONFIG
    if not all([neo_cfg.get('url'), neo_cfg.get('username'), neo_cfg.get('password')]):
        logging.error("Thông tin kết nối Neo4j chưa được cấu hình trong file .env")
        logging.warning("Server sẽ khởi động ở chế độ demo (Neo4j unavailable). Set NEO4J_URL/NEO4J_USER/NEO4J_PASSWORD to enable DB features.")
        return None

    try:
        driver = GraphDatabase.driver(
            neo_cfg.get('url'),
            auth=(neo_cfg.get('username'), neo_cfg.get('password'))
        )
        driver.verify_connectivity()
        Config.NEO4J_DRIVER = driver # Lưu driver vào config
        logging.info("Kết nối tới Neo4j AuraDB thành công.")
    except Exception as e:
        logging.error(f"Lỗi kết nối Neo4j: {e}")
        logging.warning("Tiếp tục khởi động server ở chế độ demo mà không có Neo4j.")
        return None

    # 2. Tải dữ liệu KG ban đầu nếu cần
    try:
        check_and_load_kg(driver)
    except Exception as e:
        logging.error(f"Lỗi nghiêm trọng trong quá trình kiểm tra và tải Knowledge Graph: {e}")
        # Không raise ở đây per default — chỉ log để server vẫn có thể khởi động trong môi trường dev

    # 3. Cấu hình Gemini và LlamaIndex (không block nếu model không tồn tại)
    try:
        if not getattr(_config_module, 'GEMINI_API_KEY', None):
            logging.warning("GEMINI_API_KEY không được cấu hình trong file .env; bỏ qua khởi tạo LLM.")
        elif Gemini is None or GeminiEmbedding is None:
            logging.warning("Gemini LLM classes unavailable; ensure llama-index-llms-gemini is installed to enable LLM features.")
        else:
            try:
                # Try a sensible default model name; if not found, log and continue
                # Prefer the GenAIWrapper using google.generativeai
                llm = None
                embed_model = None
                if GenAIWrapper is not None:
                    try:
                        preferred = "models/gemini-pro-latest"
                        llm = GenAIWrapper(preferred, api_key=_config_module.GEMINI_API_KEY)
                        # Use the wrapper's embed method as Config.EMBED_MODEL placeholder
                        embed_model = llm
                        logging.info(f"GenAIWrapper initialized with model {preferred}")
                    except Exception as e:
                        logging.warning(f"GenAIWrapper init failed: {e}")
                else:
                    logging.warning("GenAIWrapper not available; falling back to existing llama-index Gemini if installed.")

                # Fall back to llama-index gemini classes if available
                if llm is None and Gemini is not None:
                    try_models = ["models/gemini-pro", "gemini-pro", "text-bison@001"]
                    for m in try_models:
                        try:
                            llm = Gemini(model_name=m, api_key=_config_module.GEMINI_API_KEY)
                            embed_model = GeminiEmbedding(model_name="models/embedding-001")
                            logging.info(f"Gemini (llama-index) initialized with model {m}")
                            break
                        except Exception as e:
                            logging.warning(f"Gemini model {m} not available: {e}")

                if llm and embed_model:
                    Config.LLM = llm
                    Config.EMBED_MODEL = embed_model
                else:
                    logging.warning("Không thể khởi tạo LLM/embed model; LLM features sẽ tạm thời bị vô hiệu hoá.")
            except Exception as e:
                logging.warning(f"Lỗi khi khởi tạo Gemini (bắt lỗi): {e}")
    except Exception as e:
        logging.warning(f"Lỗi không mong đợi khi kiểm tra cấu hình Gemini: {e}")

    # Trả về driver để caller có thể sử dụng/kiểm tra
    return driver
        
# ==============================================================================
# HÀM MAIN VÀ ĐIỂM VÀO CHÍNH
# ==============================================================================

def main():
    """
    Hàm này chủ yếu để chạy server FastAPI bằng uvicorn.
    """
    import uvicorn
    
    setup_logging()
    logging.info("Bắt đầu khởi chạy FastAPI server...")
    
    # Lấy host và port từ config, có giá trị mặc định
    host = getattr(Config, "HOST", "127.0.0.1")
    port = getattr(Config, "PORT", 8000)

    # Run the FastAPI app by import path string to avoid importing api at module import time
    uvicorn.run("backend.src.api:app", host=host, port=port)

if __name__ == "__main__":
    main()