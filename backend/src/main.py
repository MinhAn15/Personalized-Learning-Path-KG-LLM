"""
Đây là file thực thi chính của dự án.
Nó khởi tạo các kết nối, thu thập thông tin đầu vào từ người dùng,
và điều phối toàn bộ quy trình đề xuất lộ trình học tập cá nhân hóa.
"""
import os
import logging
import time
import sys

# Import các thư viện và module cần thiết
from neo4j import GraphDatabase
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Import các module tự định nghĩa
# Cấu trúc `.` có nghĩa là import từ cùng một package (thư mục `src`)
try:
    from .config import Config, NEO4J_CONFIG, GEMINI_API_KEY
    from .data_loader import (
        execute_cypher_query, check_and_load_students, verify_graph, 
        load_student_profile,
    )
    from .path_generator import (
        generate_learning_path, update_learning_path, suggest_prerequisites
    )
    from .content_generator import (
        generate_learning_content, generate_quiz, evaluate_quiz
    )
except ImportError:
    # Xử lý trường hợp chạy file trực tiếp (python src/main.py)
    from config import Config, NEO4J_CONFIG, GEMINI_API_KEY
    from data_loader import (
        execute_cypher_query, check_and_load_students, verify_graph, 
        load_student_profile,
    )
    from path_generator import (
        generate_learning_path, update_learning_path, suggest_prerequisites
    )
    from content_generator import (
        generate_learning_content, generate_quiz, evaluate_quiz
    )

# ==============================================================================
# THIẾT LẬP BAN ĐẦU (SETUP)
# ==============================================================================

def setup_logging():
    """Cấu hình logging để ghi vào file."""
    log_file = os.path.join(Config.LOG_DIR, 'learning_path_system.log')
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # In log ra cả console
        ]
    )
    logging.info("Logger đã được cấu hình.")

def initialize_connections_and_settings():
    """Khởi tạo kết nối tới Neo4j và cấu hình LlamaIndex/Gemini."""
    # 1. Kết nối tới Neo4j
    if not all(NEO4J_CONFIG.values()):
        logging.error("Thông tin kết nối Neo4j chưa được cấu hình trong file .env")
        raise ValueError("Thông tin kết nối Neo4j chưa được cấu hình.")
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_CONFIG["url"],
            auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
        )
        driver.verify_connectivity()
        logging.info("Kết nối tới Neo4j AuraDB thành công.")
    except Exception as e:
        logging.error(f"Lỗi kết nối Neo4j: {e}")
        raise

    # 2. Cấu hình Gemini và LlamaIndex
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY chưa được cấu hình trong file .env")
        raise ValueError("GEMINI_API_KEY chưa được cấu hình.")
    
    Settings.llm = Gemini(model="models/gemini-pro", temperature=0.1)
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
    
    logging.info("Cấu hình LlamaIndex và Gemini LLM thành công.")
    
    return driver

import os
import logging
import sys
import time
from neo4j import GraphDatabase
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Import các module đã được tách
# Cấu trúc `.` có nghĩa là import từ cùng một package (thư mục `src`)
try:
    from .config import Config, NEO4J_CONFIG, GEMINI_API_KEY
    from .data_loader import (
        check_and_load_students, verify_graph, load_student_profile, 
        initialize_vector_index # Thêm hàm này từ data_loader
    )
    from .session_manager import run_learning_session
    from .recommendations import collaborative_filtering, apply_apriori
except ImportError:
    # Xử lý trường hợp chạy file trực tiếp (python src/main.py)
    from config import Config, NEO4J_CONFIG, GEMINI_API_KEY
    from data_loader import (
        check_and_load_students, verify_graph, load_student_profile,
        initialize_vector_index
    )
    from session_manager import run_learning_session
    from recommendations import collaborative_filtering, apply_apriori

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

def initialize_connections_and_settings():
    """Khởi tạo kết nối tới Neo4j và cấu hình LlamaIndex/Gemini."""
    logging.info("Đang khởi tạo các kết nối và thiết lập...")
    # 1. Kết nối tới Neo4j
    if not all(NEO4J_CONFIG.values()):
        logging.error("Thông tin kết nối Neo4j chưa được cấu hình trong file .env")
        raise ValueError("Thông tin kết nối Neo4j chưa được cấu hình.")
    
    try:
        driver = GraphDatabase.driver(
            NEO4J_CONFIG["url"],
            auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
        )
        driver.verify_connectivity()
        logging.info("Kết nối tới Neo4j AuraDB thành công.")
    except Exception as e:
        logging.error(f"Lỗi kết nối Neo4j: {e}")
        raise

    # 2. Cấu hình Gemini và LlamaIndex
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY chưa được cấu hình trong file .env")
        raise ValueError("GEMINI_API_KEY chưa được cấu hình.")
    
    Settings.llm = Gemini(model="models/gemini-pro", temperature=0.1)
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
    
    logging.info("Cấu hình LlamaIndex và Gemini LLM thành công.")
    
    return driver

# ==============================================================================
# HÀM MAIN VÀ ĐIỂM VÀO CHÍNH
# ==============================================================================

def main():
    """
    Hàm điều phối chính của ứng dụng, xử lý input người dùng và gọi các module xử lý.
    Đây là phiên bản tích hợp đầy đủ logic tương tác của bạn.
    """
    driver = None
    try:
        # 1. Thiết lập ban đầu
        setup_logging()
        driver = initialize_connections_and_settings()

        # 2. Tải và xác thực dữ liệu ban đầu
        verify_graph(driver)
        student_load_result = check_and_load_students(driver)
        if student_load_result["status"] != "success":
            raise ValueError(f"Lỗi tải dữ liệu học sinh: {student_load_result['error_message']}")

        # Khởi tạo VectorStoreIndex (quan trọng cho việc tìm kiếm)
        init_vector_result = initialize_vector_index(driver)
        if init_vector_result["status"] != "success":
            raise ValueError(f"Lỗi khởi tạo VectorStoreIndex: {init_vector_result['error_message']}")

        # 3. Thu thập thông tin từ người dùng (logic chi tiết của bạn)
        print("\n--- HỆ THỐNG ĐỀ XUẤT LỘ TRÌNH HỌC TẬP CÁ NHÂN HÓA ---")
        
        student_id = input("Nhập mã học sinh (ví dụ: stu001): ").strip()
        profile = load_student_profile(student_id, driver)
        if not profile or profile.get("student_id") != student_id:
            raise ValueError(f"Student ID '{student_id}' không tồn tại.")

        # Lấy trình độ
        while True:
            level = input(f"Nhập trình độ mong muốn ({'|'.join(Config.ASTAR_SKILL_LEVELS_LOW + Config.ASTAR_SKILL_LEVELS_HIGH)}): ").strip()
            if level in Config.ASTAR_SKILL_LEVELS_LOW + Config.ASTAR_SKILL_LEVELS_HIGH:
                break
            print("Trình độ không hợp lệ, vui lòng nhập lại.")

        # Lấy ngữ cảnh
        context = input(f"Nhập ngữ cảnh học tập (ví dụ: {Config.DEFAULT_CONTEXT_EXAMPLE}): ").strip()

        # Lấy phong cách học
        learning_style = input(f"Nhập phong cách học tập (Tùy chọn, ví dụ: {', '.join(Config.LEARNING_STYLE_VARK)}): ").strip()

        # Lấy mục tiêu học tập
        student_goal = ""
        while not student_goal:
            student_goal = input("Nhập mục tiêu học tập của bạn (Bắt buộc): ").strip()
            if not student_goal:
                print("Mục tiêu học tập không được để trống.")
        
        # Lấy lựa chọn sử dụng LLM
        use_llm_input = input("Bạn có muốn sử dụng AI để tạo lộ trình không? (yes/no): ").strip().lower()
        use_llm = True if use_llm_input == "yes" else False

        # 4. Chạy phiên học tập
        session_start_time = time.time()
        session_result = run_learning_session(student_id, level, context, student_goal, driver, use_llm)
        if session_result["status"] != "success":
            raise ValueError(f"Lỗi phiên học tập: {session_result['error_message']}")
        
        total_session_time = int((time.time() - session_start_time) / 60)
        logging.info(f"Tổng thời gian phiên học tập: {total_session_time} phút.")
        print(f"\nTổng thời gian phiên học tập: {total_session_time} phút.")

        # 5. Đề xuất bổ sung sau phiên học
        learned_nodes = session_result.get("learned_nodes", [])
        collab_result = collaborative_filtering(student_id, learned_nodes)
        if collab_result.get("status") == "success" and collab_result.get("recommended_nodes"):
            print(f"\nChúng tôi gợi ý thêm cho bạn các chủ đề sau: {collab_result.get('recommended_nodes')}")

        apriori_result = apply_apriori()
        if apriori_result.get("status") == "success" and apriori_result.get("association_rules"):
             print(f"\nMột số lộ trình học phổ biến: {apriori_result.get('association_rules')}")
        
        print("\nChúc mừng! Bạn đã hoàn thành chương trình.")

    except ValueError as ve:
        logging.warning(f"Lỗi đầu vào hoặc logic: {ve}")
        print(f"\nĐã có lỗi xảy ra: {ve}")
    except Exception as e:
        logging.critical(f"Ứng dụng gặp lỗi nghiêm trọng và đã dừng: {e}", exc_info=True)
        print(f"\nLỗi nghiêm trọng: {e}")
    finally:
        if 'driver' in locals() and driver:
            driver.close()
            logging.info("Đã đóng kết nối Neo4j.")

if __name__ == "__main__":
    main()