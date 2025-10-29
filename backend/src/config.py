import os
from dotenv import load_dotenv

# File .env sẽ chứa các thông tin nhạy cảm như API keys và credentials
load_dotenv()

class Config:
    """
    Lớp chứa các hằng số và cấu hình toàn cục cho dự án.
    Điều này giúp quản lý tập trung và dễ dàng thay đổi các tham số.
    """
    # ==============================================================================
    # ĐƯỜNG DẪN TỆP VÀ THƯ MỤC (Cấu hình cho môi trường local)
    # ==============================================================================
    # Xác định thư mục gốc của dự án (một cấp trên thư mục 'src' này)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Các thư mục con
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

    # Đường dẫn tệp dữ liệu đầu vào
    STUDENT_FILE = os.path.join(DATA_DIR, "input", "student.csv")
    
    # Đường dẫn cho Neo4j import (file nằm trong thư mục 'import' của Neo4j)
    IMPORT_NODES_FILE = "master_nodes.csv"
    IMPORT_RELATIONSHIPS_FILE = "master_relationships.csv"

    # Đường dẫn tệp dữ liệu đầu ra và lưu trữ
    LEARNING_PATHS_FILE = os.path.join(OUTPUT_DIR, "learning_paths.csv")
    LEARNING_DATA_FILE = os.path.join(OUTPUT_DIR, "learning_data.csv")
    TEACHER_NOTIFICATION_FILE = os.path.join(OUTPUT_DIR, "teacher_notifications.csv")

    # Thư mục lưu trữ cho LlamaIndex
    PROPERTY_GRAPH_STORAGE_DIR = os.path.join(STORAGE_DIR, "property_graph_storage")
    VECTOR_INDEX_STORAGE_DIR = os.path.join(STORAGE_DIR, "vector_index_storage")
    DOCUMENT_STORAGE_DIR = os.path.join(STORAGE_DIR, "document_storage")

    # ==============================================================================
    # THUỘC TÍNH NÚT VÀ MỐI QUAN HỆ
    # ==============================================================================
    PROPERTY_ID = "Node_ID"
    PROPERTY_SANITIZED_CONCEPT = "Sanitized_Concept"
    PROPERTY_CONTEXT = "Context"
    PROPERTY_DEFINITION = "Definition"
    PROPERTY_EXAMPLE = "Example"
    PROPERTY_LEARNING_OBJECTIVE = "Learning_Objective"
    PROPERTY_SKILL_LEVEL = "Skill_Level"
    PROPERTY_TIME_ESTIMATE = "Time_Estimate"
    PROPERTY_DIFFICULTY = "Difficulty"
    PROPERTY_PRIORITY = "Priority"
    PROPERTY_PREREQUISITES = "Prerequisites"
    PROPERTY_SEMANTIC_TAGS = "Semantic_Tags"
    PROPERTY_FOCUSED_SEMANTIC_TAGS = "Focused_Semantic_Tags"

    # Column names used in relationship CSVs
    PROPERTY_SOURCE_ID = "Source_ID"
    PROPERTY_TARGET_ID = "Target_ID"

    # Thuộc tính nút (tùy chọn)
    PROPERTY_COMMON_ERRORS = "Common_Errors"
    PROPERTY_LEARNING_STYLE_PREFERENCE = "Learning_Style_Preference"

    # Loại quan hệ
    RELATIONSHIP_REQUIRES = "REQUIRES"
    RELATIONSHIP_IS_PREREQUISITE_OF = "IS_PREREQUISITE_OF"
    RELATIONSHIP_NEXT = "NEXT"
    RELATIONSHIP_REMEDIATES = "REMEDIATES"
    RELATIONSHIP_HAS_ALTERNATIVE_PATH = "HAS_ALTERNATIVE_PATH"
    RELATIONSHIP_SIMILAR_TO = "SIMILAR_TO"
    RELATIONSHIP_IS_SUBCONCEPT_OF = "IS_SUBCONCEPT_OF"
    RELATIONSHIP_TYPES = [
        "REQUIRES", "IS_PREREQUISITE_OF", "NEXT", "REMEDIATES",
        "HAS_ALTERNATIVE_PATH", "SIMILAR_TO", "IS_SUBCONCEPT_OF"
    ]


    # ==============================================================================
    # CẤU HÌNH BÀI KIỂM TRA (QUIZ)
    # ==============================================================================
    QUIZ_NUM_QUESTIONS = 15
    QUIZ_DISTRIBUTION = {"basic": 0.5, "intermediate": 0.3, "advanced": 0.2}
    QUIZ_PASSING_SCORE = 70

    # ==============================================================================
    # THAM SỐ THUẬT TOÁN A*
    # ==============================================================================
    ASTAR_DIFFICULTY_FILTER = ["STANDARD", "ADVANCED"]
    ASTAR_SKILL_LEVELS_LOW = ["Remember", "Understand"]
    ASTAR_SKILL_LEVELS_HIGH = ["Apply", "Analyze", "Evaluate", "Create"]
    ASTAR_CURRENT_LEVEL_THRESHOLD = 50
    ASTAR_CURRENT_LEVEL_WEIGHT = 0.4
    ASTAR_ASSESSMENT_WEIGHT = 0.6
    
    ASTAR_HEURISTIC_WEIGHTS = {
        "priority": 1.0,
        "difficulty_standard": 1.0,
        "difficulty_advanced": 2.0,
        "skill_level": {
            "Remember": 1.0, "Understand": 2.0, "Apply": 3.0,
            "Analyze": 4.0, "Evaluate": 5.0, "Create": 6.0,
            "default": 0.5
        },
        "time_estimate": 1.0 / 60
    }

    # ==============================================================================
    # ACADEMIC ENUMS & CONSTANTS (for advanced modules)
    # ==============================================================================
    # Bloom's Taxonomy (uppercase for KG concept alignment)
    COGNITIVE_LEVELS = [
        "REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE", "EVALUATE", "CREATE"
    ]
    # Difficulty score bounds (quantified scale)
    DIFFICULTY_SCORE_MIN = 0.0
    DIFFICULTY_SCORE_MAX = 10.0
    # ZPD defaults (Zone of Proximal Development) — coarse placeholders
    ZPD_DEFAULT_LOWER = 0.3  # lower mastery bound
    ZPD_DEFAULT_UPPER = 0.7  # upper mastery bound
    # Spaced repetition defaults (Ebbinghaus-like)
    FORGETTING_BASE = 0.5
    REVIEW_MIN_DAYS = 1
    REVIEW_MAX_DAYS = 30

    # ==============================================================================
    # CẤU HÌNH CHUNG
    # ==============================================================================
    LEARNING_STYLE_VARK = ["visual", "auditory", "reading_writing", "kinesthetic"]
    DEFAULT_LEARNING_STYLE = "reading_writing"
    DEFAULT_CONTEXT_EXAMPLE = "e_learning"
    MIN_SIMILARITY_THRESHOLD = 0.5

    # Cấu hình cho việc tải dữ liệu từ GitHub
    IMPORT_NODES_FILE = "master_nodes.csv"
    IMPORT_RELATIONSHIPS_FILE = "master_relationships.csv"

    # Thư mục để lưu trữ các bài quiz được tạo ra
    QUIZ_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'output')

    # Biến toàn cục để giữ đối tượng LLM sau khi khởi tạo
    LLM = None

# ==============================================================================
# CẤU HÌNH KẾT NỐI (Credentials từ file .env)
# ==============================================================================
NEO4J_CONFIG = {
    "url": os.getenv("NEO4J_URL"),
    "username": os.getenv("NEO4J_USER"),
    "password": os.getenv("NEO4J_PASSWORD"),
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# LlamaIndex và một số thư viện khác tìm key trong biến môi trường
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
else:
    print("Cảnh báo: GEMINI_API_KEY không được tìm thấy trong file .env")

# Kiểm tra xem các biến môi trường đã được tải thành công chưa
if not all(NEO4J_CONFIG.values()) or not GEMINI_API_KEY:
    print("Cảnh báo: Một hoặc nhiều biến môi trường (Neo4j, Gemini) chưa được thiết lập trong file .env.")
    print("Vui lòng tạo file .env ở thư mục gốc và thêm các thông tin cần thiết.")