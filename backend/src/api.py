from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Dict, Any
import threading

# Import các hàm logic chính từ các module của bạn
from .session_manager import run_learning_session
from .session_store import create_session, get_session, update_session_path
# Defer importing initialization function to avoid circular import at module load time

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

    # Run the potentially-blocking initialization in a background thread so
    # the ASGI lifespan doesn't hang. This lets uvicorn start promptly and
    # the app will fill `driver` when initialization completes.
    try:
        # Import here to avoid circular import at module import time
        from .main import initialize_connections_and_settings

        def _init_bg():
            global driver
            try:
                d = initialize_connections_and_settings()
                driver = d
                logger.info("Khởi tạo API thành công (background).")
            except Exception as e:
                logger.error(f"Lỗi nghiêm trọng khi khởi tạo API trong background: {e}")
                driver = None

        threading.Thread(target=_init_bg, daemon=True).start()
        logger.info("Bắt đầu khởi tạo kết nối trong background.")
    except Exception as e:
        logger.error(f"Lỗi khi bắt đầu background init: {e}")
        driver = None


@app.get("/api/status")
def status() -> Dict[str, Any]:
    """Return whether the server is configured to run the real pipeline.

    This endpoint is safe to call even if Neo4j wasn't initialized; it checks
    environment/config flags (Neo4j creds and Gemini key) and reports which
    services look available so the frontend can choose demo vs real endpoints.
    """
    try:
        # Import config lazily to avoid circular import issues during startup
        from . import config as _config

        neo_ok = bool(_config.NEO4J_CONFIG.get("url")) and bool(_config.NEO4J_CONFIG.get("username")) and bool(_config.NEO4J_CONFIG.get("password"))
        gemini_ok = bool(_config.GEMINI_API_KEY)
        github_ok = bool(_config.GITHUB_TOKEN)

        real_enabled = neo_ok and gemini_ok

        message_parts = []
        if not neo_ok:
            message_parts.append("Neo4j credentials missing")
        if not gemini_ok:
            message_parts.append("Gemini API key missing")
        if not github_ok:
            message_parts.append("GitHub token missing (optional for private repos)")

        return {
            "real_enabled": real_enabled,
            "neo4j": neo_ok,
            "gemini": gemini_ok,
            "github": github_ok,
            "message": ", ".join(message_parts) if message_parts else "all required env vars present",
        }

    except Exception as e:
        logger.exception("Error while evaluating status")
        return {"real_enabled": False, "neo4j": False, "gemini": False, "github": False, "message": str(e)}

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


@app.post("/api/generate_path_demo")
def generate_path_demo(request: LearningRequest):
    """Demo endpoint returning canned learning-path data so the frontend can be previewed without Neo4j."""
    logger.info(f"Demo generate_path called for: {request.student_id}")

    # Minimal canned response - shape mimics what the real run_learning_session might return
    demo_response = {
        "status": "ok",
        "student_id": request.student_id or "demo-student",
        "path": [
            {"id": "node-1", "title": "Introduction to SQL", "estimated_minutes": 20},
            {"id": "node-2", "title": "SELECT and WHERE", "estimated_minutes": 30},
            {"id": "node-3", "title": "GROUP BY and Aggregates", "estimated_minutes": 40},
        ],
        "explanation": f"A simple demo path for goal: {request.student_goal}",
        "use_llm": bool(request.use_llm),
    }

    return demo_response


@app.get("/api/node/{node_id}")
def get_node(node_id: str):
    """Return basic node metadata from Neo4j if available.

    This endpoint is tolerant: if the Neo4j driver isn't ready it returns an
    informative error so the frontend can fall back to demo data.
    """
    try:
        from . import config as _config
        if not driver:
            return {"status": "error", "message": "Neo4j not connected"}

        # Use the Config.PROPERTY_ID name to build the query
        from .config import Config
        q = f"MATCH (n:KnowledgeNode {{{Config.PROPERTY_ID}: $node_id}}) RETURN n LIMIT 1"
        records = []
        try:
            records = []
            with driver.session(database="neo4j") as session:
                res = session.run(q, node_id=node_id)
                for r in res:
                    records.append(r["n"])
        except Exception as e:
            return {"status": "error", "message": f"Query failed: {e}"}

        if not records:
            return {"status": "not_found", "message": f"Node {node_id} not found"}

        node = records[0]
        # Convert neo4j Node to dict safely
        props = dict(node)
        return {"status": "ok", "node": props}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/session")
def create_session_endpoint(request: LearningRequest):
    """Create a demo session and immediately attempt to generate a path.

    If Neo4j is available, this will call the real `generate_learning_path`
    via `run_learning_session` logic; otherwise it will return a demo path.
    """
    try:
        payload = request.dict()

        # Create session placeholder
        sess = create_session(payload)

        # If driver available, try to generate a real path asynchronously (sync here
        # for simplicity). If driver missing, return demo path structure.
        if driver:
            try:
                # run_learning_session returns a dict with status/path
                result = run_learning_session(
                    student_id=payload.get("student_id"),
                    level=payload.get("level"),
                    context=payload.get("context"),
                    learning_style=None,
                    student_goal=payload.get("student_goal"),
                    use_llm=payload.get("use_llm", False),
                    driver=driver,
                )
                if result.get("status") == "success":
                    update_session_path(sess["session_id"], result.get("path") or [])
                    sess = get_session(sess["session_id"])
                    return {"status": "ok", "session": sess}
                else:
                    # Return session with error message
                    sess["status"] = "error"
                    sess["error_message"] = result.get("error_message")
                    return {"status": "error", "session": sess}
            except Exception as e:
                sess["status"] = "error"
                sess["error_message"] = str(e)
                return {"status": "error", "session": sess}
        else:
            # No driver: return demo path as in generate_path_demo
            demo_path = [
                {"id": "node-1", "title": "Introduction to SQL", "estimated_minutes": 20},
                {"id": "node-2", "title": "SELECT and WHERE", "estimated_minutes": 30},
                {"id": "node-3", "title": "GROUP BY and Aggregates", "estimated_minutes": 40},
            ]
            update_session_path(sess["session_id"], demo_path)
            sess = get_session(sess["session_id"])
            return {"status": "ok", "session": sess}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/session/{session_id}")
def get_session_endpoint(session_id: str):
    s = get_session(session_id)
    if not s:
        return {"status": "not_found", "message": "Session not found"}
    return {"status": "ok", "session": s}