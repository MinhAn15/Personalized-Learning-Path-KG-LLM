from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Dict, Any
import threading

# Import các hàm logic chính từ các module của bạn
from .session_manager import run_learning_session
from .session_store import (
    append_chat_message,
    create_session,
    get_chat_history,
    get_session,
    update_session_metadata,
    update_session_path,
)
from .graph_recommender import GraphRecommender
from .chat_service import ChatService
# Defer importing initialization function to avoid circular import at module load time

# --- Khởi tạo ứng dụng và các kết nối ---
app = FastAPI()
driver = None
graph_recommender: GraphRecommender | None = None
chat_service = ChatService()
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
            global driver, graph_recommender, chat_service
            try:
                d = initialize_connections_and_settings()
                driver = d
                graph_recommender = GraphRecommender(driver)
                chat_service.set_driver(driver)
                logger.info("Khởi tạo API thành công (background).")
            except Exception as e:
                logger.error(f"Lỗi nghiêm trọng khi khởi tạo API trong background: {e}")
                driver = None
                graph_recommender = None
                chat_service.set_driver(None)

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


class NextConceptRequest(BaseModel):
    student_id: str
    top_k: int = 5


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    learner_id: str | None = None
    context: str | None = None
    goal: str | None = None

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


@app.post("/api/recommend_next_concept")
def recommend_next_concept(request: NextConceptRequest):
    global graph_recommender
    if not driver:
        return {"status": "error", "message": "Neo4j not connected"}
    if graph_recommender is None:
        graph_recommender = GraphRecommender(driver)
    try:
        payload = graph_recommender.recommend_next_concept(request.student_id, request.top_k)
        return {"status": "ok", **payload}
    except Exception as exc:
        logger.exception("Failed to generate next concept recommendation")
        return {"status": "error", "message": str(exc)}


@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    global chat_service
    try:
        session = get_session(request.session_id) if request.session_id else None
        if not session:
            session = create_session(
                {
                    "source": "chat",
                    "learner_id": request.learner_id,
                    "student_id": request.learner_id,
                    "context": request.context,
                    "goal": request.goal,
                    "student_goal": request.goal,
                }
            )

        session_id = session["session_id"]

        metadata_updates: Dict[str, Any] = {}
        if request.learner_id:
            metadata_updates["learner_id"] = request.learner_id
            metadata_updates.setdefault("student_id", request.learner_id)
        if request.context:
            metadata_updates["context"] = request.context
        if request.goal:
            metadata_updates["goal"] = request.goal
            metadata_updates.setdefault("student_goal", request.goal)
        if metadata_updates:
            update_session_metadata(session_id, metadata_updates)
            session = get_session(session_id) or session

        append_chat_message(
            session_id,
            "user",
            request.message,
            {
                "learner_id": request.learner_id,
                "context": request.context,
                "goal": request.goal,
            },
        )

        reply_payload = chat_service.generate_reply(
            session,
            request.message,
            {
                "learner_id": request.learner_id,
                "context": request.context,
                "goal": request.goal,
            },
        )

        append_chat_message(
            session_id,
            "assistant",
            reply_payload["reply"],
            {
                "model": reply_payload.get("model"),
                "fallback": reply_payload.get("fallback"),
            },
        )

        history = get_chat_history(session_id, limit=30) or []

        return {
            "status": "ok",
            "session_id": session_id,
            "reply": reply_payload["reply"],
            "chat_history": history,
            "supporting_nodes": reply_payload.get("supporting_nodes", []),
            "summary": reply_payload.get("summary"),
            "context": reply_payload.get("context"),
            "model": reply_payload.get("model"),
            "fallback": reply_payload.get("fallback"),
        }
    except Exception as exc:
        logger.exception("Chat endpoint failed")
        return {"status": "error", "message": str(exc)}


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