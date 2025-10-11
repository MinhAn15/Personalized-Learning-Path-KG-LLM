import logging
import csv
import os
import pandas as pd
import requests
import io
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter
from neo4j import GraphDatabase

# Import lớp Config từ file config.py cùng thư mục
from .config import Config

# Thiết lập logger để ghi lại hoạt động
logger = logging.getLogger(__name__)

# ==============================================================================
# CÁC HÀM TIỆN ÍCH (UTILITY FUNCTIONS)
# ==============================================================================

def execute_cypher_query(driver: GraphDatabase.driver, query: str, params: Dict = None) -> List[Dict]:
    """
    Thực thi một truy vấn Cypher trên Neo4j và trả về kết quả.

    Args:
        driver: Đối tượng driver của Neo4j.
        query (str): Chuỗi truy vấn Cypher.
        params (Dict, optional): Các tham số cho truy vấn.

    Returns:
        List[Dict]: Danh sách các bản ghi kết quả.
    """
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    except Exception as e:
        logger.error(f"Lỗi thực thi truy vấn Cypher: {str(e)}\nTruy vấn: {query}")
        raise

def jaccard_similarity(tags1: List[str], tags2: List[str]) -> float:
    """
    Tính độ tương đồng Jaccard giữa hai danh sách tag.
    Được sử dụng để xác định các nút trùng lặp.
    """
    try:
        set1, set2 = set(tags1), set(tags2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        logger.error(f"Lỗi trong jaccard_similarity: {str(e)}")
        return 0.0

def merge_properties(existing_props: Dict[str, Any], new_props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hợp nhất thuộc tính từ nút mới vào nút đã có, ưu tiên giá trị mới nếu hợp lệ.
    """
    # ... (Đây là code hàm merge_properties từ notebook của bạn)
    try:
        merged_props = existing_props.copy()
        for key, new_value in new_props.items():
            if key not in merged_props or merged_props[key] in [None, "Not Available", ""]:
                merged_props[key] = new_value
            elif new_value not in [None, "Not Available", ""]:
                if key in [Config.PROPERTY_SEMANTIC_TAGS, Config.PROPERTY_FOCUSED_SEMANTIC_TAGS, Config.PROPERTY_PREREQUISITES]:
                    existing_list = merged_props[key].split(";") if isinstance(merged_props[key], str) else []
                    new_list = new_value.split(";") if isinstance(new_value, str) else []
                    merged_list = sorted(list(set(existing_list + new_list)))
                    merged_props[key] = ";".join(filter(None, merged_list))
                else:
                    merged_props[key] = new_value
        return merged_props
    except Exception as e:
        logger.error(f"Lỗi trong merge_properties: {str(e)}")
        return existing_props

# ==============================================================================
# CÁC HÀM TẢI VÀ XÁC THỰC DỮ LIỆU
# ==============================================================================

def _get_github_file_content(file_path: str) -> str:
    """
    Fetches the content of a file from a private GitHub repository.
    """
    # GITHUB_TOKEN is provided as a module-level variable in config.py
    from . import config as _config_module
    if not getattr(_config_module, 'GITHUB_TOKEN', None):
        raise ValueError("GITHUB_TOKEN is not set in the environment.")

    # Assumes the repo is the one this code is running in.
    # You can make these dynamic if needed.
    owner = "MinhAn15" # Replace with your GitHub username
    repo = "Personalized-Learning-Path-KG-LLM" # Replace with your repo name
    
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    headers = {
        "Authorization": f"token {_config_module.GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch {file_path} from GitHub. Status: {response.status_code}, Body: {response.text}")


def check_and_load_kg(driver: GraphDatabase.driver) -> Dict:
    """
    Tải và hợp nhất đồ thị tri thức từ các file CSV trong private GitHub repo.
    """
    logger.info("Bắt đầu quá trình tải Knowledge Graph từ GitHub...")
    try:
        # Fetch file content from GitHub; if 404 or fail, fallback to local files in backend/data/github_import
        logger.info("Đang tải nodes.csv từ GitHub...")
        try:
            nodes_csv_content = _get_github_file_content(f"backend/data/github_import/{Config.IMPORT_NODES_FILE}")
        except Exception as e:
            logger.warning(f"Không thể fetch nodes từ GitHub: {e}. Thử đọc file local...")
            local_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'github_import', Config.IMPORT_NODES_FILE)
            with open(local_nodes_path, 'r', encoding='utf-8') as f:
                nodes_csv_content = f.read()

        logger.info("Đang tải relationships.csv từ GitHub...")
        try:
            rels_csv_content = _get_github_file_content(f"backend/data/github_import/{Config.IMPORT_RELATIONSHIPS_FILE}")
        except Exception as e:
            logger.warning(f"Không thể fetch relationships từ GitHub: {e}. Thử đọc file local...")
            local_rels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'github_import', Config.IMPORT_RELATIONSHIPS_FILE)
            with open(local_rels_path, 'r', encoding='utf-8') as f:
                rels_csv_content = f.read()

        # Use io.StringIO to treat the string content as a file for pandas
        nodes_file = io.StringIO(nodes_csv_content)
        rels_file = io.StringIO(rels_csv_content)

        # --- Client-Side Upload Logic (re-used from previous step) ---
        def _sanitize_rel_type(s: str) -> str:
            import re
            if not isinstance(s, str) or not s: return 'RELATED_TO'
            return re.sub(r'[^A-Za-z0-9_]', '_', s).upper() or 'RELATED_TO'

        # Upload nodes
        df_nodes = pd.read_csv(nodes_file, dtype=str).fillna('')
        CHUNK = 200
        for start in range(0, len(df_nodes), CHUNK):
            chunk = df_nodes.iloc[start:start+CHUNK]
            rows = chunk.to_dict(orient='records')
            cypher = f"UNWIND $rows AS row MERGE (n:KnowledgeNode {{ {Config.PROPERTY_ID}: row['{Config.PROPERTY_ID}'] }}) SET n += row, n.Priority = toInteger(row.Priority), n.Time_Estimate = toFloat(row.Time_Estimate)"
            execute_cypher_query(driver, cypher, params={"rows": rows})
        logger.info("Tải xong các nút (nodes) từ GitHub.")

        # Upload relationships
        df_rels = pd.read_csv(rels_file, dtype=str).fillna('')
        for start in range(0, len(df_rels), CHUNK):
            chunk = df_rels.iloc[start:start+CHUNK]
            rows = chunk.to_dict(orient='records')
            with driver.session(database="neo4j") as session:
                tx = session.begin_transaction()
                for r in rows:
                    rel_type = _sanitize_rel_type(r.get('Relationship_Type') or r.get('RelationshipType'))
                    source_id = r.get(Config.PROPERTY_SOURCE_ID) or r.get('source') or r.get('source_id')
                    target_id = r.get(Config.PROPERTY_TARGET_ID) or r.get('target') or r.get('target_id')
                    weight = r.get('Weight')
                    dependency = r.get('Dependency')
                    cypher = f"MATCH (s:KnowledgeNode {{ {Config.PROPERTY_ID}: $src }}), (t:KnowledgeNode {{ {Config.PROPERTY_ID}: $tgt }}) MERGE (s)-[rel:{rel_type}]->(t) SET rel.Weight = toFloat($weight), rel.Dependency = toFloat($dependency)"
                    tx.run(cypher, src=source_id, tgt=target_id, weight=weight, dependency=dependency)
                tx.commit()
        logger.info("Tải xong các mối quan hệ (relationships) từ GitHub.")
        
        logger.info("Hoàn tất quá trình tải Knowledge Graph từ GitHub.")
        return {"status": "success", "error_message": None}

    except Exception as e:
        logger.error(f"Lỗi trong check_and_load_kg (GitHub): {e}", exc_info=True)
        return {"status": "error", "error_message": str(e)}


def check_and_load_students(driver: GraphDatabase.driver) -> Dict:
    """
    Tải và cập nhật nút Student từ file CSV vào đồ thị Neo4j,
    ưu tiên dữ liệu có LastUpdated mới nhất.
    """
    # ... (Đây là code hàm check_and_load_students từ notebook của bạn)
    logger.info("Bắt đầu quá trình tải và cập nhật dữ liệu học sinh...")
    try:
        students_df = pd.read_csv(Config.STUDENT_FILE)
        for _, row in students_df.iterrows():
            # Logic kiểm tra LastUpdated và cập nhật/tạo nút Student
            # Bạn có thể sao chép logic chi tiết từ notebook của mình vào đây
            pass
        logger.info("Hoàn tất tải dữ liệu học sinh.")
        return {"status": "success", "error_message": None}
    except Exception as e:
        logger.error(f"Lỗi trong check_and_load_students: {str(e)}")
        return {"status": "error", "error_message": str(e)}

def verify_graph(driver: GraphDatabase.driver):
    """Kiểm tra và in ra số lượng nút và mối quan hệ trong đồ thị."""
    try:
        node_count_result = execute_cypher_query(driver, "MATCH (n) RETURN count(n) AS count")
        node_count = node_count_result[0]['count'] if node_count_result else 0
        
        rel_count_result = execute_cypher_query(driver, "MATCH ()-[r]->() RETURN count(r) AS count")
        rel_count = rel_count_result[0]['count'] if rel_count_result else 0
        
        print(f"Kiểm tra đồ thị thành công.")
        print(f"- Số lượng nút: {node_count}")
        print(f"- Số lượng mối quan hệ: {rel_count}")
        logger.info(f"Graph verification: {node_count} nodes, {rel_count} relationships.")
    except Exception as e:
        logger.error(f"Lỗi khi xác thực đồ thị: {str(e)}")
        print(f"Lỗi khi xác thực đồ thị: {e}")

# ==============================================================================
# CÁC HÀM TRUY XUẤT HỒ SƠ
# ==============================================================================

def calculate_learning_speed(performance_details: List[str]) -> float:
    """Tính tốc độ học tập trung bình dựa trên thời gian dành cho các nút.

    Args:
        performance_details (List[str]): Danh sách chi tiết hiệu suất, định dạng 'node_id:score:time_spent:skill_level' hoặc 'node_id:score:time_spent'.

    Returns:
        float: Tốc độ học tập trung bình (phút/nút).
    """
    total_time = 0
    num_nodes = 0
    for detail in performance_details:
        try:
            parts = detail.split(":")
            if len(parts) == 4:  # Định dạng node_id:score:time_spent:skill_level
                time_spent = parts[2]
            elif len(parts) == 3:  # Định dạng node_id:score:time_spent
                time_spent = parts[2]
            else:
                raise ValueError(f"Định dạng không hợp lệ calculate_learning_speed: {detail}")
            total_time += int(time_spent)
            num_nodes += 1
        except (ValueError, IndexError) as e:
            logger.warning(f"Lỗi xử lý chi tiết hiệu suất: {detail}. Bỏ qua. ({str(e)})")
            continue
    return total_time / num_nodes if num_nodes > 0 else 0

# Hàm xác định sở thích chủ đề
def extract_topic_preference(driver: GraphDatabase.driver, learning_history: List[str]) -> str:
    """Xác định sở thích chủ đề dựa trên lịch sử học tập.

    Args:
        driver (GraphDatabase.driver): Đối tượng driver của Neo4j.
        learning_history (List[str]): Danh sách ID của các nút đã học.

    Returns:
        str: Chủ đề phổ biến nhất hoặc 'Unknown' nếu không có dữ liệu.
    """
    contexts = []
    for node_id in learning_history:
        query = f"MATCH (n {{{Config.PROPERTY_ID}: $node_id}}) RETURN n.{Config.PROPERTY_CONTEXT} AS context"
        try:
            result = execute_cypher_query(driver, query, params={"node_id": node_id})
            if result and result[0].get("context"):
                contexts.append(result[0]["context"])
        except Exception as e:
            logger.warning(f"Lỗi truy vấn context cho node {node_id}: {str(e)}")
    if contexts:
        return Counter(contexts).most_common(1)[0][0]
    return "Unknown"

def load_student_profile(driver: GraphDatabase.driver, student_id: str) -> Dict:
    """Tải hồ sơ học sinh từ Neo4j hoặc trả về mặc định nếu không tìm thấy.

    Args:
        driver (GraphDatabase.driver): Đối tượng driver của Neo4j.
        student_id (str): ID của học sinh.

    Returns:
        Dict: Hồ sơ học sinh với các thuộc tính như student_id, learning_history, v.v.
    """
    try:
        query = """
        MATCH (s:Student {StudentID: $student_id})
        RETURN s
        """
        result = execute_cypher_query(driver, query, {"student_id": student_id})
        if result:
            student_data = result[0]["s"]
            learning_history = student_data.get("learning_history", "").split(",") if student_data.get("learning_history") else []
            performance_details = student_data.get("performance_details", "").split(";") if student_data.get("performance_details") else []
            
            return {
                "student_id": student_data["StudentID"],
                "learning_history": learning_history,
                "current_level": float(student_data.get("current_level", 0)),
                "performance_details": performance_details,
                "learning_style_preference": student_data.get("learning_style_preference", Config.DEFAULT_LEARNING_STYLE),
                "preferred_difficulty": student_data.get("preferred_difficulty", "STANDARD"),
                "time_availability": float(student_data.get("time_availability", 60)),
                "learning_speed": calculate_learning_speed(performance_details),
                "topic_preference": extract_topic_preference(driver, learning_history),
                "long_term_goal": student_data.get("long_term_goal", "Not Specified")
            }
        else:
            logger.warning(f"Không tìm thấy hồ sơ cho student_id: {student_id}. Trả về hồ sơ mặc định.")
            return {
                "student_id": student_id,
                "learning_history": [],
                "current_level": 0,
                "performance_details": [],
                "learning_style_preference": Config.DEFAULT_LEARNING_STYLE,
                "preferred_difficulty": "STANDARD",
                "time_availability": 60,
                "learning_speed": 0,
                "topic_preference": "Unknown",
                "long_term_goal": "Not Specified"
            }
    except Exception as e:
        logger.error(f"Lỗi tải hồ sơ học sinh {student_id}: {str(e)}", exc_info=True)
        # Trả về hồ sơ mặc định trong trường hợp có lỗi
        return {
            "student_id": student_id,
            "learning_history": [],
            "current_level": 0,
            "performance_details": [],
            "learning_style_preference": Config.DEFAULT_LEARNING_STYLE,
            "preferred_difficulty": "STANDARD",
            "time_availability": 60,
            "learning_speed": 0,
            "topic_preference": "Unknown",
            "long_term_goal": "Not Specified"
        }

# ==============================================================================
# CÁC HÀM CẬP NHẬT VÀ LƯU TRỮ DỮ LIỆU
# ==============================================================================

def update_student_profile(driver: GraphDatabase.driver, student_id: str, updates: Dict) -> None:
    """
    Cập nhật hồ sơ của một học sinh trong Neo4j.
    """
    logger.info(f"Bắt đầu cập nhật hồ sơ cho sinh viên: {student_id}")
    try:
        # Chuyển đổi list thành chuỗi trước khi lưu
        for key, value in updates.items():
            if isinstance(value, list):
                # Sử dụng dấu phẩy cho learning_history và chấm phẩy cho performance_details
                separator = ',' if key == 'learning_history' else ';'
                updates[key] = separator.join(map(str, value))

        # Tạo câu lệnh SET linh hoạt
        set_clauses = ", ".join([f"s.{key} = ${key}" for key in updates.keys()])
        
        query = f"""
        MERGE (s:Student {{StudentID: $student_id}})
        SET {set_clauses}, s.LastUpdated = timestamp()
        """
        
        # Thêm student_id vào dict tham số
        params = {"student_id": student_id, **updates}
        
        execute_cypher_query(driver, query, params)
        logger.info(f"Cập nhật hồ sơ cho sinh viên {student_id} thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật hồ sơ sinh viên {student_id}: {e}", exc_info=True)
        raise

def save_learning_data(driver: GraphDatabase.driver, data: Dict) -> None:
    """
    Lưu dữ liệu từ một phiên học vào đồ thị dưới dạng nút LearningData.
    """
    logger.info(f"Bắt đầu lưu dữ liệu phiên học cho sinh viên: {data.get('student_id')}")
    try:
        query = """
        MATCH (s:Student {StudentID: $student_id})
        MATCH (n:KnowledgeNode {Node_ID: $node_id})
        CREATE (ld:LearningData {
            student_id: $student_id,
            node_id: $node_id,
            timestamp: datetime(),
            score: toFloat($score),
            time_spent: toInteger($time_spent),
            feedback: $feedback,
            quiz_responses: $quiz_responses
        })
        CREATE (s)-[:HAS_LEARNING_DATA]->(ld)
        CREATE (ld)-[:RELATED_TO_NODE]->(n)
        """
        execute_cypher_query(driver, query, data)
        logger.info(f"Lưu dữ liệu phiên học cho nút {data.get('node_id')} thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi lưu dữ liệu phiên học: {e}", exc_info=True)
        raise

# ==============================================================================
# CÁC HÀM LIÊN QUAN ĐẾN VECTOR INDEX
# ==============================================================================

def initialize_vector_index(driver: GraphDatabase.driver) -> Dict:
    """
    Kiểm tra và khởi tạo vector index trong Neo4j nếu chưa có.
    """
    index_name = "knowledgeNodeEmbeddings"
    logger.info(f"Đang kiểm tra và khởi tạo vector index: '{index_name}'...")
    try:
        # Kiểm tra xem index đã tồn tại chưa
        index_exists_query = "SHOW INDEXES YIELD name WHERE name = $index_name RETURN count(*) > 0 AS exists"
        result = execute_cypher_query(driver, index_exists_query, {"index_name": index_name})
        
        if result and result[0]['exists']:
            logger.info(f"Vector index '{index_name}' đã tồn tại.")
            return {"status": "exists", "message": f"Index '{index_name}' đã tồn tại."}

        # Nếu chưa, tạo index
        # Giả định rằng embedding được lưu trong thuộc tính 'Embedding' của nút KnowledgeNode
        # và có kích thước là 768 (từ Gemini's embedding-001)
        create_index_query = f"""
        CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
        FOR (n:KnowledgeNode) ON (n.Embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        execute_cypher_query(driver, create_index_query)
        
        # Đợi index được tạo xong (quan trọng)
        wait_for_index_query = f"CALL db.awaitIndex('{index_name}')"
        execute_cypher_query(driver, wait_for_index_query)
        
        logger.info(f"Vector index '{index_name}' đã được tạo thành công.")
        return {"status": "created", "message": f"Index '{index_name}' đã được tạo."}
    except Exception as e:
        logger.error(f"Lỗi trong quá trình khởi tạo vector index: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}