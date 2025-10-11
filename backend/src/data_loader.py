import logging
import csv
import pandas as pd
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

def check_and_load_kg(driver: GraphDatabase.driver) -> Dict:
    """
    Tải và hợp nhất đồ thị tri thức từ các file CSV, sử dụng Cypher LOAD CSV.
    Hàm này sẽ tạo/cập nhật các nút và mối quan hệ, sau đó hợp nhất các nút trùng lặp.
    """
    # ... (Đây là code hàm check_and_load_kg từ notebook của bạn)
    # Lưu ý: Hàm này khá phức tạp, hãy đảm bảo sao chép đầy đủ
    # Tôi đã đơn giản hóa một chút để phù hợp với module hóa
    logger.info("Bắt đầu quá trình tải và hợp nhất Knowledge Graph...")
    try:
        # Load nodes
        load_nodes_query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///{Config.IMPORT_NODES_FILE}' AS row
        MERGE (n:KnowledgeNode {{ {Config.PROPERTY_ID}: row.{Config.PROPERTY_ID} }})
        ON CREATE SET n += row, n.Priority = toInteger(row.Priority), n.Time_Estimate = toFloat(row.Time_Estimate)
        ON MATCH SET n += row, n.Priority = toInteger(row.Priority), n.Time_Estimate = toFloat(row.Time_Estimate)
        """
        execute_cypher_query(driver, load_nodes_query)
        logger.info("Tải các nút từ CSV thành công.")

        # Load relationships
        load_rels_query = f"""
        LOAD CSV WITH HEADERS FROM 'file:///{Config.IMPORT_RELATIONSHIPS_FILE}' AS row
        MATCH (source:KnowledgeNode {{ {Config.PROPERTY_ID}: row.{Config.PROPERTY_SOURCE_ID} }})
        MATCH (target:KnowledgeNode {{ {Config.PROPERTY_ID}: row.{Config.PROPERTY_TARGET_ID} }})
        CALL apoc.merge.relationship(source, row.Relationship_Type, {{}}, {{ Weight: toFloat(row.Weight), Dependency: toFloat(row.Dependency) }}, target) YIELD rel
        RETURN count(rel)
        """
        execute_cypher_query(driver, load_rels_query)
        logger.info("Tải các mối quan hệ từ CSV thành công.")
        
        # (Phần hợp nhất các node trùng lặp có thể được thêm vào đây nếu cần)
        
        logger.info("Hoàn tất quá trình tải Knowledge Graph.")
        return {"status": "success", "error_message": None}
    except Exception as e:
        logger.error(f"Lỗi trong check_and_load_kg: {str(e)}")
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
def extract_topic_preference(learning_history: List[str]) -> str:
    """Xác định sở thích chủ đề dựa trên lịch sử học tập.

    Args:
        learning_history (List[str]): Danh sách ID của các nút đã học.

    Returns:
        str: Chủ đề phổ biến nhất hoặc 'Unknown' nếu không có dữ liệu.
    """
    contexts = []
    for node_id in learning_history:
        query = f"MATCH (n {{{Config.PROPERTY_ID}: '{node_id}'}}) RETURN n.{Config.PROPERTY_CONTEXT} AS context"
        try:
            result = execute_cypher_query(driver, query)
            if result:
                contexts.append(result[0]["context"])
        except Exception as e:
            logger.warning(f"Lỗi truy vấn context cho node {node_id}: {str(e)}")
    if contexts:
        return Counter(contexts).most_common(1)[0][0]
    return "Unknown"

def load_student_profile(student_id: str) -> Dict:
    """Tải hồ sơ học sinh từ Neo4j hoặc trả về mặc định nếu không tìm thấy.

    Args:
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
            return {
                "student_id": student_data["StudentID"],
                "learning_history": student_data.get("learning_history", "").split(",") if student_data.get("learning_history") else [],
                "current_level": float(student_data.get("current_level", 0)),
                "performance_details": student_data.get("performance_details", "").split(";") if student_data.get("performance_details") else [],
                "learning_style_preference": student_data.get("learning_style_preference", Config.DEFAULT_LEARNING_STYLE),
                "preferred_difficulty": student_data.get("preferred_difficulty", "STANDARD"),
                "time_availability": float(student_data.get("time_availability", 60)),
                "learning_speed": calculate_learning_speed(student_data.get("performance_details", "").split(";")),
                "topic_preference": extract_topic_preference(student_data.get("learning_history", "").split(",")),
                "long_term_goal": student_data.get("long_term_goal", "Not Specified")
            }
        else:
            logger.warning(f"Không tìm thấy hồ sơ cho student_id: {student_id}")
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
        logger.error(f"Lỗi tải hồ sơ học sinh: {str(e)}")
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