import logging
import os
import csv
from typing import Dict, List

# Third-party libs used in this module
import pandas as pd
import numpy as np

# Local config and helper functions
from .config import Config
from .data_loader import execute_cypher_query, load_student_profile

logger = logging.getLogger(__name__)

def collaborative_filtering(student_id: str, learning_paths_file: str = Config.LEARNING_PATHS_FILE, learned_nodes: List[str] = [], learning_data_file: str = Config.LEARNING_DATA_FILE) -> Dict:
    """Đề xuất các nút dựa trên sự tương đồng giữa các học sinh.

    Args:
        student_id (str): ID của học sinh.
        learned_nodes (List[str]): Danh sách các nút đã học trong phiên hiện tại.
        learning_paths_file (str): Đường dẫn đến file lộ trình học tập.
        learning_data_file (str): Đường dẫn đến file dữ liệu học tập.

    Returns:
        Dict: Dictionary chứa 'recommended_nodes', 'status', và 'error_message'.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # Bước 1: Kiểm tra đầu vào
        if not isinstance(student_id, str) or not student_id:
            raise ValueError("student_id phải là chuỗi không rỗng")
        if not isinstance(learning_paths_file, str) or not isinstance(learning_data_file, str):
            logger.error(f"learning_paths_file hoặc learning_data_file không phải chuỗi: {learning_paths_file}, {learning_data_file}")
            raise ValueError("learning_paths_file và learning_data_file phải là chuỗi")
        if not learning_paths_file or not learning_data_file:
            raise ValueError("Đường dẫn file không được rỗng")

        # Bước 2: Kiểm tra file đầu vào
        if not os.path.exists(learning_paths_file) or not os.path.exists(learning_data_file):
            logger.warning(f"File lộ trình hoặc dữ liệu học tập không tồn tại: {learning_paths_file}, {learning_data_file}")
            return {
                "recommended_nodes": [],
                "status": "success",
                "error_message": "File dữ liệu không tồn tại"
            }

        # Bước 3: Đọc dữ liệu
        try:
            paths_df = pd.read_csv(learning_paths_file, encoding='utf-8', quoting=csv.QUOTE_ALL, error_bad_lines=False)
            data_df = pd.read_csv(learning_data_file, encoding='utf-8', quoting=csv.QUOTE_ALL, error_bad_lines=False)
        except pd.errors.ParserError as e:
            logger.error(f"Lỗi định dạng file CSV: {str(e)}")
            return {
                "recommended_nodes": [],
                "status": "error",
                "error_message": f"Lỗi định dạng file CSV: {str(e)}"
            }

        # Kiểm tra dữ liệu trống
        if paths_df.empty or data_df.empty:
            logger.warning("Một trong các file dữ liệu trống.")
            return {
                "recommended_nodes": [],
                "status": "success",
                "error_message": "Dữ liệu trống"
            }

        # Kiểm tra các cột cần thiết
        required_columns_paths = ["StudentID", "LearningPath"]
        required_columns_data = ["StudentID", "NodeID", "Score"]
        missing_columns_paths = [col for col in required_columns_paths if col not in paths_df.columns]
        missing_columns_data = [col for col in required_columns_data if col not in data_df.columns]
        if missing_columns_paths or missing_columns_data:
            logger.error(f"File CSV thiếu cột: {missing_columns_paths} trong paths_df, {missing_columns_data} trong data_df")
            return {
                "recommended_nodes": [],
                "status": "error",
                "error_message": f"File CSV thiếu cột: {missing_columns_paths}, {missing_columns_data}"
            }

        # Bước 4: Tạo bảng pivot
        pivot_table = data_df.pivot_table(index="StudentID", columns="NodeID", values="Score", fill_value=0)

        # Xử lý cold start
        if student_id not in pivot_table.index or len(pivot_table) <= 1:
            logger.warning(f"Học sinh {student_id} không có trong dữ liệu hoặc dữ liệu không đủ. Dùng đề xuất dựa trên nội dung.")
            profile = load_student_profile(student_id)
            context = profile.get("context", "")

            if context and context.strip():
                query = f"""
                MATCH (n)
                WHERE n.{Config.PROPERTY_CONTEXT} = $context
                AND n.{Config.PROPERTY_PRIORITY} IS NOT NULL
                RETURN n.{Config.PROPERTY_ID} AS id
                ORDER BY n.{Config.PROPERTY_PRIORITY} DESC
                LIMIT 5
                """
                params = {"context": context}
            else:
                query = f"""
                MATCH (n)
                WHERE n.{Config.PROPERTY_CONTEXT} IS NULL
                AND n.{Config.PROPERTY_PRIORITY} IS NOT NULL
                RETURN n.{Config.PROPERTY_ID} AS id
                ORDER BY n.{Config.PROPERTY_PRIORITY} DESC
                LIMIT 5
                """
                params = {}

            result = execute_cypher_query(driver, query, params=params)
            recommended_nodes = [row["id"] for row in result]
            return {"recommended_nodes": recommended_nodes, "status": "success", "error_message": None}

        # Bước 5: Tính ma trận tương đồng
        similarity_matrix = cosine_similarity(pivot_table)
        student_idx = pivot_table.index.get_loc(student_id)
        similarity_scores = similarity_matrix[student_idx]
        min_similarity_threshold = Config.MIN_SIMILARITY_THRESHOLD  # 0.5
        similar_students = [i for i in np.argsort(similarity_scores)[::-1] if similarity_scores[i] > min_similarity_threshold and i != student_idx][:4]

        # Fallback nếu không tìm thấy học sinh tương đồng
        if not similar_students:
            logger.warning(f"Không tìm thấy học sinh tương đồng cho {student_id}. Dùng top học sinh tương đồng nhất.")
            similar_students = [i for i in np.argsort(similarity_scores)[::-1] if i != student_idx][:4]

        # Bước 6: Đề xuất nút
        similar_student_ids = pivot_table.index[similar_students]
        similar_data = data_df[data_df["StudentID"].isin(similar_student_ids)]
        node_scores = similar_data.groupby("NodeID")["Score"].mean()

        # Lọc các nút đã học
        student_nodes = data_df[data_df["StudentID"] == student_id]["NodeID"].unique()
        all_learned_nodes = set(student_nodes).union(set(learned_nodes))
        recommended_nodes = node_scores[(node_scores > 80) & (~node_scores.index.isin(all_learned_nodes))].index.tolist()

        if not recommended_nodes:
            logger.info(f"Không tìm thấy nút phù hợp cho học sinh {student_id}")
            return {"recommended_nodes": [], "status": "success", "error_message": "Không tìm thấy nút phù hợp"}

        logger.info(f"Đề xuất {len(recommended_nodes)} nút cho học sinh {student_id}")
        return {
            "recommended_nodes": recommended_nodes,
            "status": "success",
            "error_message": None
        }

    except Exception as e:
        logger.error(f"Lỗi trong collaborative_filtering: {str(e)}")
        return {
            "recommended_nodes": [],
            "status": "error",
            "error_message": str(e)
        }


# Hàm khám phá quy tắc liên kết
def apply_apriori(learning_paths_file: str = Config.LEARNING_PATHS_FILE, min_support: float = 0.1, min_threshold: float = 0.5) -> Dict:
    """Khám phá các quy tắc liên kết trong lộ trình học tập bằng thuật toán Apriori.

    Args:
        learning_paths_file (str): Đường dẫn đến file lộ trình học tập.
        min_support (float): Ngưỡng hỗ trợ tối thiểu.
        min_threshold (float): Ngưỡng độ tin cậy tối thiểu.

    Returns:
        Dict: Dictionary chứa 'association_rules', 'status', và 'error_message'.
    """
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder


    try:
        # Bước 1: Kiểm tra file đầu vào
        if not os.path.exists(learning_paths_file):
            logger.warning(f"File lộ trình học tập {learning_paths_file} không tồn tại")
            return {
                "association_rules": {},
                "status": "success",
                "error_message": "File lộ trình không tồn tại"
            }

        # Bước 2: Đọc dữ liệu
        df = pd.read_csv(learning_paths_file)
        if df.empty:
            logger.warning("File lộ trình học tập rỗng")
            return {
                "association_rules": {},
                "status": "success",
                "error_message": "File lộ trình rỗng"
            }

        # Kiểm tra cột 'LearningPath'
        if "LearningPath" not in df.columns:
            logger.error("File CSV thiếu cột 'LearningPath'")
            return {
                "association_rules": {},
                "status": "error",
                "error_message": "File CSV thiếu cột 'LearningPath'"
            }

        # Chuyển đổi thành danh sách giao dịch, loại bỏ giá trị không hợp lệ
        transactions = [path.split(",") for path in df["LearningPath"] if isinstance(path, str) and path.strip() and len(path.split(",")) > 0]
        if not transactions:
            logger.warning("Không có giao dịch hợp lệ trong dữ liệu")
            return {
                "association_rules": {},
                "status": "success",
                "error_message": "Không có giao dịch hợp lệ"
            }

        # Bước 3: Chuyển đổi thành one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        one_hot = pd.DataFrame(te_ary, columns=te.columns_)


        # Bước 4: Áp dụng thuật toán Apriori
        frequent_itemsets = apriori(one_hot, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            logger.info("Không tìm thấy itemsets thường xuyên")
            return {
                "association_rules": {},
                "status": "success",
                "error_message": "Không tìm thấy itemsets thường xuyên"
            }
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
        if rules.empty:
            logger.info("Không tìm thấy quy tắc liên kết")
            return {
                "association_rules": {},
                "status": "success",
                "error_message": "Không tìm thấy quy tắc liên kết"
            }

        # Bước 5: Tạo dictionary quy tắc liên kết
        association_dict = {}
        for _, row in rules.iterrows():
            antecedent = list(row["antecedents"])
            consequent = list(row["consequents"])
            confidence = row["confidence"]

            for ant in antecedent:
                if ant not in association_dict:
                    association_dict[ant] = []
                association_dict[ant].append((consequent, confidence)) # Lưu toàn bộ consequents

        logger.info(f"Tìm thấy {len(association_dict)} quy tắc liên kết")
        return {
            "association_rules": association_dict,
            "status": "success",
            "error_message": None
        }

    except Exception as e:
        logger.error(f"Lỗi trong apply_apriori: {str(e)}")
        return {
            "association_rules": {},
            "status": "error",
            "error_message": str(e)
        }