"""
Module này chứa các hàm logic chính để tạo lộ trình học tập cá nhân hóa.
Bao gồm việc xác định điểm bắt đầu/kết thúc, chạy thuật toán tìm đường A*,
và các logic phụ trợ để điều chỉnh và gợi ý lộ trình.
"""
import logging
import heapq
import json
import re
from typing import List, Dict, Any

# Import các thư viện và module cần thiết
from neo4j import GraphDatabase
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

from .config import Config
from .data_loader import load_student_profile, execute_cypher_query, jaccard_similarity

# Thiết lập logger
logger = logging.getLogger(__name__)

# ==============================================================================
# CÁC HÀM XÁC ĐỊNH ĐIỂM BẮT ĐẦU VÀ ĐÍCH
# ==============================================================================
import re
# Hàm chuẩn hóa tag thành snake_case
def to_snake_case(tag):
    # Loại bỏ khoảng trắng thừa
    tag = tag.strip()
    # Chuyển thành lowercase
    tag = tag.lower()
    # Thay khoảng trắng, dấu gạch nối, hoặc ký tự đặc biệt bằng '_'
    tag = re.sub(r'[\s-]+', '_', tag)
    # Loại bỏ các ký tự không phải chữ, số, hoặc '_'
    tag = re.sub(r'[^a-z0-9_]', '', tag)
    # Loại bỏ nhiều dấu '_' liên tiếp
    tag = re.sub(r'_+', '_', tag)
    # Loại bỏ dấu '_' ở đầu hoặc cuối
    tag = tag.strip('_')
    return tag if tag else 'unknown'  # Trả về 'unknown' nếu tag rỗng

def generate_goal_tags(student_goal, context, llm, num=30):
    """
    Sử dụng LLM để tạo danh sách các tag liên quan đến student_goal và context.

    Args:
        student_goal (str): Mục tiêu học tập của học sinh.
        context (str): Ngữ cảnh bổ sung (nếu có).
        llm: Đối tượng LLM từ llama_index (ví dụ: Gemini) để tạo nội dung.

    Returns:
        list: Danh sách các tag dưới dạng chuỗi.
    """
    prompt = (
        f"Dựa trên mục tiêu học tập của học sinh: '{student_goal}' "
        f"và ngữ cảnh: '{context if context else 'không có'}', "
        "hãy tạo một danh sách các tag liên quan đến mục tiêu này. "
        "Tag nên là các từ khóa hoặc cụm từ ngắn gọn mô tả các khái niệm hoặc chủ đề chính, "
        "phù hợp để tìm kiếm các nội dung học tập liên quan. "
        "Hãy tập trung vào việc tạo ra các tag liên quan nhất và hữu ích nhất, "
        "bao gồm:\n\n"
        "- **Core Tags**: Từ đồng nghĩa và ý nghĩa cốt lõi của mục tiêu học tập.\n"
        "- **Contextual Tags**: Các khái niệm và ứng dụng liên quan đến mục tiêu và ngữ cảnh.\n"
        "- **Extended Tags**: Các liên kết lý thuyết và liên ngành rộng hơn.\n\n"
        "Nếu cần thiết, hãy sử dụng ontologies giáo dục như WordNet hoặc DBpedia để bổ sung ý nghĩa cho danh sách tag. "
        "Hãy ưu tiên các tag cụ thể, có tính ứng dụng cao và liên quan chặt chẽ đến mục tiêu và ngữ cảnh, "
        "đồng thời loại bỏ các tag quá chung chung, không cần thiết hoặc không liên quan.\n\n"
        "**Lưu ý quan trọng**:\n"
        "- Mỗi tag phải là một phần tử riêng biệt, ngắn gọn và dễ hiểu (ví dụ: 'business_intelligence', 'data_analysis').\n"
        "- Không ghép nhiều tag thành một chuỗi dài.\n"
        "- Kiểm tra và loại bỏ các tag trùng lặp hoặc không cần thiết.\n\n"
        "Cuối cùng, hãy sắp xếp các tag theo thứ tự alphabet và trả về dưới dạng danh sách phân cách bởi dấu chấm phẩy (;). "
        "Ví dụ: 'business_intelligence; data_analysis; machine_learning'."
    )

    # Gọi LLM để tạo phản hồi
    response = llm.complete(prompt).text

    # Chuyển đổi phản hồi thành danh sách tag
    tags = [to_snake_case(tag.strip()) for tag in response.split(";")]
    return tags


# Hàm cập nhật heuristic dựa trên kết quả bài kiểm tra
def update_heuristic_based_on_assessment(assessment_history: Dict) -> None:
    """Cập nhật trọng số heuristic dựa trên kết quả bài kiểm tra.

    Args:
        assessment_history (Dict): Lịch sử bài kiểm tra với avg_score.
    """
    try:
        if assessment_history["avg_score"] < 50:
            Config.ASTAR_HEURISTIC_WEIGHTS["difficulty_standard"] += 0.05
            logger.info("Tăng trọng số difficulty_standard do điểm trung bình thấp")
    except Exception as e:
        logger.error(f"Lỗi trong update_heuristic_based_on_assessment: {str(e)}")




def determine_start_node(student_id: str, level: str, context: str, student_goal: str = None, student_file: str = "students.csv", skip_quiz: bool = False) -> Dict:
    """Xác định điểm bắt đầu cho học sinh dựa trên trình độ, ngữ cảnh và mục tiêu học tập.

    Args:
        student_id (str): ID của học sinh.
        level (str): Trình độ hiện tại (e.g., 'Remember', 'Understand').
        context (str): Ngữ cảnh học tập (e.g., 'e_learning').
        student_goal (str, optional): Mục tiêu học tập (e.g., 'sql basics').
        student_file (str): Đường dẫn đến file hồ sơ học sinh.
        skip_quiz (bool): Bỏ qua bài kiểm tra nếu True.

    Returns:
        Dict: {'start_node': str, 'status': str, 'error_message': str or None}
    """
    global vector_index, driver, Settings, Config
    global_start_node = None

    try:
        # Validate inputs
        if not student_goal and not context:
            raise ValueError("Phải cung cấp ít nhất một trong student_goal hoặc context")
        query_txt = f"Mục tiêu học tập: {student_goal or context}"
        if level not in Config.ASTAR_SKILL_LEVELS_LOW + Config.ASTAR_SKILL_LEVELS_HIGH:
            raise ValueError(f"Trình độ không hợp lệ: {level}")

        # Load student profile
        profile = load_student_profile(student_id)
        current_level = profile.get("current_level", 0)
        learned_nodes = profile.get("learning_history", [])

        # Logic mới cho skill_levels
        if current_level <= Config.ASTAR_CURRENT_LEVEL_THRESHOLD:
            skill_levels = Config.ASTAR_SKILL_LEVELS_LOW
        else:
            # Lấy tất cả cấp độ từ LOW và HIGH
            all_levels = Config.ASTAR_SKILL_LEVELS_LOW + Config.ASTAR_SKILL_LEVELS_HIGH
            skill_weights = Config.ASTAR_HEURISTIC_WEIGHTS["skill_level"]
            level_weight = skill_weights.get(level, skill_weights["default"])
            # Lọc các cấp độ có trọng số nhỏ hơn hoặc bằng level đầu vào
            skill_levels = [
                skill for skill in all_levels
                if skill_weights.get(skill, skill_weights["default"]) <= level_weight
            ]

        # Ensure skill_levels is defined
        if not skill_levels:
            logger.error("Skill levels not defined. Check Config.ASTAR_SKILL_LEVELS_LOW and Config.ASTAR_SKILL_LEVELS_HIGH.")
            return {"status": "error", "error_message": "Skill levels not defined"}

        # Step 1: Tìm kiếm ngữ nghĩa với VectorStoreIndex
        potential_nodes = []
        if vector_index:
            query_engine = vector_index.as_query_engine(similarity_top_k=20)
            query_txt = f"Mục tiêu học tập: {student_goal or context}. Trình độ: {level}. Ngữ cảnh: {context or 'không có'}"
            response = query_engine.query(query_txt)
            potential_nodes = [
                doc.metadata.get("node_id")
                for doc in response.source_nodes
                if doc.metadata.get("node_id") and doc.metadata.get("node_id") != "Không có ID"
                and doc.metadata.get("skill_level") in skill_levels
            ]
            logger.info(f"Potential nodes from vector index: {potential_nodes}")
        else:
            logger.warning("Vector index not initialized.")

        # Step 2: Truy vấn đồ thị với Cypher query trực tiếp
        candidates = []
        goal_tags = generate_goal_tags(student_goal=student_goal, context=context, llm=Settings.llm, num=30)
        logger.info(f"goal_tags: {goal_tags}")
        goal_tags_str = ', '.join([f"'{tag}'" for tag in goal_tags])
        logger.info(f"goal_tags_str: {goal_tags_str}")

        # Xây dựng truy vấn Cypher linh hoạt dựa trên potential_nodes
        if potential_nodes:
            node_list = ', '.join([f"'{node}'" for node in potential_nodes])
            cypher_query = f"""
            MATCH (n:KnowledgeNode)
            WHERE n.Node_ID IN [{node_list}]
            AND (
                ANY(tag IN {goal_tags} WHERE tag IN split(n.Semantic_Tags, ';'))
                OR ANY(tag IN {goal_tags} WHERE tag IN split(n.Sanitized_Concept, ';'))
                OR n.Learning_Objective CONTAINS '{student_goal or ''}'
            )
            AND n.Skill_Level IN {skill_levels}
            """
        else:
            cypher_query = f"""
            MATCH (n:KnowledgeNode)
            WHERE (
                ANY(tag IN {goal_tags} WHERE tag IN split(n.Semantic_Tags, ';'))
                OR ANY(tag IN {goal_tags} WHERE tag IN split(n.Sanitized_Concept, ';'))
                OR n.Learning_Objective CONTAINS '{student_goal or ''}'
            )
            AND n.Skill_Level IN {skill_levels}
            """

        # Thêm điều kiện context nếu context không rỗng
        if context:
            cypher_query += f" AND n.Context = '{context}'"

        cypher_query += """
        AND EXISTS ((n)-[:IS_PREREQUISITE_OF|IS_SUBCONCEPT_OF|NEXT]->())
        RETURN n.Node_ID AS id,
               n.Sanitized_Concept AS sanitized_concept,
               n.Learning_Objective AS learning_objective,
               n.Skill_Level AS skill_level,
               n.Priority AS priority,
               n.Time_Estimate AS time_estimate,
               n.Difficulty AS difficulty,
               n.Semantic_Tags AS semantic_tags
        """

        # Thực hiện truy vấn Cypher
        result = execute_cypher_query(driver=driver, query=cypher_query)
        logger.info(f"result: {result}")
        candidates = [record for record in result]
        logger.info(f"candidates_1: {candidates}")


        #Step 3: Nếu không có candidates, lấy nút đi vào
        if not candidates:
            logger.info("No candidates with outgoing relationships. Querying incoming nodes.")
            incoming_query = f"""
            MATCH (n:KnowledgeNode)<-[:IS_PREREQUISITE_OF|IS_SUBCONCEPT_OF|NEXT]-(incoming:KnowledgeNode)
            WHERE n.Skill_Level IN $skill_levels
            AND EXISTS ((incoming)-[:IS_PREREQUISITE_OF|IS_SUBCONCEPT_OF|NEXT]->())
            """
            if context:
                incoming_query += f" AND n.Context = '{context}'"
            incoming_query += """
            RETURN incoming.Node_ID AS id,
                   incoming.Sanitized_Concept AS sanitized_concept,
                   incoming.Learning_Objective AS learning_objective,
                   incoming.Skill_Level AS skill_level,
                   incoming.Priority AS priority,
                   incoming.Time_Estimate AS time_estimate,
                   incoming.Difficulty AS difficulty,
                   incoming.Semantic_Tags AS semantic_tags
            """
            result = execute_cypher_query(driver=driver, query=incoming_query, params={"skill_levels": skill_levels})
            candidates = [record for record in result]


        # Step 4: Fallback nếu không có candidates
        if not candidates:
            # Fallback 1: Lấy nodes theo level và context (nếu có)
            query = f"""
            MATCH (n:KnowledgeNode)
            WHERE n.Skill_Level IN {skill_levels}
            AND EXISTS ((n)-[:IS_PREREQUISITE_OF|IS_SUBCONCEPT_OF|NEXT]->())
            """
            if context:
                query += f" AND n.Context = '{context}'"
            query += """
            RETURN n.Node_ID AS id,
                  n.Sanitized_Concept AS sanitized_concept,
                  n.Learning_Objective AS learning_objective,
                  n.Skill_Level AS skill_level,
                  n.Priority AS priority,
                  n.Time_Estimate AS time_estimate,
                  n.Difficulty AS difficulty,
                  n.Semantic_Tags AS semantic_tags
            ORDER BY n.Priority ASC
            """
            candidates = execute_cypher_query(driver, query, params={"level": level})
            if candidates:
                logger.info(f"candidates_in_Fallback_1: {candidates}")
            if not candidates:
                # Fallback 2: Lấy nodes theo context (nếu có)
                query = """
                MATCH (n:KnowledgeNode)
                WHERE EXISTS ((n)-[:IS_PREREQUISITE_OF|IS_SUBCONCEPT_OF|NEXT]->())
                """
                if context:
                    query += f" AND n.Context = '{context}'"
                query += f"""
                RETURN n.Node_ID AS id,
                      n.Sanitized_Concept AS sanitized_concept,
                      n.Learning_Objective AS learning_objective,
                      n.Skill_Level AS skill_level,
                      n.Priority AS priority,
                      n.Time_Estimate AS time_estimate,
                      n.Difficulty AS difficulty,
                      n.Semantic_Tags AS semantic_tags
                WHERE n.Skill_Level IN {skill_levels}
                """
                # ORDER BY n.Priority ASC
                # LIMIT 10
                candidates = execute_cypher_query(driver, query)
                if candidates:
                    logger.info(f"candidates_in_Fallback_2: {candidates}")

        # Step 4: Lọc bổ sung với Jaccard similarity
        if candidates:
            similarities = []
            for node in candidates:
                semantic_tags = node.get("semantic_tags", "").split(";")
                similarity = jaccard_similarity(goal_tags, semantic_tags)
                node["similarity"] = similarity
                similarities.append(similarity)

            threshold = 0.7
            filtered_candidates = [node for node in candidates if node["similarity"] >= threshold]

            if not filtered_candidates:
                threshold_DYM = max(min(max(similarities) * 0.8, 0.7), 0.1) if similarities else 0.3
                filtered_candidates = [node for node in candidates if node["similarity"] >= threshold_DYM]

                if not filtered_candidates:
                    logger.error("Không có nút nào trong ngưỡng. Chọn top 5 nút có similarity cao nhất")
                    filtered_candidates = sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:5]
            logger.info(f"similarities: {similarities}")

        else:
            # Fallback chung nếu không có candidates
            query = f"""
            MATCH (n:KnowledgeNode)
            WHERE EXISTS ((n)-[:IS_PREREQUISITE_OF|IS_SUBCONCEPT_OF|NEXT]->())
            AND n.Skill_Level IN {skill_levels}
            """
            if context:
                query += f" AND n.Context = '{context}'"
            else:
                query += " AND true"

            query += """
            RETURN n.{id_prop} AS id,
                  n.{concept_prop} AS sanitized_concept,
                  n.{objective_prop} AS learning_objective,
                  n.{skill_prop} AS skill_level,
                  n.{priority_prop} AS priority,
                  n.{time_prop} AS time_estimate,
                  n.{difficulty_prop} AS difficulty,
                  n.{tags_prop} AS semantic_tags
            ORDER BY n.{priority_prop} ASC
            LIMIT 10
            """.format(
                id_prop=Config.PROPERTY_ID,
                concept_prop=Config.PROPERTY_SANITIZED_CONCEPT,
                objective_prop=Config.PROPERTY_LEARNING_OBJECTIVE,
                skill_prop=Config.PROPERTY_SKILL_LEVEL,
                priority_prop=Config.PROPERTY_PRIORITY,
                time_prop=Config.PROPERTY_TIME_ESTIMATE,
                difficulty_prop=Config.PROPERTY_DIFFICULTY,
                tags_prop=Config.PROPERTY_SEMANTIC_TAGS,
                skill_levels=skill_levels
            )
            filtered_candidates = execute_cypher_query(driver, query)

        filtered_nodes = [node for node in filtered_candidates if node["id"] not in learned_nodes]
        candidates = filtered_nodes

        if not candidates:
            logger.info("No candidates found after filtering learned nodes. Using priority-based fallback.")
            query = f"""
            MATCH (n:KnowledgeNode)
            WHERE n.Skill_Level IN {skill_levels}
            AND EXISTS ((n)-[:IS_PREREQUISITE_OF|IS_SUBCONCEPT_OF|NEXT]->())
            """
            if context:
                query += f" AND n.Context = '{context}'"
            query += """
            RETURN n.Node_ID AS id,
                  n.Sanitized_Concept AS sanitized_concept,
                  n.Learning_Objective AS learning_objective,
                  n.Skill_Level AS skill_level,
                  n.Priority AS priority,
                  n.Time_Estimate AS time_estimate,
                  n.Difficulty AS difficulty,
                  n.Semantic_Tags AS semantic_tags
            ORDER BY n.Priority ASC
            LIMIT 5
            """
            candidates = execute_cypher_query(driver, query, params={"level": level})
            candidates = [node for node in candidates if node["id"] not in learned_nodes]
            if not candidates:
                logger.error("No candidate nodes found after all fallbacks")
                return {"status": "error", "error_message": "Không tìm thấy nút ứng viên nào"}

        # Skip quiz if requested
        if skip_quiz:
            best_start_node = candidates[0]["id"]
			      # Kiểm tra sự tồn tại của nút
            check_query = f"MATCH (n {{{Config.PROPERTY_ID}: $node_id}}) RETURN n"
            result = execute_cypher_query(driver, check_query, params={"node_id": best_start_node})
            if not result:
                logger.error(f"Nút '{best_start_node}' không tồn tại trong đồ thị")
                return {"status": "error", "error_message": f"Nút '{best_start_node}' không tồn tại trong đồ thị"}
            return {"start_node": best_start_node, "status": "success", "error_message": None}

        # Generate quiz
        prompt = PromptTemplate(f"""
        Based on: {candidates}
        Generate exactly 15 multiple-choice questions (4 options each) for {student_goal} in context {context}.
        Each question should be in the following JSON format:
        {{
            "text": "Question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "correct_index": 0-3,
            "related_concept": "Sanitized concept or concept",
            "difficulty": "STANDARD|ADVANCED"
        }}
        Ensure questions cover a broad range of concepts and vary in difficulty 80% STANDARD, 20% ADVANCED.
        Return only the JSON list of questions.
        """)
        candidate_str = "\n".join([f"{c['id']}: {c['learning_objective']}" for c in candidates])
        quiz_json = Settings.llm.complete(prompt.format(candidates=candidate_str, goal=student_goal or "general knowledge", context=context or "general"))
        questions = json.loads(clean_json_response(quiz_json.text))
        if len(questions) != 15:
            questions = questions[:15] if len(questions) > 15 else questions + [{"text": "Placeholder", "options": ["A", "B", "C", "D"], "correct_index": 0, "related_concept": "general", "difficulty": "STANDARD"}] * (15 - len(questions))

        # Display quiz with options
        print('Hãy thực hiện bài kiểm tra đầu vào để xác định lộ trình học tập (Chọn 1 trong 4 đáp án A B C D). . .')
        answers = []
        for i, q in enumerate(questions):
            print(f"\nCâu {i+1} ({q['difficulty']}): {q['text']}")
            for idx, option in enumerate(q['options']):
                print(f"{chr(65 + idx)}. {option}") # Display as A, B, C, D
            while True:
                try:
                    answer = input("Chọn đáp án (A B C D): ").strip().upper()
                    if answer in ['A', 'B', 'C', 'D']:
                        answer_idx = ord(answer) - 65
                        break
                    print("Đầu vào không hợp lệ: Vui lòng chọn A, B, C hoặc D.")
                except KeyboardInterrupt:
                    print("Đầu vào bị gián đoạn. Chọn đáp án mặc định: A.")
                    answer_idx = 0
                    break
            answers.append(answer_idx)


        # Calculate score
        assessment_score = sum(1 for i, ans in enumerate(answers) if ans == questions[i]["correct_index"]) / 15 * 100
        logger.info(f"assessment_score: {assessment_score}")

        # Select best start node
        node_weights = {q["related_concept"]: ({"STANDARD": 1.0, "ADVANCED": 2.0}[q["difficulty"]] if answers[i] != q["correct_index"] else 0) for i, q in enumerate(questions)}
        bloom_levels = Config.ASTAR_SKILL_LEVELS_LOW + Config.ASTAR_SKILL_LEVELS_HIGH
        student_level_idx = bloom_levels.index(level)

        combined_score = Config.ASTAR_CURRENT_LEVEL_WEIGHT * current_level + Config.ASTAR_ASSESSMENT_WEIGHT * assessment_score
        logger.info(f"combined_score: {combined_score}")
        node_candidates = [
            {
                "id": c["id"],
                "combined": abs((Config.ASTAR_SKILL_LEVEL_SCORES.get(c["skill_level"], Config.ASTAR_SKILL_LEVEL_SCORES["default"])/6 * 25
                                + (c.get("priority") or 0)/5 * 10)
                                - Config.ASTAR_CURRENT_LEVEL_WEIGHT * current_level
                                - Config.ASTAR_ASSESSMENT_WEIGHT * assessment_score),
                "min_gap": abs(bloom_levels.index(skill_level) - student_level_idx) if (skill_level := (c.get("skill_level") if c.get("skill_level") in bloom_levels else "Understand")) in bloom_levels else 0,
                "priority": c.get("priority") or 0,
                "time_estimate": c.get("time_estimate") or 30,
                "difficulty": c.get("difficulty", "STANDARD")
            }
            for c in candidates
        ]

        logger.info(f"node_candidates: {node_candidates}")

        if not node_candidates:
            logger.error("No node candidates generated")
            if candidates:
                best_start_node = candidates[0]["id"]
            else:
                return {"status": "error", "error_message": "Không có nút nào để chọn làm điểm bắt đầu"}
        else:
            node_candidates.sort(key=lambda x: (x["combined"], x["min_gap"], -x["priority"], x["time_estimate"]))
            best_start_node = node_candidates[0]["id"]
            logger.info(f"Best start node selected: {best_start_node}")

        # Kiểm tra sự tồn tại của best_start_node
        check_query = f"MATCH (n {{{Config.PROPERTY_ID}: $node_id}}) RETURN n"
        result = execute_cypher_query(driver, check_query, params={"node_id": best_start_node})
        if not result:
            logger.error(f"Nút '{best_start_node}' không tồn tại trong đồ thị")
            return {"status": "error", "error_message": f"Nút '{best_start_node}' không tồn tại trong đồ thị"}

        # Lưu kết quả bài kiểm tra để sử dụng cho feedback loop
        assessment_history = {"avg_score": assessment_score}
        update_heuristic_based_on_assessment(assessment_history)
        logger.info(f"Assessment history saved: {assessment_history}")

        global_start_node = best_start_node
        return {"start_node": best_start_node, "status": "success", "error_message": None}

    except Exception as e:
        logger.error(f"Error in determine_start_node: {str(e)}")
        return {"start_node": None, "status": "error", "error_message": str(e)}

#  Áp dụng thuật toán A* để chọn nút đích tối ưu
def get_heuristic(node_id, goal_similarities, driver):
    similarity = goal_similarities.get(node_id, 0.0)
    node_info_query = f"""
    MATCH (n {{{Config.PROPERTY_ID}: $node_id}})
    RETURN n.{Config.PROPERTY_DIFFICULTY} AS difficulty, n.{Config.PROPERTY_SKILL_LEVEL} AS skill_level,
          n.{Config.PROPERTY_PRIORITY} AS priority, n.{Config.PROPERTY_TIME_ESTIMATE} AS time_estimate
    """
    node_info = execute_cypher_query(driver, node_info_query, params={"node_id": node_id})
    if not node_info:
        return float('inf')
    difficulty_score = Config.ASTAR_HEURISTIC_WEIGHTS["difficulty_standard"] if node_info[0]["difficulty"] == "STANDARD" else Config.ASTAR_HEURISTIC_WEIGHTS["difficulty_advanced"]
    skill_level_score = Config.ASTAR_HEURISTIC_WEIGHTS["skill_level"].get(node_info[0]["skill_level"], Config.ASTAR_HEURISTIC_WEIGHTS["skill_level"]["default"])
    priority_score = node_info[0].get("priority", 0) * Config.ASTAR_HEURISTIC_WEIGHTS["priority"]
    time_estimate_score = node_info[0].get("time_estimate", 0) * Config.ASTAR_HEURISTIC_WEIGHTS["time_estimate"]
    h_score = (0.4 * (1 - similarity)) + (0.2 * difficulty_score / 3.0) + (0.2 * skill_level_score / 6.0) + (0.1 * priority_score / 5.0) + (0.1 * time_estimate_score)
    return h_score


def a_star_search(start: str, potential_goals: List[tuple], driver) -> str:
    open_set = [(0, 0, start, [start])]
    closed_set = set()
    goal_nodes = {goal[0]["id"] for goal in potential_goals}
    goal_similarities = {goal[0]["id"]: goal[1] for goal in potential_goals}

    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)
        if current in closed_set:
            continue
        closed_set.add(current)
        if current in goal_nodes:
            return current
        neighbors = execute_cypher_query(
            driver,
            f"""
            MATCH (current:KnowledgeNode {{{Config.PROPERTY_ID}: $current}})
            -[r:{Config.RELATIONSHIP_NEXT}|{Config.RELATIONSHIP_REQUIRES}|{Config.RELATIONSHIP_IS_PREREQUISITE_OF}|{Config.RELATIONSHIP_HAS_ALTERNATIVE_PATH}|{Config.RELATIONSHIP_SIMILAR_TO}]->(neighbor:KnowledgeNode)
            RETURN neighbor.{Config.PROPERTY_ID} AS id, r.Weight AS weight
            """,
            params={"current": current}
        )
        for record in neighbors:
            neighbor = record["id"]
            weight = record.get("weight", 1.0)
            if neighbor in closed_set:
                continue
            new_g_score = g_score + weight
            h_score = get_heuristic(neighbor, goal_similarities, driver)
            f_score = new_g_score + h_score
            heapq.heappush(open_set, (f_score, new_g_score, neighbor, path + [neighbor]))

    # Fallback: Kiểm tra từng goal_node trong potential_goals
    for goal, _ in potential_goals:
        check_query = f"MATCH (n {{{Config.PROPERTY_ID}: $node_id}}) RETURN n"
        if execute_cypher_query(driver, check_query, params={"node_id": goal["id"]}):
            return goal["id"]
    raise ValueError("Không tìm thấy goal_node hợp lệ trong potential_goals")


def expand_tags(tags: List[str]) -> List[str]:
    """Mở rộng danh sách tag với từ đồng nghĩa từ WordNet và chuẩn hóa thành lowercase với dấu '_'.

    Args:
        tags (List[str]): Danh sách tag ngữ nghĩa.

    Returns:
        List[str]: Danh sách tag đã được mở rộng và chuẩn hóa.
    """
    
    # Hàm mở rộng tag ngữ nghĩa
    from nltk.corpus import wordnet
    import nltk
    nltk.download('wordnet')

    try:
        expanded = set()
        for tag in tags:
            # Chuẩn hóa tag ban đầu thành lowercase và giữ dấu '_'
            normalized_tag = tag.lower()
            expanded.add(normalized_tag)

            # Tách các từ trong tag nếu có dấu '_'
            words = normalized_tag.split('_')

            for word in words:
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        # Chuẩn hóa lemma thành lowercase và thay khoảng trắng bằng '_'
                        lemma_name = lemma.name().lower().replace(' ', '_')
                        expanded.add(lemma_name)
        return list(expanded)
    except Exception as e:
        logger.error(f"Lỗi trong expand_tags: {str(e)}")
        return tags
    


def determine_goal_node(student_id: str, context: str, student_goal: str = None, start_node: str = None, student_file: str = Config.STUDENT_FILE) -> Dict:
    """Xác định điểm đích (goal node) dựa trên mục tiêu học tập, điểm bắt đầu và hồ sơ học sinh.

    Args:
        student_id (str): ID của học sinh.
        context (str): Ngữ cảnh học tập (e.g., 'math', 'programming').
        student_goal (str, optional): Mục tiêu học tập cụ thể của học sinh.
        start_node (str): ID của nút khởi đầu.
        student_file (str): Đường dẫn đến file hồ sơ học sinh.

    Returns:
        Dict: Dictionary chứa 'goal_node', 'status', và 'error_message'.
    """
    try:
        # Bước 1: Kiểm tra đầu vào
        if not student_id or not isinstance(student_id, str):
            raise ValueError("student_id phải là chuỗi không rỗng")
        if not start_node or not isinstance(start_node, str):
            raise ValueError("Nút khởi đầu phải là chuỗi không rỗng")

        # Bước 2: Tải hồ sơ học sinh
        profile = load_student_profile(student_id)
        if not profile:
            logger.warning(f"Không tìm thấy hồ sơ cho student_id '{student_id}'. Sử dụng giá trị mặc định.")
            current_level = 0
            learning_style_preference = Config.DEFAULT_LEARNING_STYLE
            skill_level = "Understand"
        else:
            current_level = profile.get("current_level", 0)
            learning_style_preference = profile.get("learning_style_preference", Config.DEFAULT_LEARNING_STYLE)
            skill_level = profile.get("skill_level", "Understand")

        # Bước 3: Xác định độ khó dựa trên trình độ hiện tại
        difficulty = Config.ASTAR_DIFFICULTY_FILTER[0] if current_level < Config.ASTAR_CURRENT_LEVEL_THRESHOLD else Config.ASTAR_DIFFICULTY_FILTER[1]

        # Bước 4: Truy vấn các nút liên quan từ điểm bắt đầu
        query = f"""
        MATCH (start {{{Config.PROPERTY_ID}: $start_node}})-[:{Config.RELATIONSHIP_NEXT}|{Config.RELATIONSHIP_REQUIRES}|{Config.RELATIONSHIP_IS_PREREQUISITE_OF}|{Config.RELATIONSHIP_HAS_ALTERNATIVE_PATH}|{Config.RELATIONSHIP_SIMILAR_TO}|{Config.RELATIONSHIP_IS_SUBCONCEPT_OF}*0..]->(related)
        WHERE related.{Config.PROPERTY_DIFFICULTY} = $difficulty
        AND NOT (related)-[:{Config.RELATIONSHIP_NEXT}|{Config.RELATIONSHIP_IS_PREREQUISITE_OF}|{Config.RELATIONSHIP_IS_SUBCONCEPT_OF}]->()
        AND NOT ()-[:{Config.RELATIONSHIP_REQUIRES}]->(related)
        RETURN related.{Config.PROPERTY_ID} AS id, related.{Config.PROPERTY_SANITIZED_CONCEPT} AS sanitized_concept,
              related.{Config.PROPERTY_LEARNING_OBJECTIVE} AS learning_objective, related.{Config.PROPERTY_SKILL_LEVEL} AS skill_level,
              related.{Config.PROPERTY_DIFFICULTY} AS difficulty, related.{Config.PROPERTY_SEMANTIC_TAGS} AS semantic_tags,
              related.{Config.PROPERTY_LEARNING_STYLE_PREFERENCE} AS learning_style
        """
        related_nodes = execute_cypher_query(driver, query, params={"start_node": start_node, "difficulty": difficulty})

        if not related_nodes:
            logger.warning("Không tìm thấy nút liên quan. Tìm kiếm trong toàn bộ đồ thị.")
            query = f"""
            MATCH (start {{{Config.PROPERTY_ID}: $start_node}})-[:{Config.RELATIONSHIP_NEXT}|{Config.RELATIONSHIP_REQUIRES}|{Config.RELATIONSHIP_IS_PREREQUISITE_OF}|{Config.RELATIONSHIP_HAS_ALTERNATIVE_PATH}|{Config.RELATIONSHIP_SIMILAR_TO}|{Config.RELATIONSHIP_IS_SUBCONCEPT_OF}*0..]->(related)
            WHERE NOT EXISTS {{
                MATCH (related)-[:{Config.RELATIONSHIP_NEXT}|{Config.RELATIONSHIP_IS_PREREQUISITE_OF}]->()
            }}
            RETURN related.{Config.PROPERTY_ID} AS id, related.{Config.PROPERTY_SANITIZED_CONCEPT} AS sanitized_concept,
                  related.{Config.PROPERTY_LEARNING_OBJECTIVE} AS learning_objective, related.{Config.PROPERTY_SKILL_LEVEL} AS skill_level,
                  related.{Config.PROPERTY_DIFFICULTY} AS difficulty, related.{Config.PROPERTY_SEMANTIC_TAGS} AS semantic_tags,
                  related.{Config.PROPERTY_LEARNING_STYLE_PREFERENCE} AS learning_style
            ORDER BY related.{Config.PROPERTY_PRIORITY} DESC
            LIMIT 10
            """
            related_nodes = execute_cypher_query(driver, query, params={"start_node": start_node, "difficulty": difficulty})

        if not related_nodes:
            raise ValueError("Không có nút nào trong đồ thị để chọn làm nút đích")

        # Bước 5: Tính độ tương đồng với mục tiêu học tập
        goal_tags = generate_goal_tags(student_goal=student_goal, context=context, llm=Settings.llm, num=30)
        bloom_levels = Config.ASTAR_SKILL_LEVELS_LOW + Config.ASTAR_SKILL_LEVELS_HIGH
        student_level_idx = bloom_levels.index(skill_level)
        potential_goals = []

        for node in related_nodes:
            node_level_idx = bloom_levels.index(node.get("skill_level", "Understand") if node.get("skill_level") in bloom_levels else "Understand")
            if abs(node_level_idx - student_level_idx) > 2:
                continue
            # Xử lý semantic_tags nếu không tồn tại
            semantic_tags_str = node.get("semantic_tags", "")
            semantic_tags = semantic_tags_str.split(";") if isinstance(semantic_tags_str, str) and semantic_tags_str else []
            expanded_tags = expand_tags(semantic_tags)
            similarity = jaccard_similarity(goal_tags, expanded_tags) if goal_tags else 0.0
            if similarity >= 0.3 and learning_style_preference in node.get("learning_style", Config.DEFAULT_LEARNING_STYLE):
                potential_goals.append((node, similarity))

        # Nếu không tìm thấy goal dựa trên similarity, dùng LLM để lọc từ related_nodes
        if not potential_goals and student_goal:
            logger.warning("Không tìm thấy goal dựa trên similarity. Sử dụng LLM để lọc từ related_nodes.")
            if not related_nodes:
                raise ValueError("Không có related_nodes để lọc")

            # Chuẩn bị dữ liệu cho LLM
            node_descriptions = [
                f"ID: {node['id']}, Khái niệm: {node['sanitized_concept']}, Mục tiêu học tập: {node['learning_objective']}, Thẻ ngữ nghĩa: {node['semantic_tags']}"
                for node in related_nodes
            ]
            prompt = f"Dựa trên mục tiêu học tập: '{student_goal}', hãy chọn nút phù hợp nhất từ danh sách sau:\n" + "\n".join(node_descriptions)
            llm_response = Settings.llm.complete(prompt).text.strip()

            # LM trả về ID của nút phù hợp nhất
            best_node_id = llm_response.split()[0]
            for node in related_nodes:
                if node["id"] == best_node_id:
                    potential_goals.append((node, 1.0))  # Gán similarity cao
                    logger.info(f"Bước 5: Tính độ tương đồng - Node {best_node_id} được chọn bởi LLM với similarity=1.0")
                    break

        potential_goals = sorted(potential_goals, key=lambda x: x[1], reverse=True)[:Config.GOAL_NODE_SUGGESTION_LIMIT]

        if not potential_goals:
            logger.warning("Không có nút nào đạt ngưỡng tương đồng. Chọn nút mặc định.")
            sorted_nodes = sorted(related_nodes, key=lambda x: x.get("priority", 0), reverse=True)

            if sorted_nodes:
                default_goal_node = sorted_nodes[0]["id"]
                logger.info(f"Bước 5: Tính độ tương đồng - Chọn default goal_node: {default_goal_node}")
                return {"goal_node": default_goal_node, "status": "success", "error_message": None}
            else:
                raise ValueError("Không có nút nào để chọn làm nút đích")

        logger.info("Bước 6: Tìm goal node tối ưu - Bắt đầu A* search")
        best_goal = a_star_search(start_node, potential_goals, driver)
        # Kiểm tra sự tồn tại của best_goal
        check_query = f"MATCH (n {{{Config.PROPERTY_ID}: $best_goal}}) RETURN n"
        if not execute_cypher_query(driver, check_query, params={"best_goal": best_goal}):
            raise ValueError(f"Goal node '{best_goal}' không tồn tại trong đồ thị")

        logger.info(f"Đã chọn goal_node: {best_goal}")
        return {"goal_node": best_goal, "status": "success", "error_message": None}

    except Exception as e:
        logger.error(f"Lỗi trong determine_goal_node: {str(e)}")
        return {"goal_node": None, "status": "error", "error_message": str(e)}

# ==============================================================================
# THUẬT TOÁN TÌM ĐƯỜNG A* (A-STAR)
# ==============================================================================

def heuristic(node_id: str, driver, goal_tags: List[str], dynamic_weights: Dict, Config, context: str, heuristic_cache: Dict = None) -> float:
    if heuristic_cache is None:
        heuristic_cache = {}

    if node_id in heuristic_cache:
        return heuristic_cache[node_id]

    # Truy vấn Cypher với coalesce để xử lý giá trị thiếu
    query = f"""
    MATCH (n {{{Config.PROPERTY_ID}: $node_id}})
    RETURN coalesce(n.similarity, 0.0) AS similarity,
           coalesce(n.{Config.PROPERTY_PRIORITY}, 0.0) AS priority,
           coalesce(n.{Config.PROPERTY_DIFFICULTY}, 'STANDARD') AS difficulty,
           coalesce(n.{Config.PROPERTY_SKILL_LEVEL}, 'Understand') AS skill_level,
           coalesce(n.{Config.PROPERTY_TIME_ESTIMATE}, 30.0) AS time_estimate,
           coalesce(n.{Config.PROPERTY_CONTEXT}, '') AS context
    """
    result = execute_cypher_query(driver, query, params={"node_id": node_id})

    if not result:
        heuristic_cache[node_id] = 0.0
        return 0.0

    node = result[0]

    # Đảm bảo các giá trị là số hoặc chuỗi hợp lệ
    similarity = float(node.get("similarity", 0.0))
    priority = float(node.get("priority", 0.0))
    difficulty = node.get("difficulty", "STANDARD")
    skill_level = node.get("skill_level", "Understand")
    time_estimate = float(node.get("time_estimate", 30.0))
    context_match = 0.5 if node.get("context", "") == context else 0.0

    logger.debug(f"Node {node_id}: similarity={similarity}, priority={priority}, time_estimate={time_estimate}")

    # Kiểm tra dynamic_weights và gán giá trị mặc định nếu không hợp lệ
    if not isinstance(dynamic_weights, dict):
        print(f"dynamic_weights không phải dictionary cho node {node_id}, sử dụng mặc định")
        dynamic_weights = {
            'difficulty_standard': 0.0,
            'difficulty_advanced': 0.0,
            'skill_level': {'default': 0.0},
            'time_estimate': 0.0
        }

    # Gán giá trị với mặc định nếu thiếu
    difficulty_weight = float(dynamic_weights.get('difficulty_standard', 0.0) if difficulty == Config.ASTAR_DIFFICULTY_FILTER[0] else dynamic_weights.get('difficulty_advanced', 0.0))
    skill_level_weights = dynamic_weights.get('skill_level', {})
    if not isinstance(skill_level_weights, dict):
        skill_level_weights = {'default': 0.0}
    skill_weight = float(skill_level_weights.get(skill_level, skill_level_weights.get('default', 0.0)))
    weight_time_estimate = float(dynamic_weights.get('time_estimate', 0.0))

    # Log để debug
    print(f"Node {node_id}: similarity={similarity}, priority={priority}, difficulty={difficulty}, "
          f"skill_level={skill_level}, time_estimate={time_estimate}, context_match={context_match}")
    print(f"Dynamic weights: difficulty_weight={difficulty_weight}, skill_weight={skill_weight}, "
          f"weight_time_estimate={weight_time_estimate}")

    # Tính heuristic
    h = (0.4 * (1 - similarity) +
         priority * 0.1 / 5.0 +
         difficulty_weight * 0.2 / 3.0 +
         skill_weight * 0.2 / 6.0 +
         time_estimate * Config.ASTAR_HEURISTIC_WEIGHTS["time_estimate"] * weight_time_estimate -
         context_match)

    heuristic_cache[node_id] = h
    return h


def calculate_dynamic_weights(student_id: str = None) -> Dict:
    """Compute dynamic heuristic weights based on student profile.

    Currently this function focuses on making the 'time_estimate' weight truly dynamic.
    It reads the student's performance details (which encode time_spent per node in the
    format 'node_id:score:time_spent') and computes an average time-per-node. The
    returned `dynamic_weights['time_estimate']` is a scalar that will be multiplied by
    Config.ASTAR_HEURISTIC_WEIGHTS["time_estimate"] inside `heuristic()`.

    Behavior:
    - If no student_id or no timing history is available, returns sensible defaults
      (time_estimate scalar = 1.0).
    - The scalar is computed as 60 / avg_time_spent (so avg_time=60 -> 1.0,
      avg_time=30 -> 2.0, avg_time=120 -> 0.5) and then clamped to [0.2, 3.0].

    This keeps the time term interpretable while allowing it to increase when a
    student typically spends less time (so we prefer short nodes) and decrease when
    they have more time available.
    """
    # Base dynamic weights copy (only keys used by heuristic)
    dynamic = {
        'difficulty_standard': float(Config.ASTAR_HEURISTIC_WEIGHTS.get('difficulty_standard', 0.0)),
        'difficulty_advanced': float(Config.ASTAR_HEURISTIC_WEIGHTS.get('difficulty_advanced', 0.0)),
        'skill_level': dict(Config.ASTAR_HEURISTIC_WEIGHTS.get('skill_level', {'default': 0.0})),
        'time_estimate': 1.0
    }

    try:
        if not student_id:
            return dynamic

        profile = load_student_profile(student_id)
        if not profile:
            return dynamic

        perf = profile.get('performance_details', []) or []
        times = []
        for entry in perf:
            # Expecting format: "node_id:score:time_spent"
            try:
                parts = str(entry).split(":")
                if len(parts) >= 3:
                    t = float(parts[2])
                    if t > 0:
                        times.append(t)
            except Exception:
                continue

        if times:
            avg_time = sum(times) / len(times)
            # Build scalar such that avg_time=60 -> 1.0, avg_time small -> >1, avg_time large -> <1
            scalar = 60.0 / max(1.0, avg_time)
            # Clamp to avoid extreme influence
            scalar = max(0.2, min(3.0, scalar))
            dynamic['time_estimate'] = float(scalar)

    except Exception as e:
        logger.warning(f"calculate_dynamic_weights failed for student_id={student_id}: {e}")

    return dynamic


def a_star_custom(start_node: str, goal_node: str, driver, goal_tags: List[str], dynamic_weights: Dict, Config, context: str) -> List[str]:
    heuristic_cache = {}  # Bộ nhớ đệm heuristic
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, driver, goal_tags, dynamic_weights, Config, context, heuristic_cache)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            return path[::-1]

        query = f"""
        MATCH (n {{{Config.PROPERTY_ID}: $node_id}})-[r]->(neighbor)
        RETURN neighbor.{Config.PROPERTY_ID} AS id, r.Weight AS weight
        """
        neighbors = execute_cypher_query(driver, query, params={"node_id": current})

        for neighbor in neighbors:
            tentative_g_score = g_score[current] + neighbor["weight"]
            neighbor_id = neighbor["id"]
            if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                came_from[neighbor_id] = current
                g_score[neighbor_id] = tentative_g_score
                f_score[neighbor_id] = tentative_g_score + heuristic(neighbor_id, driver, goal_tags, dynamic_weights, Config, context, heuristic_cache)
                heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

    return None




# ==============================================================================
# HÀM CHÍNH TẠO LỘ TRÌNH
# ==============================================================================

def generate_learning_path(student_id: str, level: str, start_node: str = None, context: str = None,
                          learning_style: str = None, student_goal: str = None,
                          student_file: str = Config.STUDENT_FILE, use_llm: bool = False) -> Dict:
    """Tạo lộ trình học tập từ điểm bắt đầu đến điểm đích với ngữ cảnh là tiêu chí ưu tiên phụ."""
    global global_start_node
    try:
        # Kiểm tra đầu vào
        if not student_id or not isinstance(student_id, str):
            raise ValueError("student_id phải là chuỗi không rỗng")
        if student_goal and not isinstance(student_goal, str):
            raise ValueError("student_goal phải là chuỗi không rỗng")

        # Xác định điểm bắt đầu
        if start_node is None:
            if global_start_node is None:
                start_node_result = determine_start_node(student_id, level, context, student_goal, student_file, skip_quiz=True)
                if start_node_result["status"] != "success":
                    raise ValueError(f"Lỗi xác định điểm bắt đầu: {start_node_result['error_message']}")
                start_node = start_node_result["start_node"]
            else:
                start_node = global_start_node

        # Kiểm tra sự tồn tại của start_node
        query = f"""
        MATCH (n {{{Config.PROPERTY_ID}: $node_id}})
        RETURN n.{Config.PROPERTY_ID} AS id
        """
        result_start = execute_cypher_query(driver, query, params={"node_id": start_node})
        if not result_start:
            raise ValueError(f"Nút khởi đầu '{start_node}' không tồn tại trong Knowledge Graph.")

        # Xác định điểm đích
        goal_node_result = determine_goal_node(student_id, context, student_goal, start_node, student_file)
        if goal_node_result["status"] != "success" or goal_node_result["goal_node"] is None:
            raise ValueError(f"Lỗi xác định điểm đích: {goal_node_result['error_message']}")
        goal_node = goal_node_result["goal_node"]

        result_goal = execute_cypher_query(driver, query, params={"node_id": goal_node})
        if not result_goal:
            raise ValueError(f"Nút đích '{goal_node}' không tồn tại trong Knowledge Graph")

        # Trường hợp start_node == goal_node
        if start_node == goal_node:
            logger.warning("start_node và goal_node trùng nhau, tìm các nút liên quan.")
            query = f"""
            MATCH (n {{{Config.PROPERTY_ID}: $node_id}})
            OPTIONAL MATCH (n)-[:{Config.RELATIONSHIP_NEXT}|{Config.RELATIONSHIP_REQUIRES}]->(related)
            RETURN n.{Config.PROPERTY_ID} AS id,
                   n.{Config.PROPERTY_SANITIZED_CONCEPT} AS sanitized_concept,
                   n.{Config.PROPERTY_CONTEXT} AS context,
                   n.{Config.PROPERTY_DEFINITION} AS definition,
                   n.{Config.PROPERTY_EXAMPLE} AS example,
                   n.{Config.PROPERTY_LEARNING_OBJECTIVE} AS learning_objective,
                   n.{Config.PROPERTY_SKILL_LEVEL} AS skill_level,
                   n.{Config.PROPERTY_DIFFICULTY} AS difficulty,
                   n.{Config.PROPERTY_TIME_ESTIMATE} AS time_estimate,
                   related.{Config.PROPERTY_ID} AS related_id,
                   related.{Config.PROPERTY_SANITIZED_CONCEPT} AS related_concept,
                   related.{Config.PROPERTY_CONTEXT} AS related_context,
                   related.{Config.PROPERTY_DEFINITION} AS related_definition,
                   related.{Config.PROPERTY_EXAMPLE} AS related_example,
                   related.{Config.PROPERTY_LEARNING_OBJECTIVE} AS related_objective,
                   related.{Config.PROPERTY_SKILL_LEVEL} AS related_skill_level,
                   related.{Config.PROPERTY_DIFFICULTY} AS related_difficulty,
                   related.{Config.PROPERTY_TIME_ESTIMATE} AS related_time_estimate
            LIMIT 5
            """
            result = execute_cypher_query(driver, query, params={"node_id": start_node})
            path = []
            step = 1

            if result:
                node = result[0]
                path.append({
                    "step": step,
                    "node_id": node["id"],
                    "sanitized_concept": node["sanitized_concept"] or "Unknown Concept",
                    "context": node["context"] or context or "general",
                    "definition": node["definition"] or "Không có định nghĩa",
                    "example": node["example"] or "Không có ví dụ",
                    "learning_objective": node["learning_objective"] or f"Học {node['sanitized_concept'] or 'Unknown Concept'}",
                    "skill_level": node["skill_level"] or level,
                    "difficulty": node["difficulty"] or "STANDARD",
                    "time_estimate": node["time_estimate"] or 30
                })
                step += 1

                for record in result:
                    if record["related_id"]:
                        path.append({
                            "step": step,
                            "node_id": record["related_id"],
                            "sanitized_concept": record["related_concept"] or "Unknown Concept",
                            "context": record["related_context"] or context or "general",
                            "definition": record["related_definition"] or "Không có định nghĩa",
                            "example": record["related_example"] or "Không có ví dụ",
                            "learning_objective": record["related_objective"] or f"Học {record['related_concept'] or 'Unknown Concept'}",
                            "skill_level": record["related_skill_level"] or level,
                            "difficulty": record["related_difficulty"] or "STANDARD",
                            "time_estimate": record["related_time_estimate"] or 30
                        })
                        step += 1

            if not path:
                logger.error("Không tìm thấy nút liên quan nào để xây dựng lộ trình")
                return {"path": [], "status": "error", "error_message": "Không tìm thấy nút liên quan nào để xây dựng lộ trình"}
            return {"path": path, "status": "success", "error_message": None}

        logger.warning(f"startnode trong generate learning path: {start_node}")
        logger.warning(f"goalnode trong generate learning path: {goal_node}")

        # Tải hồ sơ học sinh và thiết lập tham số
        profile = load_student_profile(student_id)
        skill_levels = Config.ASTAR_SKILL_LEVELS_LOW if profile.get("current_level", 0.0) < Config.ASTAR_CURRENT_LEVEL_THRESHOLD else Config.ASTAR_SKILL_LEVELS_HIGH
        learning_style_filter = f" AND n.{Config.PROPERTY_LEARNING_STYLE_PREFERENCE} CONTAINS $learning_style_preference" if profile.get("learning_style_preference") else ""

    # Tính trọng số động (dựa trên hồ sơ học sinh nếu có)
    dynamic_weights = calculate_dynamic_weights(student_id)

        # Kiểm tra sự tồn tại của đường dẫn từ start_node đến goal_node
        path_check_query = f"""
        MATCH path = (start:KnowledgeNode {{{Config.PROPERTY_ID}: $start_id}})-[:NEXT|REQUIRES|IS_PREREQUISITE_OF|HAS_ALTERNATIVE_PATH|SIMILAR_TO|IS_SUBCONCEPT_OF*1..10]->(end:KnowledgeNode {{{Config.PROPERTY_ID}: $end_id}})
        RETURN path LIMIT 1
        """
        path_check_result = execute_cypher_query(driver, path_check_query, params={"start_id": start_node, "end_id": goal_node})

        if not path_check_result:
            logger.warning(f"Không có đường dẫn từ {start_node} đến {goal_node}. Fallback sang sử dụng LLM.")
            use_llm = True

        # Tạo lộ trình học tập
        if use_llm:
            context_filter = f"WHERE ALL(n IN nodes(path) WHERE n.Context = '{context}' OR '{context}' IS NULL)" if context else ""
            cypher_template = PromptTemplate(
                f"""
                #TASK:
                Generate a valid Cypher query to find a learning path from node with Node_ID '{start_node}' to node with Node_ID '{goal_node}' in a Neo4j graph database.
                #Instructions:
                The path should use relationships of types NEXT, REQUIRES, IS_PREREQUISITE_OF, HAS_ALTERNATIVE_PATH, SIMILAR_TO, or IS_SUBCONCEPT_OF, with a maximum path length of 10.
                Include an optional context filter for nodes with Context = '{context}' if context is provided.
                Return only the Cypher query string, without any explanation or additional text.
                Example:
                MATCH path = (start:KnowledgeNode {{Node_ID: '{start_node}'}})-[:NEXT|REQUIRES|IS_PREREQUISITE_OF|HAS_ALTERNATIVE_PATH|SIMILAR_TO|IS_SUBCONCEPT_OF*1..10]->(end:KnowledgeNode {{Node_ID: '{goal_node}'}})
                {context_filter}
                RETURN [node IN nodes(path) | node.Node_ID] AS path
                """
            )
            cypher_retriever = TextToCypherRetriever(
                graph_store=kg_index.property_graph_store,
                llm=Settings.llm,
                text_to_cypher_template=cypher_template,
                response_template="Generated Cypher query:\n(query)\n\nCypher Response:\n{response}",
                cypher_validator=None,
                allowed_output_field=["text", "label", "type"]
            )
            query_engine = kg_index.as_query_engine(sub_retrievers=[cypher_retriever], response_synthesizer=TreeSummarize())
            llm_response = query_engine.query(
                f"Create learning path from `{start_node}` to `{goal_node}` with optional context `{context}`"
            )
            logger.info(f"LLM Cypher response: {llm_response}")

            path = []
            if hasattr(llm_response, 'response'):
                cypher_query = extract_cypher_query(str(llm_response.response))
                if cypher_query:
                    try:
                        result = execute_cypher_query(driver, cypher_query)
                        if result and 'path' in result[0]:
                            path = result[0]['path']
                        else:
                            logger.warning("Truy vấn Cypher không trả về lộ trình hợp lệ.")
                    except Exception as e:
                        logger.warning(f"Thực thi truy vấn Cypher thất bại: {str(e)}")
                else:
                    logger.warning("Không trích xuất được truy vấn Cypher hợp lệ từ phản hồi LLM.")
            else:
                logger.warning("LLM response không hợp lệ (không có thuộc tính response).")

            if not path:
                logger.info("Cypher không tạo được lộ trình. Sử dụng LLM để tự sinh lộ trình học tập.")
                fallback_prompt = PromptTemplate(
                    f"""
                    Dựa trên điểm bắt đầu '{start_node}' và mục tiêu học tập '{student_goal}',
                    hãy tạo một lộ trình học tập *bằng tiếng Việt* gồm 3 bước tiến bộ từ cơ bản đến nâng cao, theo định dạng sau:
                    - step: số thứ tự bước (bắt đầu từ 1)
                    - node_id: mã định danh nút (tự tạo, ví dụ: generated_node_1, generated_node_2, ...)
                    - sanitized_concept: tên khái niệm
                    - context: ngữ cảnh (nếu không có thì dùng '{context}' hoặc 'general')
                    - definition: định nghĩa ngắn gọn
                    - example: ví dụ minh họa
                    - learning_objective: mục tiêu học tập của bước
                    - skill_level: mức độ kỹ năng (dùng '{level}' cho bước 1, tăng dần như 'Understand', 'Apply' cho các bước tiếp theo)
                    - difficulty: độ khó (STANDARD cho bước 1, tăng dần như INTERMEDIATE, ADVANCED)
                    - time_estimate: thời gian ước tính (số phút, mặc định 30, tăng dần như 45, 60 cho các bước tiếp theo)
                    Trả về danh sách 3 bước dưới dạng JSON, không giải thích thêm.
                    """
                )
                query_engine_fallback = kg_index.as_query_engine(response_synthesizer=TreeSummarize())
                llm_fallback_response = query_engine_fallback.query(
                    fallback_prompt.format(start_node=start_node, student_goal=student_goal or "general learning", level=level, context=context or "general")
                )
                logger.info(f"LLM fallback response: {llm_fallback_response}")

                if hasattr(llm_fallback_response, 'response'):
                    response_text = str(llm_fallback_response.response)
                    try:
                        path_data = json.loads(response_text)
                        if not isinstance(path_data, list):
                            raise ValueError("Phản hồi LLM không đúng định dạng danh sách JSON")
                        learning_path = []
                        for idx, step in enumerate(path_data, 1):
                            if not isinstance(step, dict):
                                raise ValueError(f"Bước {idx} không phải dictionary")
                            standardized_step = {
                                "step": step.get("step", idx),
                                "node_id": step.get("node_id", f"generated_node_{idx}"),
                                "sanitized_concept": step.get("sanitized_concept", "Unknown Concept"),
                                "context": step.get("context", context or "general"),
                                "definition": step.get("definition", "Không có định nghĩa"),
                                "example": step.get("example", "Không có ví dụ"),
                                "learning_objective": step.get("learning_objective", f"Học {step.get('sanitized_concept', 'Unknown Concept')}"),
                                "skill_level": step.get("skill_level", level),
                                "difficulty": step.get("difficulty", "STANDARD"),
                                "time_estimate": step.get("time_estimate", 30)
                            }
                            learning_path.append(standardized_step)
                    except Exception as e:
                        logger.warning(f"LLM fallback thất bại: {str(e)}. Trả về lộ trình mặc định với 1 bước.")
                        learning_path = [
                            {
                                "step": 1,
                                "node_id": start_node,
                                "sanitized_concept": "Not Available",
                                "context": context or "general",
                                "definition": "Tổ chức dữ liệu dựa trên một tiêu chí chung.",
                                "example": "Not Available",
                                "learning_objective": f"Học tri thức liên quan đến {start_node} và để đạt được mục tiêu {student_goal or 'general learning'}.",
                                "skill_level": level,
                                "difficulty": "STANDARD",
                                "time_estimate": 30
                            }
                        ]
                else:
                    logger.warning("LLM fallback response không hợp lệ (không có thuộc tính response).")
                    learning_path = [
                        {
                            "step": 1,
                            "node_id": start_node,
                            "sanitized_concept": "Unknown Concept",
                            "context": context or "general",
                            "definition": "Không có định nghĩa",
                            "example": "Không có ví dụ",
                            "learning_objective": f"Học {start_node}",
                            "skill_level": level,
                            "difficulty": "STANDARD",
                            "time_estimate": 30
                        }
                    ]
                return {"path": learning_path, "status": "success", "error_message": None}

        else:
            goal_tags = get_goal_tags(driver, goal_node)
            calculate_and_store_similarity(driver, goal_tags)
            path = a_star_custom(start_node, goal_node, driver, goal_tags, dynamic_weights, Config, context)
            if not path:
                return {"path": [], "status": "error", "error_message": "Không có lộ trình"}

        # Xây dựng lộ trình học tập
        learning_path = []
        for idx, node_id in enumerate(path):
            query = f"""
            MATCH (n {{{Config.PROPERTY_ID}: $node_id}})
            RETURN properties(n) AS props
            """
            result = execute_cypher_query(driver, query, params={"node_id": node_id})
            if result:
                node_props = result[0]["props"]
                standardized_step_data = {key.lower().replace(" ", "_"): value for key, value in node_props.items()}
                standardized_step_data["step"] = idx + 1
                learning_path.append(standardized_step_data)

        if not learning_path:
            logger.warning("Không tìm thấy lộ trình học tập hợp lệ")
            return {"path": [], "status": "error", "error_message": "Không tìm thấy lộ trình học tập hợp lệ"}

        return {"path": learning_path, "status": "success", "error_message": None}

    except Exception as e:
        logger.error(f"Lỗi trong generate_learning_path: {str(e)}")
        return {"path": [], "status": "error", "error_message": str(e)}

# ==============================================================================
# CÁC HÀM CẬP NHẬT VÀ GỢI Ý
# ==============================================================================

def suggest_prerequisites(goal_node: str, student_id: str, context: str = None) -> Dict:
    """Gợi ý các kiến thức tiên quyết cần học trước khi tiếp cận mục tiêu.

    Args:
        goal_node (str): ID của nút mục tiêu.
        student_id (str): ID của học sinh.
        context (str, optional): Ngữ cảnh học tập.

    Returns:
        Dict: Dictionary chứa 'prerequisites' (danh sách kiến thức tiên quyết), 'status', và 'error_message'.
    """
    try:
        # Bước 1: Kiểm tra đầu vào
        if not isinstance(goal_node, str) or not goal_node:
            raise ValueError("goal_node phải là chuỗi không rỗng")
        if not isinstance(student_id, str) or not student_id:
            raise ValueError("student_id phải là chuỗi không rỗng")

        # Bước 2: Truy vấn các nút tiên quyết chưa được học sinh hoàn thành
        query = f"""
        MATCH (goal {{{Config.PROPERTY_ID}: $goal_node}})<-[:{Config.RELATIONSHIP_IS_PREREQUISITE_OF}]-(prereq)
        WHERE NOT EXISTS {{ MATCH (s:Student {{{Config.PROPERTY_ID}: $student_id}})-[:MASTERED]->(prereq) }}
        RETURN prereq.{Config.PROPERTY_ID} AS id, prereq.{Config.PROPERTY_SANITIZED_CONCEPT} AS sanitized_concept,
              prereq.{Config.PROPERTY_CONTEXT} AS context, prereq.{Config.PROPERTY_PRIORITY} AS priority,
              prereq.{Config.PROPERTY_DIFFICULTY} AS difficulty, prereq.{Config.PROPERTY_SKILL_LEVEL} AS skill_level,
              prereq.{Config.PROPERTY_TIME_ESTIMATE} AS time_estimate
        UNION
        MATCH (goal {{{Config.PROPERTY_ID}: $goal_node}})-[:{Config.RELATIONSHIP_REQUIRES}]->(prereq)
        WHERE NOT EXISTS {{ MATCH (s:Student {{{Config.PROPERTY_ID}: $student_id}})-[:MASTERED]->(prereq) }}
        RETURN prereq.{Config.PROPERTY_ID} AS id, prereq.{Config.PROPERTY_SANITIZED_CONCEPT} AS sanitized_concept,
              prereq.{Config.PROPERTY_CONTEXT} AS context, prereq.{Config.PROPERTY_PRIORITY} AS priority,
              prereq.{Config.PROPERTY_DIFFICULTY} AS difficulty, prereq.{Config.PROPERTY_SKILL_LEVEL} AS skill_level,
              prereq.{Config.PROPERTY_TIME_ESTIMATE} AS time_estimate
        """
        result = execute_cypher_query(driver, query, params={"goal_node": goal_node, "student_id": student_id})

        # Thêm: Nếu không có kết quả, lấy các nút IS_SUBCONCEPT_OF
        if not result:
            subconcept_query = f"""
            MATCH (goal {{{Config.PROPERTY_ID}: $goal_node}})<-[:{Config.RELATIONSHIP_IS_SUBCONCEPT_OF}]-(prereq)
            WHERE NOT EXISTS {{ MATCH (s:Student {{{Config.PROPERTY_ID}: $student_id}})-[:MASTERED]->(prereq) }}
            RETURN prereq.{Config.PROPERTY_ID} AS id, prereq.{Config.PROPERTY_SANITIZED_CONCEPT} AS sanitized_concept,
                  prereq.{Config.PROPERTY_CONTEXT} AS context, prereq.{Config.PROPERTY_PRIORITY} AS priority,
                  prereq.{Config.PROPERTY_DIFFICULTY} AS difficulty, prereq.{Config.PROPERTY_SKILL_LEVEL} AS skill_level,
                  prereq.{Config.PROPERTY_TIME_ESTIMATE} AS time_estimate
            """
            result = execute_cypher_query(driver, subconcept_query, params={"goal_node": goal_node, "student_id": student_id})

        # Bước 3: Sắp xếp kết quả
        prerequisites = []
        if result:
            if context:
                # Ưu tiên nút có ngữ cảnh phù hợp
                for row in result:
                    row["context_match"] = 1 if row["context"] == context else 0
                result.sort(key=lambda x: (-x["priority"], -x["context_match"]))
            else:
                result.sort(key=lambda x: -x["priority"])

            prerequisites = [
                {
                    "id": row["id"],
                    "sanitized_concept": row["sanitized_concept"],
                    "context": row["context"],
                    "priority": row["priority"],
                    "difficulty": row["difficulty"],
                    "skill_level": row["skill_level"],
                    "time_estimate": row["time_estimate"]
                }
                for row in result
                if all(key in row for key in ["id", "sanitized_concept", "context", "priority", "difficulty", "skill_level", "time_estimate"])
            ]

        # Bước 4: Ghi log và trả về
        if not prerequisites:
            logger.info(f"Không tìm thấy kiến thức tiên quyết cho goal node {goal_node}")
        else:
            logger.info(f"Tìm thấy {len(prerequisites)} kiến thức tiên quyết cho goal node {goal_node}")

        return {
            "prerequisites": prerequisites,
            "status": "success",
            "error_message": None
        }

    except Exception as e:
        logger.error(f"Lỗi trong suggest_prerequisites: {str(e)}")
        return {
            "prerequisites": [],
            "status": "error",
            "error_message": str(e)
        }

# Hàm cập nhật lộ trình học tập
def update_learning_path(student_id: str, node: Dict, current_path: List[Dict], score: float, max_attempts: int = 2) -> Dict:
    """Cập nhật lộ trình học tập dựa trên hiệu suất học sinh.

    Args:
        student_id (str): ID của học sinh.
        node (Dict): Thông tin nút hiện tại.
        current_path (List[Dict]): Lộ trình học tập hiện tại.
        score (float): Điểm bài kiểm tra (0-100).
        max_attempts (int): Số lần thử tối đa trước khi chuyển lộ trình thay thế.

    Returns:
        Dict: Dictionary chứa 'path', 'status', và 'error_message'.
    """
    try:
        # Bước 1: Kiểm tra đầu vào
        if not isinstance(student_id, str) or not student_id:
            raise ValueError("student_id phải là chuỗi không rỗng")
        if not isinstance(node, dict):
            raise ValueError("Nút phải là một dictionary")
        required_fields = [Config.PROPERTY_ID, "difficulty", "context"]
        missing_fields = [field for field in required_fields if field not in node]
        if missing_fields:
            raise ValueError(f"Nút thiếu các trường bắt buộc: {missing_fields}")
        if not isinstance(current_path, list):
            raise ValueError("Lộ trình hiện tại phải là một danh sách")
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            raise ValueError("Điểm phải là số từ 0 đến 100")

        # Bước 2: Kiểm tra nút có trong lộ trình không
        node_id = node.get(Config.PROPERTY_ID)
        if node_id is None:
            raise ValueError("Nút thiếu trường bắt buộc: Config.PROPERTY_ID")

        current_node = next((n for n in current_path if n.get(Config.PROPERTY_ID) == node_id), None)
        if current_node is None:
            logger.warning(f"Nút {node_id} không có trong lộ trình hiện tại. Không cập nhật.")
            return {
                "path": current_path,
                "status": "success",
                "error_message": None
            }

        # Bước 3: Tải hồ sơ và kiểm tra số lần thử
        profile = load_student_profile(student_id)
        if profile is None:
            raise ValueError(f"Không tìm thấy hồ sơ học sinh với ID: {student_id}")
        performance_details = profile.get("performance_details", [])
        node_attempts = sum(1 for detail in performance_details if detail.startswith(node_id))

        # Bước 4: Xử lý nếu điểm thấp hơn ngưỡng
        if score < Config.QUIZ_PASSING_SCORE:
            if score <= 10:
                with open(Config.TEACHER_NOTIFICATION_FILE, "a", encoding="utf-8") as f:
                        f.write(f"Học sinh {student_id} thất bại tại {node_id} sau {node_attempts} lần. Mất căn bản tri thức này. Cần chú ý.\n")
            if node_attempts >= max_attempts:
                # Tìm lộ trình thay thế
                alt_node = _find_alternative_node(node_id, node["difficulty"], node["context"])
                if alt_node:
                    with open(Config.TEACHER_NOTIFICATION_FILE, "a", encoding="utf-8") as f:
                        f.write(f"Học sinh {student_id} thất bại tại {node_id} sau {node_attempts} lần. Cần chú y! Hệ thống đã tự động chuyển sang lộ trình thay thế.\n")
                    logger.info(f"Chuyển sang lộ trình thay thế: {alt_node['id']}")
                    current_path[current_path.index(current_node)] = _construct_node(alt_node, current_node)

                else:
                    logger.warning(f"Không tìm thấy lộ trình thay thế cho {node_id}.")
                    with open(Config.TEACHER_NOTIFICATION_FILE, "a", encoding="utf-8") as f:
                        f.write(f"Học sinh {student_id} thất bại tại {node_id} sau {node_attempts} lần. Không có lộ trình thay thế.\n")
            else:
                # Tìm nút khắc phục
                remedy_node = _find_remedy_node(node_id, node["difficulty"], node["context"])
                if remedy_node:
                    logger.info(f"Thêm nút khắc phục: {remedy_node['id']}")
                    current_path.insert(current_path.index(current_node), _construct_node(remedy_node, current_node))
                else:
                    logger.warning(f"Không tìm thấy nút khắc phục cho {node_id}. Gửi thông báo.")
                    with open(Config.TEACHER_NOTIFICATION_FILE, "a", encoding="utf-8") as f:
                        f.write(f"Học sinh {student_id} cần nút khắc phục cho {node_id} sau điểm thấp.\n")

        return {"path": current_path, "status": "success", "error_message": None}

    except Exception as e:
        logger.error(f"Lỗi trong update_learning_path: {str(e)}")
        return {"path": current_path, "status": "error", "error_message": str(e)}


