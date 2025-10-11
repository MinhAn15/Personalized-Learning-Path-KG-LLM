import logging
import time
import csv
from typing import Dict, List, Optional, Any

from .config import Config
from .path_generator import (
    generate_learning_path,
    update_learning_path,
    suggest_prerequisites,
    determine_start_node,
    determine_goal_node,
)
from .content_generator import generate_learning_content, generate_quiz, evaluate_quiz

# Import data loader functions and wrap them to match session_manager expectations
from .data_loader import update_student_profile as dl_update_student_profile, save_learning_data as dl_save_learning_data

logger = logging.getLogger(__name__)



def get_time_spent(start_time=None, autosave=False):
    """Lấy thời gian học, hỗ trợ autosave hoặc nhập thủ công.

    Args:
        start_time (float, optional): Thời gian bắt đầu (epoch time).
        autosave (bool): Nếu True, tính thời gian tự động từ start_time.

    Returns:
        int: Thời gian học (phút), giới hạn từ 1 đến 1440.
    """
    import time

    if autosave:
        if start_time is None:
            print("Không có start_time. Chuyển sang nhập thủ công.")
        else:
            time_spent = int((time.time() - start_time) / 60)  # Chuyển từ giây sang phút
            print(f"Thời gian thực tế từ hệ thống: {time_spent} phút.")
            return max(1, min(time_spent, 1440))  # Giới hạn 1-1440 phút

    # Nhập thủ công nếu không dùng autosave hoặc start_time không hợp lệ
    while True:
        try:
            user_input = input("Nhập thời gian học (phút) hoặc nhấn Enter để chọn mặc định (30 phút): ").strip()
            if user_input == "":
                print("Đã chọn thời gian mặc định: 30 phút.")
                return int(30)
            time_spent = int(user_input)
            if 1 <= int(time_spent) <= 1440:
                print(f"Thời gian học được đặt: {time_spent} phút.")
                return time_spent
            else:
                print("Thời gian phải từ 1 đến 1440 phút. Vui lòng nhập lại.")
        except ValueError:
            print("Thời gian phải là số nguyên. Vui lòng nhập lại.")

def adjust_heuristic_weights(feedback_data: Dict) -> None:
    """Điều chỉnh trọng số heuristic dựa trên phản hồi học sinh.

    Args:
        feedback_data (Dict): Dữ liệu phản hồi với các khóa 'performance' và 'difficulty'.
    """
    try:
        # Ánh xạ performance và difficulty thành điểm số
        performance_map = {"low": -1, "medium": 0, "high": 1}
        difficulty_map = {"STANDARD": 1, "ADVANCED": 2}

        performance = feedback_data.get("performance", "medium")
        difficulty = feedback_data.get("difficulty", "STANDARD")

        performance_score = performance_map.get(performance, 0)
        difficulty_factor = difficulty_map.get(difficulty, 1)

        # Công thức điều chỉnh đơn giản
        adjustment = performance_score * 0.1 * difficulty_factor

        # Cập nhật trọng số heuristic
        if difficulty == "STANDARD":
            Config.ASTAR_HEURISTIC_WEIGHTS["difficulty_standard"] += adjustment
        elif difficulty == "ADVANCED":
            Config.ASTAR_HEURISTIC_WEIGHTS["difficulty_advanced"] += adjustment

        logger.info(f"Đã điều chỉnh trọng số heuristic: {adjustment} cho {difficulty}")
    except Exception as e:
        logger.error(f"Lỗi trong adjust_heuristic_weights: {str(e)}")

def run_learning_session(student_id: str, level: str, context: str, learning_style: str = None, student_goal: str = None, student_file: str = Config.STUDENT_FILE, use_llm: bool = False, driver: Optional[Any] = None) -> Dict:
    """Thực thi toàn bộ phiên học tập từ xác định lộ trình đến đánh giá và cập nhật.

    Args:
        student_id (str): ID của học sinh.
        level (str): Trình độ hiện tại (e.g., 'Remember', 'Understand').
        context (str): Ngữ cảnh học tập (e.g., 'math', 'programming').
        learning_style (str, optional): Phong cách học tập.
        student_goal (str, optional): Mục tiêu học tập.
        student_file (str): Đường dẫn đến file hồ sơ học sinh.

    Returns:
        Dict: Dictionary chứa 'status' và 'error_message'.
    """
    global global_start_node
    try:
        # Bước 1: Kiểm tra đầu vào
        if not isinstance(student_id, str) or not student_id:
            logger.warning("student_id phải là chuỗi không rỗng")
            raise ValueError("student_id phải là chuỗi không rỗng")

        # Bước 2: Xác định điểm bắt đầu và điểm đích
        start_node_result = determine_start_node(student_id, level, context, student_goal, student_file, skip_quiz=False)
        if start_node_result["status"] != "success":
            logger.warning(f"Lỗi xác định điểm bắt đầu: {start_node_result['error_message']}")
            raise ValueError(f"Lỗi xác định điểm bắt đầu: {start_node_result['error_message']}")
        start_node = start_node_result["start_node"]

        goal_node_result = determine_goal_node(student_id, context, student_goal, start_node, student_file)
        if goal_node_result["status"] != "success":
            logger.warning(f"Lỗi xác định điểm đích: {goal_node_result['error_message']}")
            raise ValueError(f"Lỗi xác định điểm đích: {goal_node_result['error_message']}")
        goal_node = goal_node_result["goal_node"]

        # Bước 3: Gợi ý kiến thức tiên quyết
        prereq_result = suggest_prerequisites(goal_node, student_id, context)
        if prereq_result["status"] != "success":
            logger.warning(f"Lỗi gợi ý kiến thức tiên quyết: {prereq_result['error_message']}")
        prerequisites = prereq_result["prerequisites"]

        # Bước 4: Tạo lộ trình học tập
        path_result = generate_learning_path(student_id=student_id, level=level,start_node=start_node, context=context, learning_style=learning_style, student_goal=student_goal, student_file=student_file, use_llm=use_llm)
        if path_result["status"] != "success":
            logger.warning(f"Lỗi tạo lộ trình học tập: {path_result['error_message']}")
            raise ValueError(f"Lỗi tạo lộ trình học tập: {path_result['error_message']}")
        learning_path = path_result["path"]

        if not learning_path:
            logger.warning("Không thể tạo lộ trình học tập")
            raise ValueError("Không thể tạo lộ trình học tập")

        # Bước 5: Thêm kiến thức tiên quyết vào lộ trình nếu có
        # if prerequisites:
        #     prereq_nodes = [
        #         {
        #             "step": i + 1,
        #             "node_id": prereq.get("id", "Unknown ID"),
        #             "sanitized_concept": prereq.get("sanitized_concept", "Unknown Concept"),
        #             "context": prereq.get("context", "Unknown Context"),
        #             "definition": prereq.get("definition", "Không có định nghĩa"),
        #             "example": prereq.get("example", "Không có ví dụ"),
        #             "learning_objective": f"Học {prereq.get('sanitized_concept', 'Unknown Concept')}",
        #             "skill_level": prereq.get("skill_level", "Remember"),
        #             "difficulty": prereq.get("difficulty", "STANDARD"),
        #             "time_estimate": prereq.get("time_estimate", 30)
        #         }
        #         for i, prereq in enumerate(prerequisites)
        #     ]
        #     learning_path = prereq_nodes + [
        #         {**node, "step": node["step"] + len(prereq_nodes)}
        #         for node in learning_path
        #     ]

        if prerequisites:
            prereq_nodes = [
                {
                    "step": i + 1,
                    "node_id": prereq.get("id", "Unknown ID"),
                    "sanitized_concept": prereq.get("sanitized_concept", "Unknown Concept"),
                    "context": prereq.get("context", "Unknown Context"),
                    "definition": prereq.get("definition", "Không có định nghĩa"),
                    "example": prereq.get("example", "Không có ví dụ"),
                    "learning_objective": f"Học {prereq.get('sanitized_concept', 'Unknown Concept')}",
                    "skill_level": prereq.get("skill_level", "Remember"),
                    "difficulty": prereq.get("difficulty", "STANDARD"),
                    "time_estimate": prereq.get("time_estimate", 30)
                }
                for i, prereq in enumerate(prerequisites)
            ]

            # Loại bỏ các nút trùng lặp trong prereq_nodes nếu đã có trong learning_path
            learning_path_ids = {node["node_id"] for node in learning_path}
            unique_prereq_nodes = [node for node in prereq_nodes if node["node_id"] not in learning_path_ids]

            # Kết hợp unique_prereq_nodes và learning_path
            learning_path = unique_prereq_nodes + [
                {**node, "step": node["step"] + len(unique_prereq_nodes)}
                for node in learning_path
            ]

        logger.info(f"Lộ trình học tập: {[node['node_id'] for node in learning_path]}")
        print(f"Lộ trình học tập: {[node['node_id'] for node in learning_path]}")

        # Bước 6: Thực thi phiên học tập
        current_step = 0
        learning_data = []

        # Kiểm tra số lượng bước trong lộ trình
        if len(learning_path) <= 1:
            logger.warning(f"Lộ trình học tập chỉ có {len(learning_path)} bước, không đủ để tiếp tục.")
            raise ValueError("Lộ trình học tập không đủ bước để tiếp tục.")

        while (current_step < len(learning_path)):
            node = learning_path[current_step]

            # Sinh nội dung học tập
            content_result = generate_learning_content(node, student_id, student_file, student_goal)
            if content_result["status"] != "success":
                logger.warning(f"Lỗi sinh nội dung cho nút {node['node_id']}: {content_result['error_message']}")
                current_step += 1
                continue
            print(f"\nBước {node['step']} - Nội dung học tập:\n{content_result['content']}")

            # Lưu thời gian bắt đầu cho nút hiện tại
            start_time = time.time()

            # Chờ người dùng học
            while True:
                if input('Bạn xác nhận đã học xong? (Y/N)') == 'Y':
                    break
                else:
                    time.sleep(0.3)
                    print('Cố lên! Tiếp tục học nhé!')
                    continue

            # Tạo bài kiểm tra
            quiz_result = generate_quiz(node)
            if quiz_result["status"] != "success":
                logger.warning(f"Lỗi tạo bài kiểm tra cho nút {node['node_id']}: {quiz_result['error_message']}")
                current_step += 1
                continue
            student_file = quiz_result["student_file"]
            answer_file = quiz_result["answer_file"]

            # Đọc câu hỏi từ student_file để hiển thị cho học sinh
            with open(student_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                questions = list(reader)

            # Thu thập câu trả lời học sinh
            student_answers = {}
            for q in questions:
                q_num = q["Question_Number"]
                print(f"\nCâu {q_num} ({q['Level']}): {q['Question_Text']}")
                print(f"A. {q['Option_1']}")
                print(f"B. {q['Option_2']}")
                print(f"C. {q['Option_3']}")
                print(f"D. {q['Option_4']}")
                while True:
                    try:
                        answer = input("Chọn đáp án (A B C D): ").strip().upper()
                        if answer in ['A', 'B', 'C', 'D']:
                            option_index = ord(answer) - ord('A')
                            student_answers[q_num] = q[f"Option_{option_index + 1}"]
                            break
                        print("Đầu vào không hợp lệ: Vui lòng chọn A, B, C hoặc D.")
                    except KeyboardInterrupt:
                        print("Đầu vào bị gián đoạn. Chọn đáp án mặc định: A.")
                        student_answers[q_num] = q["Option_1"]
                        break

            # Cập nhật student_file với câu trả lời của học sinh
            updated_questions = []
            for q in questions:
                q["Student_Answer"] = student_answers.get(q["Question_Number"], "")
                updated_questions.append(q)
            with open(student_file, "w", encoding="utf-8", newline='') as f:
                fieldnames = questions[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(updated_questions)

            # Đánh giá bài kiểm tra
            quiz_eval_result = evaluate_quiz(student_file, answer_file)
            if quiz_eval_result["status"] != "success":
                logger.warning(f"Lỗi đánh giá bài kiểm tra: {quiz_eval_result['error_message']}")
                current_step += 1
                continue
            score = quiz_eval_result["score"]
            print(f"Điểm của bạn: {score}%")

                # Hỏi người dùng về autosave
            autosave = input("Bạn có muốn autosave thời gian không? (yes/no): ").lower() == "yes"

            # Lấy thời gian học
            time_spent = get_time_spent(start_time=start_time if autosave else None, autosave=autosave)
            time_spent = time_spent if time_spent else 30

            # Cập nhật hồ sơ học sinh (sử dụng data_loader wrapper)
            try:
                updates = {
                    "learning_history": [node["node_id"]],
                    "performance_details": [f"{node['node_id']}:{score}:{time_spent}"]
                }
                if driver:
                    dl_update_student_profile(driver, student_id, updates)
                else:
                    # Call with driver=None to allow import-time checks; dl_update_student_profile expects a driver,
                    # so wrap call to avoid runtime error when driver is not provided.
                    dl_update_student_profile(None, student_id, updates)
            except Exception as e:
                logger.warning(f"Lỗi cập nhật hồ sơ: {str(e)}")

            # Lưu dữ liệu học tập
            learning_data.append({"node_id": node["node_id"], "score": score, "time_spent": time_spent})

            # Cập nhật lộ trình nếu cần
            if int(score) >= Config.QUIZ_PASSING_SCORE:
                current_step += 1
            else:
                while True:
                    if input('Bạn có muốn cập nhật lộ trình học? (Y/N)') == 'Y':
                        path_result = update_learning_path(student_id, node, learning_path, score)
                        if path_result["status"] != "success":
                            logger.warning(f"Lỗi cập nhật lộ trình: {path_result['error_message']}")
                        else:
                            learning_path = path_result["path"]
                        break
                    else:
                        print('Không đổi lộ trình...Cố lên! Tiếp tục học nhé!')
                        break

        # Learned_node for collaborative_filtering
        learned_nodes = [
            {"node_id": data["node_id"], "score": data["score"], "time_spent": data["time_spent"]}
            for data in learning_data if int(data["score"]) >= Config.QUIZ_PASSING_SCORE
        ]

        # Bước 6.2 Thu thập phản hồi từ học sinh
        performance = input("Đánh giá hiệu suất của bạn (low/medium/high): ").strip().lower()
        difficulty = input("Đánh giá độ khó của nội dung (STANDARD/ADVANCED): ").strip().upper()

        # Kiểm tra giá trị phản hồi hợp lệ
        valid_performance = ["low", "medium", "high"]
        valid_difficulty = ["STANDARD", "ADVANCED"]
        if performance not in valid_performance or difficulty not in valid_difficulty:
            logger.warning("Phản hồi không hợp lệ. Sử dụng giá trị mặc định.\n\
            performance = 'medium'\n\
            difficulty = 'STANDARD'")
            performance = "medium"
            difficulty = "STANDARD"

        feedback_data = {
            "performance": performance,
            "difficulty": difficulty
        }

        # Gọi hàm điều chỉnh trọng số heuristic
        adjust_heuristic_weights(feedback_data)

        # Bước 7: Lưu dữ liệu học tập (sử dụng data_loader wrapper)
        try:
            if driver:
                # save each learning_data entry via dl_save_learning_data
                for entry in learning_data:
                    data = {
                        "student_id": student_id,
                        "node_id": entry["node_id"],
                        "score": entry["score"],
                        "time_spent": entry["time_spent"],
                        "feedback": "",
                        "quiz_responses": ""
                    }
                    dl_save_learning_data(driver, data)
            else:
                # Attempt to call with None driver to keep import-time behavior consistent
                for entry in learning_data:
                    data = {
                        "student_id": student_id,
                        "node_id": entry["node_id"],
                        "score": entry["score"],
                        "time_spent": entry["time_spent"],
                        "feedback": "",
                        "quiz_responses": ""
                    }
                    dl_save_learning_data(None, data)
        except Exception as e:
            logger.warning(f"Lỗi lưu dữ liệu: {str(e)}")
            raise ValueError(f"Lỗi lưu dữ liệu: {str(e)}")

        logger.info("Phiên học tập hoàn tất")
        print("\nPhiên học tập hoàn tất!")

        return {
            "status": "success",
            "error_message": None,
            "learned_nodes": learned_nodes
        }

    except Exception as e:
        logger.error(f"Lỗi trong run_learning_session: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e)
        }