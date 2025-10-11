"""
Module này chịu trách nhiệm sinh ra các nội dung học tập và bài kiểm tra.
Nó sử dụng LLM (thông qua LlamaIndex Settings) và các prompt được thiết kế
cẩn thận để tạo ra tài liệu phù hợp với từng nút kiến thức và hồ sơ học viên.
"""

import logging
import json
import re
import csv
import os
from typing import Dict, List

from neo4j import GraphDatabase
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

# Import các module/hàm cần thiết từ các file khác trong dự án
from .config import Config
from .data_loader import load_student_profile, execute_cypher_query

# Thiết lập logger
logger = logging.getLogger(__name__)

# ==============================================================================
# HÀM TIỆN ÍCH
# ==============================================================================

# Hàm làm sạch phản hồi JSON
def clean_json_response(response_str: str) -> str:
    """Làm sạch chuỗi JSON để đảm bảo định dạng hợp lệ.

    Args:
        response_str (str): Chuỗi JSON gốc.

    Returns:
        str: Chuỗi JSON đã được làm sạch.

    Raises:
        ValueError: Nếu không tìm thấy đối tượng JSON hợp lệ.
    """
    try:
        start = min(response_str.find('{'), response_str.find('['))
        if start == -1:
            raise ValueError("Không tìm thấy đối tượng JSON trong phản hồi")
        response_str = response_str[start:]
        end = max(response_str.rfind('}'), response_str.rfind(']')) + 1
        if end == 0:
            raise ValueError("Không tìm thấy đối tượng JSON trong phản hồi")
        response_str = response_str[:end]
        response_str = re.sub(r',\s*}', '}', response_str)
        response_str = re.sub(r',\s*]', ']', response_str)
        return response_str
    except Exception as e:
        logger.error(f"Lỗi làm sạch JSON: {str(e)}")
        raise

# ==============================================================================
# CÁC HÀM SINH NỘI DUNG
# ==============================================================================

def generate_learning_content(node: Dict, student_id: str = None, student_file: str = Config.STUDENT_FILE, student_goal: str = None) -> Dict:
    """Sinh nội dung học tập tùy chỉnh cho một nút trên lộ trình.

    Args:
        node (Dict): Thông tin nút (sanitized_concept, definition, example, learning_objective, skill_level, node_id, context, difficulty, time_estimate, semantic_tags, v.v.).
        student_id (str, optional): ID của học sinh để lấy phong cách học tập.
        student_file (str): Đường dẫn đến file hồ sơ học sinh.

    Returns:
        Dict: Dictionary chứa 'content', 'status', và 'error_message'.
    """
    try:
        # Bước 1: Kiểm tra đầu vào
        if not isinstance(node, dict):
            raise ValueError("Nút phải là một dictionary")
        required_fields = ["node_id", "sanitized_concept"]
        missing_fields = [field for field in required_fields if field not in node or not node[field]]
        if missing_fields:
            raise ValueError(f"Nút thiếu các trường bắt buộc: {missing_fields}")

        # Bước 2: Xác định phong cách học tập
        learning_style = node.get("learning_style_preference")
        if not learning_style and student_id:
            profile = load_student_profile(student_id)
            learning_style = profile.get("learning_style_preference", None)
        learning_style = learning_style or "reading_writing"

        # Bước 3: Truy vấn thông tin subconcepts (nếu có)
        query = f"""
        MATCH (n {{{Config.PROPERTY_ID}: $node_id}})<-[:{Config.RELATIONSHIP_IS_SUBCONCEPT_OF}]-(sub)
        RETURN sub.{Config.PROPERTY_DEFINITION} AS definition, sub.{Config.PROPERTY_EXAMPLE} AS example
        """
        subconcepts = execute_cypher_query(driver, query, params={"node_id": node["node_id"]})
        subconcept_content = ""
        if subconcepts:
            subconcept_content = "\n".join([f"Khái niệm nền: {sub['definition']} (Ví dụ: {sub['example']})" for sub in subconcepts])

        # Bước 4: Chuẩn bị các phần thông tin cho prompt từ tất cả thuộc tính của node và subconcepts
        node_info = "\n".join([f"- {key.replace('_', ' ').title()}: {value if value else 'N/A'}" for key, value in node.items()])
        if subconcept_content:
            node_info += f"\n\nĐể hiểu rõ hơn, bạn cần nắm các khái niệm nền sau:\n{subconcept_content}"

        # Bước 5: Tạo prompt cho LLM với tất cả thông tin từ node và subconcepts
        # prompt_template = PromptTemplate(f"""
        # # MISSION
        # Bạn là một chuyên gia giáo dục, nhiệm vụ của bạn là tạo ra nội dung học tập **bằng tiếng Việt** chi tiết, dễ hiểu và phù hợp với học sinh dựa trên thông tin sau:

        # - **Mục tiêu học tập của học sinh**: {student_goal or 'N/A'}
        # - **Thông tin khái niệm**: {node_info}

        # # METHODOLOGY
        # - Đóng vai là một chuyên gia hiểu rõ về mục tiêu học tập của học sinh: {student_goal}
        # - **Tập trung giải thích khái niệm chính**: Khái niệm chính là "{node['sanitized_concept']}". Hãy giải thích nó một cách rõ ràng, dễ hiểu, và liên kết với các thông tin khác như định nghĩa, ví dụ, và mục tiêu học tập.
        # - **Cung cấp ví dụ thực tế**: Đưa ra ít nhất một ví dụ cụ thể để minh họa khái niệm chính.
        # - **Liên kết với mục tiêu học tập**: Giải thích cách khái niệm này giúp học sinh đạt được mục tiêu học tập "{node.get('learning_objective', 'N/A')}" ở trình độ kỹ năng "{node.get('skill_level', 'N/A')}".
        # - **Tích hợp kỹ thuật Socratic & Thought-Provoking**: Đặt ít nhất một câu hỏi hoặc tình huống giả định để khuyến khích học sinh suy nghĩ sâu hơn. Ví dụ: "Nếu bạn thay đổi X thành Y, điều gì sẽ xảy ra? Hãy khám phá lý do."
        # - **Điều chỉnh theo phong cách học tập**: Nội dung phải phù hợp với phong cách học tập "{learning_style}". Ví dụ:
        #   - Nếu là "visual", hãy mô tả bằng hình ảnh hoặc biểu đồ.
        #   - Nếu là "auditory", hãy giải thích bằng ví dụ âm thanh hoặc câu chuyện.
        #   - Nếu là "reading_writing", tập trung vào văn bản và ghi chú.
        #   - Nếu là "kinesthetic", đưa ra hoạt động thực hành hoặc ví dụ thực tiễn.
        # - **Nếu có khái niệm nền**, trình bày theo cấu trúc ba bước: xác định, giải thích, minh họa.
        # - **Nếu thông tin nào đó không có (N/A)**, hãy tự tạo nội dung phù hợp dựa trên kiến thức chung.
        # - Trả về văn bản thuần túy.

        # # SUBCONCEPTS GUIDANCE
        # Nếu có khái niệm nền, trình bày riêng biệt từng khái niệm theo cấu trúc ba bước sau:
        # 1. **Xác định**: Nêu rõ khái niệm nền là gì.
        # 2. **Giải thích**: Mô tả ngắn gọn ý nghĩa và vai trò của nó.
        # 3. **Minh họa**: Đưa ra một ví dụ đơn giản để làm rõ khái niệm.

        # # OUTPUT
        # - Phần nội dung mà học sinh sẽ dùng để học tập.
        # - Lưu ý trả về dạng văn bản học tập, bỏ qua các đại từ nhân xưng.
        # """)
        prompt_template = PromptTemplate(f"""
        # MISSION
        Tạo nội dung học tập **bằng tiếng Việt** chi tiết, mang tính học thuật, dễ hiểu, và phù hợp với học sinh, đạt độ dài tối thiểu **3.000-4.000 từ**, dựa trên thông tin sau:
        - **Mục tiêu học tập của học sinh**: {student_goal or 'N/A'}
        - **Thông tin khái niệm**: {node_info}

        # METHODOLOGY
        - Đóng vai chuyên gia giáo dục am hiểu mục tiêu học tập: {student_goal}.
        - **Giải thích khái niệm chính**: Khái niệm chính là "{node['sanitized_concept']}". Trình bày chi tiết qua các phần:
          - **Giới thiệu**: Đặt khái niệm trong bối cảnh thực tế, giải thích tầm quan trọng, và trình bày bối cảnh lịch sử/phát triển của khái niệm (300-400 từ).
          - **Giải thích chi tiết**: Mô tả ý nghĩa, cơ chế hoạt động, và vai trò của khái niệm, liên kết với lý thuyết giáo dục (ví dụ: Bloom’s Taxonomy, VARK) và các mô hình học tập liên quan (700-1.000 từ).
          - **Ví dụ thực tế**: Đưa ra ít nhất bốn ví dụ cụ thể, đa dạng, minh họa ứng dụng trong các tình huống khác nhau (ví dụ: học thuật, nghề nghiệp, đời sống) (400-600 từ).
          - **Ứng dụng thực tế**: Phân tích cách áp dụng khái niệm trong ít nhất ba kịch bản thực tế (học tập, công việc, hoặc xã hội), bao gồm lợi ích và thách thức (400-600 từ).
          - **So sánh và phân biệt**: So sánh khái niệm với ít nhất hai khái niệm liên quan để làm rõ sự khác biệt và điểm tương đồng, nhấn mạnh ưu/nhược điểm (300-400 từ).
          - **Phân tích nâng cao**: Thảo luận các khía cạnh phức tạp, tranh luận học thuật, hoặc ứng dụng nâng cao của khái niệm (300-400 từ).
          - **Tóm tắt**: Nhấn mạnh ý chính, lợi ích lâu dài, và giá trị của việc hiểu khái niệm (150-200 từ).
        - **Liên kết với mục tiêu học tập**: Giải thích cách khái niệm giúp đạt được mục tiêu "{node.get('learning_objective', 'N/A')}" ở trình độ kỹ năng "{node.get('skill_level', 'N/A')}", liên hệ với các cấp độ tư duy của Bloom (200-300 từ).
        - **Tích hợp kỹ thuật Socratic**: Đặt ít nhất bốn câu hỏi hoặc tình huống giả định để khuyến khích tư duy phản biện, ví dụ:
          - "Điều gì xảy ra nếu không áp dụng khái niệm này trong tình huống X?"
          - "Làm thế nào để tối ưu hóa khái niệm trong bối cảnh Y?"
          - "Tại sao khái niệm này quan trọng hơn khái niệm Z trong trường hợp cụ thể?"
          - "Nếu thay đổi yếu tố A thành B, kết quả sẽ thay đổi thế nào?" (200-300 từ).
          - **Đáp án mẫu**: Cung cấp hướng dẫn trả lời hoặc đáp án mẫu cho mỗi câu hỏi, giải thích lý do và cách tư duy để đạt câu trả lời (300-400 từ tổng cộng).
        - **Điều chỉnh theo phong cách học tập**: Tùy chỉnh nội dung theo "{learning_style}":
          - **Visual**: Mô tả cách sử dụng sơ đồ tư duy, biểu đồ, hoặc hình ảnh minh họa chi tiết, kèm hướng dẫn vẽ sơ đồ (100-150 từ).
          - **Auditory**: Đề xuất câu chuyện, kịch bản âm thanh, hoặc gợi ý tổ chức thảo luận nhóm (100-150 từ).
          - **Reading/Writing**: Cung cấp văn bản chi tiết, gợi ý ghi chú, bài tập viết phản ánh, hoặc tóm tắt bằng văn bản (100-150 từ).
          - **Kinesthetic**: Đưa ra thí nghiệm, dự án thực hành, hoặc hoạt động mô phỏng, kèm hướng dẫn thực hiện (100-150 từ).
        - **Tăng tính tương tác**: Bao gồm:
          - **Năm câu hỏi trắc nghiệm**: Mỗi câu có 4 đáp án, kèm giải thích chi tiết cho đáp án đúng và lý do các đáp án sai là không phù hợp (300-400 từ).
          - **Hai bài tập thực hành**: Một bài tập cơ bản và một bài tập nâng cao, liên quan đến khái niệm, với hướng dẫn thực hiện (200-300 từ).
          - **Tài nguyên bổ sung**: Gợi ý ít nhất năm tài nguyên (sách, bài báo, video, khóa học trực tuyến) với mô tả ngắn về nội dung và cách sử dụng (200-300 từ).
        - **Xử lý khái niệm nền**: Nếu có, trình bày chi tiết ít nhất ba khái niệm nền theo ba bước:
          1. **Xác định**: Khái niệm nền là gì.
          2. **Giải thích**: Ý nghĩa, vai trò, và mối liên hệ với khái niệm chính.
          3. **Minh họa**: Ví dụ thực tế hoặc tình huống cụ thể (250-350 từ/khái niệm).
        - **Trường hợp thiếu thông tin (N/A)**: Tự tạo nội dung hợp lý dựa trên kiến thức chung, đảm bảo liên quan đến giáo dục và mục tiêu học tập.
        - Trả về văn bản thuần túy, không dùng đại từ nhân xưng, ngôn ngữ chính xác, khách quan, phù hợp với tài liệu học thuật.

        # SUBCONCEPTS GUIDANCE
        Với khái niệm nền, trình bày rõ mối liên hệ với khái niệm chính, đảm bảo mỗi khái niệm nền được giải thích đầy đủ, minh họa bằng ví dụ cụ thể, và liên kết với mục tiêu học tập.

        # OUTPUT
        - Nội dung học tập hoàn chỉnh, đạt độ dài **3.000-4.000 từ**, có cấu trúc logic từ tổng quan đến chi tiết, dễ theo dõi, mang tính học thuật với các phần được phân chia rõ ràng (giới thiệu, giải thích, ví dụ, ứng dụng, so sánh, phân tích, tóm tắt, câu hỏi, bài tập, tài nguyên). Bao gồm đáp án mẫu và giải thích cho câu hỏi tự kiểm tra và Socratic.
""")

        # Bước 6: Gọi LLM để sinh nội dung
        content = Settings.llm.complete(prompt_template.format()).text.strip()

        if not content:
            raise ValueError("LLM trả về nội dung rỗng")

        return {
            "content": content,
            "status": "success",
            "error_message": None
        }

    except Exception as e:
        logger.error(f"Lỗi trong generate_learning_content: {str(e)}")
        return {
            "content": "",
            "status": "error",
            "error_message": str(e)
        }

def generate_quiz(node: Dict, output_dir: str = Config.QUIZ_OUTPUT_DIR) -> Dict:
    """Tạo bài kiểm tra với 20 câu hỏi trắc nghiệm cho một nút.

    Args:
        node (Dict): Thông tin nút (node_id, sanitized_concept, context, definition, example, learning_objective, skill_level, difficulty).
        output_dir (str): Thư mục lưu file bài kiểm tra.

    Returns:
        Dict: Dictionary chứa 'student_file', 'answer_file', 'status', và 'error_message'.
    """
    try:
        # Bước 1: Kiểm tra đầu vào
        if not isinstance(node, dict):
            raise ValueError("Nút phải là một dictionary")

        # Ghi log nội dung node để debug
        logger.debug(f"Nội dung node: {node}")

        required_fields = [
            Config.PROPERTY_ID.lower(),  # node_id
            Config.PROPERTY_SANITIZED_CONCEPT.lower(),  # sanitized_concept
            Config.PROPERTY_CONTEXT.lower(),  # context
            Config.PROPERTY_DEFINITION.lower(),  # definition
            Config.PROPERTY_EXAMPLE.lower(),  # example
            Config.PROPERTY_LEARNING_OBJECTIVE.lower(),  # learning_objective
            Config.PROPERTY_SKILL_LEVEL.lower(),  # skill_level
            Config.PROPERTY_DIFFICULTY.lower()  # difficulty
        ]
        missing_fields = [field for field in required_fields if field not in node or not node[field]]
        if missing_fields:
            raise ValueError(f"Nút thiếu các trường bắt buộc: {missing_fields}")

        # Bước 2: Tạo prompt cho LLM
        common_errors = node.get('common_errors', [])
        common_errors_str = ", ".join(common_errors) if isinstance(common_errors, list) else common_errors if isinstance(common_errors, str) else ""

        prompt_parts = [
            "Dựa trên metadata của nút:",
            f"- Khái niệm: {node.get('sanitized_concept', '')}",
            f"- Ngữ cảnh: {node.get('context', '')}",
            f"- Định nghĩa: {node.get('definition', '')}",
            f"- Ví dụ: {node.get('example', '')}",
            f"- Mục tiêu học tập: {node.get('learning_objective', '')}",
            f"- Trình độ kỹ năng: {node.get('skill_level', '')}",
            f"- Độ khó: {node.get('difficulty', '')}",
            f"- Lỗi thường gặp: {common_errors_str}" if common_errors_str else "Không có lỗi thường gặp"
        ]

        prompt = PromptTemplate("\n".join(part for part in prompt_parts) + f"""
        Tạo chính xác {Config.QUIZ_NUM_QUESTIONS} câu hỏi trắc nghiệm **bằng tiếng Việt**, mỗi câu có 4 lựa chọn, dựa trên thông tin nút ở trên:
        - {int(Config.QUIZ_DISTRIBUTION['basic'] * 100)}% câu hỏi cơ bản (kiến thức nền, khái niệm cơ bản).
        - {int(Config.QUIZ_DISTRIBUTION['intermediate'] * 100)}% câu hỏi trung cấp (ứng dụng đơn giản, ví dụ thực tế).
        - {int(Config.QUIZ_DISTRIBUTION['advanced'] * 100)}% câu hỏi nâng cao (phân tích, suy luận).
        - Đảm bảo câu hỏi liên quan chặt chẽ đến khái niệm chính, sử dụng định nghĩa, ví dụ, và mục tiêu học tập để tạo nội dung.
        - Nếu có lỗi thường gặp, đưa chúng vào các đáp án sai để kiểm tra sự hiểu biết của học sinh.
        - Mỗi câu hỏi phải có lời giải thích chi tiết (explanation) để học sinh hiểu lý do chọn đáp án đúng.
        - Lưu ý:
            + Phải tạo chính xác số lượng {Config.QUIZ_NUM_QUESTIONS} câu hỏi.
            + Câu trả lời chính xác luôn luôn ở vị trí index 0, hay tương ứng là vị trí "Option 1".
        Trả về JSON:
        {{
            "questions": [
                {{
                    "text": "Question text",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "correct_index": 0-3,
                    "explanation": "Explanation text",
                    "level": "basic|intermediate|advanced"
                }}
            ]
        }}
        """)

        # Bước 3: Gọi LLM để sinh bài kiểm tra
        for attempt in range(2):  # Thử tối đa 2 lần
            try:
                quiz_json = Settings.llm.complete(
                    prompt.format(
                        num_questions=Config.QUIZ_NUM_QUESTIONS,
                        basic_pct=int(Config.QUIZ_DISTRIBUTION["basic"] * 100),
                        intermediate_pct=int(Config.QUIZ_DISTRIBUTION["intermediate"] * 100),
                        advanced_pct=int(Config.QUIZ_DISTRIBUTION["advanced"] * 100)
                    )
                ).text
                cleaned_quiz_str = clean_json_response(quiz_json)
                quiz = json.loads(cleaned_quiz_str)
                if not isinstance(quiz, dict) or "questions" not in quiz or not isinstance(quiz["questions"], list):
                    raise ValueError("Phản hồi LLM không hợp lệ")
                break
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == 1:
                    raise ValueError(f"Không thể tạo bài kiểm tra sau 2 lần thử: {str(e)}")
                logger.warning(f"Lỗi khi gọi LLM: {str(e)}, thử lại...")

        # Bước 4: Kiểm tra tỷ lệ câu hỏi
        total_pct = Config.QUIZ_DISTRIBUTION["basic"] + Config.QUIZ_DISTRIBUTION["intermediate"] + Config.QUIZ_DISTRIBUTION["advanced"]
        if abs(total_pct - 1.0) > 0.01:
            raise ValueError("Tổng tỷ lệ câu hỏi phải bằng 100%")

        # Bước 5: Lưu bài kiểm tra vào file CSV
        os.makedirs(output_dir, exist_ok=True)
        student_file = f"{output_dir}/student_quiz_{node['node_id']}.csv"
        answer_file = f"{output_dir}/answer_key_{node['node_id']}.csv"

        with open(student_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question_Number", "Question_Text", "Option_1", "Option_2", "Option_3", "Option_4", "Level"])
            for i, q in enumerate(quiz["questions"], 1):
                writer.writerow([i, q["text"]] + q["options"] + [q["level"]])

        with open(answer_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question_Number", "type", "correct", "explanation", "options"])
            for i, q in enumerate(quiz["questions"], 1):
                options_str = ",".join(q["options"])  # Chuyển danh sách options thành chuỗi
                writer.writerow([i, "multiple_choice", q["correct_index"], q["explanation"], options_str])

        return {
            "student_file": student_file,
            "answer_file": answer_file,
            "status": "success",
            "error_message": None
        }

    except Exception as e:
        logger.error(f"Lỗi trong generate_quiz: {str(e)}")
        return {
            "student_file": None,
            "answer_file": None,
            "status": "error",
            "error_message": str(e)
        }
    
def evaluate_quiz(student_file: str, answer_file: str) -> Dict:
    """Đánh giá bài kiểm tra của học sinh dựa trên file đáp án.

    Args:
        student_file (str): Đường dẫn đến file CSV chứa câu trả lời của học sinh.
        answer_file (str): Đường dẫn đến file CSV chứa đáp án đúng.

    Returns:
        Dict: Dictionary chứa 'score', 'total_questions', 'correct_count', 'feedback', 'status', và 'error_message'.
    """
    try:
        # Đọc file đáp án
        with open(answer_file, "r", encoding="utf-8") as f:
            answer_reader = csv.DictReader(f)
            answer_data = {row["Question_Number"]: row for row in answer_reader}

        # Đọc file câu trả lời của học sinh
        with open(student_file, "r", encoding="utf-8") as f:
            student_reader = csv.DictReader(f)
            student_data = {row["Question_Number"]: row["Student_Answer"].strip().lower() for row in student_reader}

        total_questions = len(answer_data)
        correct_count = 0
        feedback = []

        # Đánh giá từng câu hỏi
        for q_num, answer in answer_data.items():
            if q_num not in student_data:
                feedback.append(f"Câu hỏi {q_num}: Thiếu câu trả lời từ học sinh.")
                continue

            question_type = answer["type"]
            correct_answer = answer["correct"]
            explanation = answer["explanation"]
            student_answer = student_data[q_num]

            # Xử lý câu hỏi trắc nghiệm (multiple_choice)
            if question_type == "multiple_choice":
                options = [opt.strip().lower() for opt in answer["options"].split(",")]
                try:
                    correct_index = int(correct_answer)
                    correct_option = options[correct_index]
                except (ValueError, IndexError):
                    logger.error(f"Lỗi định dạng đáp án đúng '{correct_answer}' cho câu hỏi {q_num}")
                    feedback.append(f"Câu hỏi {q_num}: Lỗi định dạng đáp án đúng trong file đáp án.")
                    continue

                if student_answer == correct_option:
                    correct_count += 1
                else:
                    feedback.append(f"Câu hỏi {q_num}: Sai. Đáp án đúng là '{correct_option}'. Giải thích: {explanation}")
            else:
                # Trường hợp không mong đợi: ghi log và giả định là trắc nghiệm
                logger.warning(f"Loại câu hỏi không mong đợi '{question_type}' cho câu hỏi {q_num}. Giả định là multiple_choice.")
                options = [opt.strip().lower() for opt in answer["options"].split(",")]
                try:
                    correct_index = int(correct_answer)
                    correct_option = options[correct_index]
                except (ValueError, IndexError):
                    logger.error(f"Lỗi định dạng đáp án đúng '{correct_answer}' cho câu hỏi {q_num}")
                    feedback.append(f"Câu hỏi {q_num}: Lỗi định dạng đáp án đúng trong file đáp án.")
                    continue

                if student_answer == correct_option:
                    correct_count += 1
                else:
                    feedback.append(f"Câu hỏi {q_num}: Sai. Đáp án đúng là '{correct_option}'. Giải thích: {explanation}")

        # Tính điểm
        score = (correct_count / total_questions) * 100 if total_questions > 0 else 0.0

        return {
            "score": f"{score:.0f}",
            "total_questions": total_questions,
            "correct_count": correct_count,
            "feedback": feedback,
            "status": "success",
            "error_message": None
        }

    except FileNotFoundError as e:
        logger.error(f"Không tìm thấy file: {str(e)}")
        return {
            "score": "0.00",
            "total_questions": 0,
            "correct_count": 0,
            "feedback": [],
            "status": "error",
            "error_message": f"Không tìm thấy file: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Lỗi trong evaluate_quiz: {str(e)}")
        return {
            "score": "0.00",
            "total_questions": 0,
            "correct_count": 0,
            "feedback": [],
            "status": "error",
            "error_message": str(e)
        }