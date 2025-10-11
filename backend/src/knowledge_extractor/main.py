"""
Main entry point for the knowledge extraction and validation pipeline.
"""
import os
import argparse
import logging
from .extractor import PDFKnowledgeExtractor
from .validator import CSVValidator, GraphValidator
from ..config import Config
from neo4j import GraphDatabase

# --- Thêm vào ---
# Import các thành phần cần thiết để khởi tạo LLM
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
# ----------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def run_extraction(pdf_path: str, output_dir: str):
    """
    Runs the full extraction pipeline for a single PDF file.
    1. Extracts knowledge to CSVs.
    2. Validates and refines the CSVs.
    """
    logging.info(f"Bắt đầu quá trình trích xuất cho: {pdf_path}")

    # --- Thêm vào: Khởi tạo LLM ---
    # Logic này đảm bảo script có thể hoạt động độc lập
    try:
        logging.info("Đang khởi tạo mô hình ngôn ngữ Gemini...")
        llm = Gemini(model_name="models/gemini-pro", api_key=Config.GEMINI_API_KEY)
        # Gán LLM vào cả Config và Settings để đảm bảo tính nhất quán
        Config.LLM = llm
        Settings.llm = llm
        logging.info("Khởi tạo LLM thành công.")
    except Exception as e:
        logging.error(f"Lỗi nghiêm trọng khi khởi tạo LLM: {e}")
        logging.error("Hãy đảm bảo biến GEMINI_API_KEY trong file .env đã được cấu hình chính xác.")
        return
    # ---------------------------------

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for prompts
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generator_prompt_path = os.path.join(project_root, 'prompts', 'spr_generator_prompt.txt')
    validation_prompt_path = os.path.join(project_root, 'prompts', 'validation_prompt.txt')

    # Initialize the extractor
    extractor = PDFKnowledgeExtractor(
        llm_provider=Config.LLM, # Using the configured LLM from main app
        generation_prompt_path=generator_prompt_path,
        validation_prompt_path=validation_prompt_path
    )

    # --- Extraction and Refinement Loop ---
    nodes_df, rels_df = extractor.extract_and_refine(pdf_path, max_refinements=3)

    # --- Save final CSV files ---
    nodes_csv_path = os.path.join(output_dir, 'nodes.csv')
    rels_csv_path = os.path.join(output_dir, 'relationships.csv')
    
    nodes_df.to_csv(nodes_csv_path, index=False)
    logging.info(f"Đã lưu file nodes vào: {nodes_csv_path}")
    
    rels_df.to_csv(rels_csv_path, index=False)
    logging.info(f"Đã lưu file relationships vào: {rels_csv_path}")

    # --- Programmatic CSV Validation ---
    logging.info("Bắt đầu kiểm tra logic của file CSV...")
    csv_validator = CSVValidator(nodes_csv_path, rels_csv_path)
    if csv_validator.run_all_validations():
        logging.info("Kiểm tra logic CSV thành công. Các file hợp lệ.")
    else:
        logging.warning("Phát hiện lỗi logic trong file CSV. Vui lòng xem lại các log ở trên.")


def run_graph_validation():
    """
    Connects to Neo4j and runs a suite of validation queries.
    """
    logging.info("Bắt đầu quá trình xác thực toàn bộ đồ thị trên Neo4j...")
    try:
        driver = GraphDatabase.driver(Config.NEO4J_URL, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        graph_validator = GraphValidator(driver)
        
        if graph_validator.run_all_validations():
            logging.info("Xác thực đồ thị hoàn tất. Không tìm thấy lỗi nghiêm trọng.")
        else:
            logging.warning("Xác thực đồ thị phát hiện các vấn đề tiềm ẩn. Vui lòng xem lại log.")
            
        driver.close()
    except Exception as e:
        logging.error(f"Không thể kết nối hoặc chạy xác thực trên Neo4j: {e}")
        logging.error("Hãy đảm bảo file .env của bạn đã được cấu hình đúng và Neo4j AuraDB đang hoạt động.")


def main():
    """
    Command-line interface to run the extraction and validation processes.
    """
    parser = argparse.ArgumentParser(description="Công cụ trích xuất và xác thực Knowledge Graph từ tài liệu PDF.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Các lệnh có sẵn')

    # --- 'extract' command ---
    extract_parser = subparsers.add_parser('extract', help='Trích xuất kiến thức từ một file PDF và tạo ra các file CSV.')
    extract_parser.add_argument('--pdf', type=str, required=True, help='Đường dẫn đến file PDF nguồn.')
    extract_parser.add_argument('--output-dir', type=str, required=True, help='Thư mục để lưu các file nodes.csv và relationships.csv.')

    # --- 'validate-graph' command ---
    subparsers.add_parser('validate-graph', help='Chạy các bài kiểm tra xác thực trên toàn bộ đồ thị Neo4j.')

    args = parser.parse_args()

    if args.command == 'extract':
        run_extraction(pdf_path=args.pdf, output_dir=args.output_dir)
    elif args.command == 'validate-graph':
        run_graph_validation()

if __name__ == '__main__':
    main()
