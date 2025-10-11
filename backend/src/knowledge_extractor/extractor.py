"""
Contains the core logic for extracting knowledge from PDFs using an LLM.
"""
import os
import pandas as pd
import logging
from typing import Tuple, Optional
from llama_index.core import Document
from llama_index.core.llms import LLM
import re
import io

class PDFKnowledgeExtractor:
    def __init__(self, llm_provider: LLM, generation_prompt_path: str, validation_prompt_path: str):
        self.llm = llm_provider
        self.generation_prompt = self._load_prompt(generation_prompt_path)
        self.validation_prompt = self._load_prompt(validation_prompt_path)
        self.logger = logging.getLogger(__name__)

    def _load_prompt(self, path: str) -> str:
        """Loads a prompt from a file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Lỗi: Không tìm thấy file prompt tại: {path}")
            raise

    def _load_pdf_content(self, pdf_path: str) -> str:
        """Loads content from a PDF file."""
        # In a real scenario, you would use a PDF parsing library like PyMuPDF or pdfplumber
        # For this example, we'll simulate reading text.
        # Replace this with your actual PDF parsing logic.
        try:
            # This is a placeholder. Use a real PDF library.
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            if not text.strip():
                raise ValueError("Không thể trích xuất nội dung từ PDF. File có thể là hình ảnh hoặc bị trống.")
            return text
        except ImportError:
            self.logger.error("Lỗi: Thư viện 'pypdf' chưa được cài đặt. Hãy chạy 'pip install pypdf'.")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi khi đọc file PDF '{pdf_path}': {e}")
            raise

    def _parse_llm_output(self, llm_response: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Parses the LLM's markdown-formatted response into two DataFrames.
        """
        self.logger.debug(f"Raw LLM response:\n{llm_response}")
        
        try:
            # Find nodes CSV block
            nodes_match = re.search(r"```csv\s*nodes.csv\s*([\s\S]*?)```", llm_response, re.IGNORECASE)
            if not nodes_match:
                self.logger.warning("Không tìm thấy khối 'nodes.csv' trong phản hồi của LLM.")
                return None, None
            nodes_csv_str = nodes_match.group(1).strip()

            # Find relationships CSV block
            rels_match = re.search(r"```csv\s*relationships.csv\s*([\s\S]*?)```", llm_response, re.IGNORECASE)
            if not rels_match:
                self.logger.warning("Không tìm thấy khối 'relationships.csv' trong phản hồi của LLM.")
                return None, None
            rels_csv_str = rels_match.group(1).strip()

            # Convert CSV strings to DataFrames
            nodes_df = pd.read_csv(io.StringIO(nodes_csv_str), dtype=str)
            rels_df = pd.read_csv(io.StringIO(rels_csv_str), dtype=str)
            
            return nodes_df, rels_df
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích phản hồi của LLM: {e}")
            self.logger.error(f"Dữ liệu CSV không hợp lệ:\nNodes:\n{nodes_csv_str}\nRels:\n{rels_csv_str}")
            return None, None

    def extract_and_refine(self, pdf_path: str, max_refinements: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        The main pipeline:
        1. Load PDF content.
        2. Perform initial extraction using the generation prompt.
        3. Iteratively validate and refine the output using the validation prompt.
        """
        pdf_content = self._load_pdf_content(pdf_path)
        
        # --- Initial Extraction ---
        self.logger.info("Bước 1: Bắt đầu trích xuất kiến thức ban đầu từ PDF...")
        initial_prompt = self.generation_prompt.format(document_content=pdf_content)
        response = self.llm.complete(initial_prompt)
        
        nodes_df, rels_df = self._parse_llm_output(response.text)
        if nodes_df is None or rels_df is None:
            raise ValueError("Trích xuất ban đầu thất bại. Không thể phân tích phản hồi của LLM.")

        self.logger.info(f"Trích xuất ban đầu thành công. Tìm thấy {len(nodes_df)} nút và {len(rels_df)} mối quan hệ.")

        # --- Iterative Refinement ---
        for i in range(max_refinements):
            self.logger.info(f"Bước 2.{i+1}: Bắt đầu vòng lặp kiểm tra và tinh chỉnh lần thứ {i+1}/{max_refinements}...")
            
            # Convert current dataframes back to CSV strings for the prompt
            current_nodes_csv = nodes_df.to_csv(index=False)
            current_rels_csv = rels_df.to_csv(index=False)

            refinement_prompt = self.validation_prompt.format(
                document_content=pdf_content,
                nodes_csv=current_nodes_csv,
                relationships_csv=current_rels_csv
            )
            
            response = self.llm.complete(refinement_prompt)
            
            # The LLM should respond with "VALID" or provide corrected CSVs
            if "VALID" in response.text.upper():
                self.logger.info("LLM xác nhận dữ liệu hợp lệ. Kết thúc quá trình tinh chỉnh.")
                break

            self.logger.info("LLM đã đề xuất các thay đổi. Đang áp dụng...")
            refined_nodes_df, refined_rels_df = self._parse_llm_output(response.text)

            if refined_nodes_df is not None and refined_rels_df is not None:
                self.logger.info(f"Cập nhật thành công: {len(refined_nodes_df)} nút, {len(refined_rels_df)} mối quan hệ.")
                nodes_df = refined_nodes_df
                rels_df = refined_rels_df
            else:
                self.logger.warning("Không thể phân tích các file CSV đã tinh chỉnh từ LLM. Giữ lại phiên bản trước đó.")
        
        else: # This 'else' belongs to the 'for' loop
             self.logger.warning(f"Đã đạt đến giới hạn {max_refinements} lần tinh chỉnh. Sử dụng kết quả cuối cùng.")

        return nodes_df, rels_df
