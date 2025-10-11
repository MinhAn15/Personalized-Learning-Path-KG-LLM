"""
Contains validation logic for both individual CSV files and the entire graph.
"""
import pandas as pd
import logging
from neo4j import GraphDatabase
from typing import List, Dict

class CSVValidator:
    """Performs programmatic validation on the generated CSV files."""
    def __init__(self, nodes_path: str, rels_path: str):
        self.logger = logging.getLogger(__name__)
        try:
            self.nodes_df = pd.read_csv(nodes_path, dtype=str)
            self.rels_df = pd.read_csv(rels_path, dtype=str)
        except FileNotFoundError as e:
            self.logger.error(f"Lỗi: Không tìm thấy file CSV tại đường dẫn: {e.filename}")
            raise

    def _check_columns(self) -> bool:
        """Checks for required columns."""
        REQUIRED_NODE_COLS = ['Node_ID', 'Name', 'Type']
        REQUIRED_REL_COLS = ['Source_ID', 'Target_ID', 'Type']
        
        nodes_ok = all(col in self.nodes_df.columns for col in REQUIRED_NODE_COLS)
        if not nodes_ok:
            self.logger.error(f"File nodes.csv thiếu các cột bắt buộc. Cần có: {REQUIRED_NODE_COLS}. Hiện có: {list(self.nodes_df.columns)}")
        
        rels_ok = all(col in self.rels_df.columns for col in REQUIRED_REL_COLS)
        if not rels_ok:
            self.logger.error(f"File relationships.csv thiếu các cột bắt buộc. Cần có: {REQUIRED_REL_COLS}. Hiện có: {list(self.rels_df.columns)}")
            
        return nodes_ok and rels_ok

    def _check_id_uniqueness(self) -> bool:
        """Checks if all Node_IDs are unique."""
        if self.nodes_df['Node_ID'].duplicated().any():
            duplicates = self.nodes_df[self.nodes_df['Node_ID'].duplicated()]['Node_ID'].tolist()
            self.logger.error(f"Lỗi: Tìm thấy các Node_ID bị trùng lặp trong nodes.csv: {duplicates}")
            return False
        return True

    def _check_relationship_integrity(self) -> bool:
        """Checks if all Source_ID and Target_ID in relationships exist in nodes."""
        all_node_ids = set(self.nodes_df['Node_ID'])
        
        missing_sources = set(self.rels_df['Source_ID']) - all_node_ids
        if missing_sources:
            self.logger.error(f"Lỗi: relationships.csv có các Source_ID không tồn tại trong nodes.csv: {missing_sources}")
        
        missing_targets = set(self.rels_df['Target_ID']) - all_node_ids
        if missing_targets:
            self.logger.error(f"Lỗi: relationships.csv có các Target_ID không tồn tại trong nodes.csv: {missing_targets}")
            
        return not missing_sources and not missing_targets

    def run_all_validations(self) -> bool:
        """Runs all CSV validation checks."""
        self.logger.info("--- Bắt đầu kiểm tra logic CSV ---")
        results = [
            self._check_columns(),
            self._check_id_uniqueness(),
            self._check_relationship_integrity()
        ]
        self.logger.info("--- Kết thúc kiểm tra logic CSV ---")
        return all(results)


class GraphValidator:
    """Runs validation queries against the live Neo4j database."""
    def __init__(self, driver: GraphDatabase.driver):
        self.driver = driver
        self.logger = logging.getLogger(__name__)
        self.validation_queries = self._get_validation_queries()

    def _get_validation_queries(self) -> List[Dict]:
        """Defines the suite of validation queries."""
        return [
            {
                "name": "Kiểm tra các nút bị cô lập (không có mối quan hệ)",
                "query": "MATCH (n) WHERE NOT (n)--() RETURN n.Node_ID AS IsolatedNode, n.Name AS NodeName LIMIT 25",
                "level": "warning"
            },
            {
                "name": "Kiểm tra các nút gốc (không có mối quan hệ trỏ đến)",
                "query": "MATCH (n) WHERE NOT ()-[:IS_COMPOSED_OF|HAS_PREREQUISITE|HAS_PART]->(n) RETURN n.Node_ID AS RootNode, n.Name AS NodeName, n.Type AS NodeType LIMIT 25",
                "level": "info"
            },
            {
                "name": "Kiểm tra các nút lá (không có mối quan hệ đi ra)",
                "query": "MATCH (n) WHERE NOT (n)-[]->() RETURN n.Node_ID AS LeafNode, n.Name AS NodeName, n.Type AS NodeType LIMIT 25",
                "level": "info"
            },
            {
                "name": "Kiểm tra các mối quan hệ có thuộc tính bị thiếu",
                "query": "MATCH ()-[r]->() WHERE r.Type IS NULL OR r.Type = '' RETURN id(r) AS RelationshipId LIMIT 25",
                "level": "error"
            },
            {
                "name": "Kiểm tra các chu trình (cycle) trong các mối quan hệ HAS_PREREQUISITE",
                "query": """
                    MATCH path = (n)-[:HAS_PREREQUISITE*]->(n)
                    RETURN n.Node_ID AS NodeInCycle, [node in nodes(path) | node.Node_ID] AS CyclePath
                    LIMIT 10
                """,
                "level": "error"
            },
            {
                "name": "Thống kê các loại nút (Node Types)",
                "query": "MATCH (n) RETURN n.Type AS NodeType, count(n) AS Count ORDER BY Count DESC",
                "level": "info"
            },
            {
                "name": "Thống kê các loại mối quan hệ (Relationship Types)",
                "query": "MATCH ()-[r]->() RETURN type(r) AS RelationshipType, count(r) AS Count ORDER BY Count DESC",
                "level": "info"
            }
        ]

    def run_all_validations(self) -> bool:
        """Executes all defined validation queries against the database."""
        self.logger.info("--- Bắt đầu kiểm tra toàn bộ Graph trên Neo4j ---")
        has_errors = False
        with self.driver.session(database="neo4j") as session:
            for test in self.validation_queries:
                self.logger.info(f"Đang chạy: {test['name']}...")
                try:
                    result = session.run(test['query'])
                    records = [record.data() for record in result]
                    
                    if records:
                        if test['level'] == 'error':
                            has_errors = True
                            self.logger.error(f"  [LỖI] {test['name']} phát hiện {len(records)} vấn đề:")
                        elif test['level'] == 'warning':
                            self.logger.warning(f"  [CẢNH BÁO] {test['name']} phát hiện {len(records)} vấn đề:")
                        else: # info
                            self.logger.info(f"  [THÔNG TIN] {test['name']} có kết quả:")
                        
                        # Log results in a readable format
                        df = pd.DataFrame(records)
                        self.logger.info("\n" + df.to_string(index=False))

                    else:
                        self.logger.info(f"  [OK] Không tìm thấy vấn đề nào cho '{test['name']}'.")
                
                except Exception as e:
                    self.logger.error(f"  [LỖI THỰC THI] Không thể chạy kiểm tra '{test['name']}': {e}")
                    has_errors = True
        
        self.logger.info("--- Kết thúc kiểm tra Graph ---")
        return not has_errors
