import os
import pandas as pd
import re

def sanitize_prefix(path_part):
    """Cleans a path part to be used as a prefix."""
    s = path_part.replace(' ', '_').replace('-', '_')
    s = re.sub(r'[^a-zA-Z0-9_]', '', s)
    return s.upper()

def main():
    """
    Scans subdirectories for nodes.csv and relationships.csv,
    prefixes IDs to ensure uniqueness, and aggregates them into
    two master CSV files.
    """
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_root_dir = os.path.join(project_root, 'data', 'input')
    output_dir = os.path.join(project_root, 'data', 'github_import')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    all_nodes = []
    all_relationships = []

    print(f"Bắt đầu quét các thư mục con trong: {input_root_dir}")

    # Walk through the directory tree
    for dirpath, _, filenames in os.walk(input_root_dir):
        if 'nodes.csv' in filenames and 'relationships.csv' in filenames:
            
            relative_path = os.path.relpath(dirpath, input_root_dir)
            print(f"  -> Tìm thấy cặp file trong: {relative_path}")

            # Create a unique prefix from the folder path
            # e.g., "CSV DEMO SQL/JOIN" -> "SQL_JOIN"
            path_parts = [part for part in relative_path.split(os.sep) if 'DEMO' not in part]
            prefix = '_'.join([sanitize_prefix(p) for p in path_parts])

            if not prefix:
                print(f"     Cảnh báo: Không thể tạo prefix cho '{relative_path}'. Bỏ qua.")
                continue

            # --- Process nodes.csv ---
            nodes_path = os.path.join(dirpath, 'nodes.csv')
            df_nodes = pd.read_csv(nodes_path, dtype=str)
            
            # Store original IDs and create new prefixed IDs
            df_nodes['Original_Node_ID'] = df_nodes['Node_ID']
            df_nodes['Node_ID'] = prefix + '_' + df_nodes['Node_ID'].astype(str)
            
            # Also prefix any prerequisite IDs if the column exists
            if 'Prerequisites' in df_nodes.columns:
                def prefix_prereqs(prereq_str):
                    if not isinstance(prereq_str, str) or pd.isna(prereq_str) or not prereq_str.strip():
                        return ''
                    
                    # Split, prefix, and rejoin
                    original_ids = [item.strip() for item in prereq_str.split(';')]
                    prefixed_ids = [prefix + '_' + item_id for item_id in original_ids if item_id]
                    return ';'.join(prefixed_ids)

                df_nodes['Prerequisites'] = df_nodes['Prerequisites'].apply(prefix_prereqs)

            all_nodes.append(df_nodes)

            # --- Process relationships.csv ---
            rels_path = os.path.join(dirpath, 'relationships.csv')
            df_rels = pd.read_csv(rels_path, dtype=str)

            # Prefix Source_ID and Target_ID
            df_rels['Source_ID'] = prefix + '_' + df_rels['Source_ID'].astype(str)
            df_rels['Target_ID'] = prefix + '_' + df_rels['Target_ID'].astype(str)
            
            all_relationships.append(df_rels)

    if not all_nodes:
        print("Không tìm thấy file nodes.csv nào để xử lý.")
        return

    # Concatenate all dataframes
    master_nodes_df = pd.concat(all_nodes, ignore_index=True)
    master_rels_df = pd.concat(all_relationships, ignore_index=True)

    # Define output paths
    output_nodes_path = os.path.join(output_dir, 'master_nodes.csv')
    output_rels_path = os.path.join(output_dir, 'master_relationships.csv')

    # Save to new master CSV files
    master_nodes_df.to_csv(output_nodes_path, index=False)
    master_rels_df.to_csv(output_rels_path, index=False)

    print("\nHoàn tất!")
    print(f"Đã tạo file master nodes: {output_nodes_path} ({len(master_nodes_df)} dòng)")
    print(f"Đã tạo file master relationships: {output_rels_path} ({len(master_rels_df)} dòng)")
    print("\nBước tiếp theo: Commit và push các file mới trong 'backend/data/github_import/' lên GitHub.")

if __name__ == '__main__':
    main()
