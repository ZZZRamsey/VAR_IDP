import os
import json
import time

def main():
    # Base paths configuration
    # Note: Using the absolute path provided in the requirements
    base_data_path = "/fs-ift/atlas/zouyuefeng/zls/code/VAR_IDP/data/FaceID-6M/laion_512"
    mllm_json_dir = os.path.join(base_data_path, "mllm_json")
    mllm_rec_json_dir = os.path.join(base_data_path, "mllm_rec_json")
    insight_json_dir = os.path.join(base_data_path, "insightface_json")
    
    # Setup output directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, "mllm_analysis_res")
    
    # Create sub-directories for mllm and mllm_rec analysis
    output_dir_mllm = os.path.join(base_output_dir, "mllm_json")
    output_dir_rec = os.path.join(base_output_dir, "mllm_rec_json")
    
    for d in [output_dir_mllm, output_dir_rec]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created output directory: {d}")
        else:
            print(f"Output directory already exists: {d}")

    # ==========================================
    # Task 1: List existing files in mllm_json
    # ==========================================
    print(f"\n--- Analyzing mllm_json ---")
    print(f"Scanning mllm_json directory: {mllm_json_dir}")
    
    mllm_files_set = set()
    if os.path.exists(mllm_json_dir):
        # We only care about .json files usually
        mllm_files_list = [f for f in os.listdir(mllm_json_dir) if f.endswith('.json')]
        mllm_files_set = set(mllm_files_list)
    else:
        print(f"Warning: Directory {mllm_json_dir} does not exist.")

    mllm_list_output_file = os.path.join(output_dir_mllm, "mllm_files_list.txt")
    with open(mllm_list_output_file, 'w') as f:
        for filename in sorted(list(mllm_files_set)):
            name_only = os.path.splitext(filename)[0]
            f.write(name_only + "\n")
    
    print(f"Saved {len(mllm_files_set)} filenames from mllm_json to {mllm_list_output_file}")


    # ==========================================
    # Task 2: Compare with insightface_json
    # Find files in insightface_json NOT in mllm_json
    # ==========================================
    print(f"Scanning insightface_json directory: {insight_json_dir}")
    
    insight_files_set = set()
    if os.path.exists(insight_json_dir):
        insight_files_list = [f for f in os.listdir(insight_json_dir) if f.endswith('.json')]
        insight_files_set = set(insight_files_list)
    else:
        print(f"Warning: Directory {insight_json_dir} does not exist.")

    # Calculate difference: insight - mllm
    missing_files = []
    for f in insight_files_set:
        if f not in mllm_files_set:
            missing_files.append(f)
    
    missing_files_output_path = os.path.join(output_dir_mllm, "missing_in_mllm.txt")
    with open(missing_files_output_path, 'w') as f:
        for filename in sorted(missing_files):
            name_only = os.path.splitext(filename)[0]
            f.write(name_only + "\n")
            
    print(f"Found {len(missing_files)} files present in insightface_json but missing in mllm_json.")
    print(f"Saved missing files list to {missing_files_output_path}")


    # ==========================================
    # Task 3: Analyze mllm_json content
    # Find files where is_qualified == True
    # And files where is_qualified == False (Unqualified)
    # ==========================================
    print("Analyzing mllm_json contents for 'is_qualified' status...")
    
    qualified_files = []
    unqualified_files = []
    processed_count = 0
    error_count = 0
    
    # Iterate through mllm files we found earlier
    total_files = len(mllm_files_set)
    
    for filename in mllm_files_set:
        file_path = os.path.join(mllm_json_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # The structure is expected to be: {"result": "{\"is_qualified\": true, ...}", ...}
            if 'result' in content:
                result_raw = content['result']
                
                # result_raw should be a string containing JSON
                if isinstance(result_raw, str):
                    try:
                        # Sometimes the model might output markdown code blocks like ```json ... ```
                        clean_result = result_raw
                        if clean_result.strip().startswith("```json"):
                            clean_result = clean_result.strip().replace("```json", "").replace("```", "")
                        elif clean_result.strip().startswith("```"):
                             clean_result = clean_result.strip().replace("```", "")
                        
                        result_data = json.loads(clean_result)
                        
                        # Check is_qualified field
                        if result_data.get('is_qualified') is True:
                            qualified_files.append(filename)
                        else:
                            unqualified_files.append(filename)
                            
                    except json.JSONDecodeError:
                        # If parsing inner JSON fails
                        # print(f"JSON Parse Error in result of {filename}")
                        error_count += 1
                elif isinstance(result_raw, dict):
                    # In case it's already a dict
                    if result_raw.get('is_qualified') is True:
                        qualified_files.append(filename)
                    else:
                        unqualified_files.append(filename)
            
        except Exception as e:
            # print(f"Error reading {filename}: {e}")
            error_count += 1
            
        processed_count += 1
        if processed_count % 5000 == 0:
            print(f"Processed {processed_count}/{total_files} files...")

    # Save qualified files list
    qualified_output_path = os.path.join(output_dir_mllm, "qualified_files.txt")
    with open(qualified_output_path, 'w') as f:
        for filename in sorted(qualified_files):
            name_only = os.path.splitext(filename)[0]
            f.write(name_only + "\n")

    # Save unqualified files list
    unqualified_output_path = os.path.join(output_dir_mllm, "unqualified_files.txt")
    with open(unqualified_output_path, 'w') as f:
        for filename in sorted(unqualified_files):
            name_only = os.path.splitext(filename)[0]
            f.write(name_only + "\n")

    print(f"Analysis complete for mllm_json.")
    print(f"Total MLLM files scanned: {processed_count}")
    print(f"Qualified files count: {len(qualified_files)}")
    print(f"Unqualified files count: {len(unqualified_files)}")
    print(f"Failed to parse/read count: {error_count}")
    print(f"Saved qualified files list to {qualified_output_path}")
    print(f"Saved unqualified files list to {unqualified_output_path}")


    # ==========================================
    # Task 4: Analyze mllm_rec_json
    # ==========================================
    print(f"\n--- Analyzing mllm_rec_json ---")
    print(f"Scanning mllm_rec_json directory: {mllm_rec_json_dir}")

    mllm_rec_files_set = set()
    if os.path.exists(mllm_rec_json_dir):
        mllm_rec_files_list = [f for f in os.listdir(mllm_rec_json_dir) if f.endswith('.json')]
        mllm_rec_files_set = set(mllm_rec_files_list)
    else:
        print(f"Warning: Directory {mllm_rec_json_dir} does not exist.")

    mllm_rec_list_output_file = os.path.join(output_dir_rec, "mllm_rec_files_list.txt")
    with open(mllm_rec_list_output_file, 'w') as f:
        for filename in sorted(list(mllm_rec_files_set)):
            name_only = os.path.splitext(filename)[0]
            f.write(name_only + "\n")
    
    print(f"Saved {len(mllm_rec_files_set)} filenames from mllm_rec_json to {mllm_rec_list_output_file}")
    
    # Calculate difference: insight - mllm_rec
    missing_rec_files = []
    # Note: insight_files_set was loaded in Task 2
    if 'insight_files_set' in locals() and insight_files_set:
        for f in insight_files_set:
            if f not in mllm_rec_files_set:
                missing_rec_files.append(f)
    
        missing_rec_files_output_path = os.path.join(output_dir_rec, "missing_in_mllm_rec.txt")
        with open(missing_rec_files_output_path, 'w') as f:
            for filename in sorted(missing_rec_files):
                name_only = os.path.splitext(filename)[0]
                f.write(name_only + "\n")
                
        print(f"Found {len(missing_rec_files)} files present in insightface_json but missing in mllm_rec_json.")
        print(f"Saved missing files list to {missing_rec_files_output_path}")
    else:
        print("InsightFace files not loaded, skipping comparison for mllm_rec_json.")

if __name__ == "__main__":
    main()
