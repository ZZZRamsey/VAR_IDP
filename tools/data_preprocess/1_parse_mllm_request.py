import json
import os
import glob
import re

def parse_mllm_results(json_dir):
    unqualified_images = []
    
    # Get all json files in the directory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} json files in {json_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get the result string
            result_str = data.get("result", "")
            if not result_str:
                print(f"Warning: No result found in {json_file}")
                continue
                
            # Try to parse the result string directly
            clean_json_str = result_str.strip()
            
            # Remove markdown code blocks if present
            # clean_json_str = re.sub(r'^```json\s*|\s*```$', '', result_str.strip(), flags=re.MULTILINE)
            
            try:
                result_data = json.loads(clean_json_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try extracting valid JSON object
                try:
                    start = clean_json_str.find('{')
                    end = clean_json_str.rfind('}')
                    if start != -1 and end != -1:
                        result_data = json.loads(clean_json_str[start:end+1])
                    else:
                        raise ValueError("No JSON content found")
                except Exception as e:
                    print(f"Error parsing JSON in result of {json_file}: {e}")
                    continue
                
            #################################################################
            # Check if qualified
            is_stitched = result_data.get("is_stitched", False)
            is_blurry_or_distorted = result_data.get("is_blurry_or_distorted", False)
            is_qualified = result_data.get("is_qualified", False)
            is_face_truncated_or_edge = result_data.get("is_face_truncated_or_edge", True)
            is_identity_clear = result_data.get("is_identity_clear", False)
            
            good_img = not is_stitched and not is_blurry_or_distorted and is_qualified and not is_face_truncated_or_edge and is_identity_clear
            #################################################################

            if not good_img:
                # Get image path from original task data
                original_task = data.get("original", {})
                image_path = original_task.get("image_path", "")
                if image_path:
                    image_name = os.path.basename(image_path).split(".")[0]
                    unqualified_images.append(image_name)
                    
                    # Optional: Print reason
                    analysis = result_data.get("analysis", "No analysis provided")
                    print(f"Unqualified: {image_name} - Reason: {analysis}")
                else:
                    print(f"Warning: No image path found in original task of {json_file}")

        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
            continue
            
    return unqualified_images

def main():
    # Directory containing the MLLM output JSONs
    # Based on previous context, this is likely in data/assert/mllm_json
    # Adjust this path if necessary
    mllm_json_dir = "/data1/zls/code/AR/VAR_IDP/assets/mllm_json"
    
    if not os.path.exists(mllm_json_dir):
        print(f"Directory not found: {mllm_json_dir}")
        return

    unqualified_paths = parse_mllm_results(mllm_json_dir)
    
    print("\n" + "="*50)
    print(f"Total Unqualified Images: {len(unqualified_paths)}")
    print("="*50)
    
    # Print the array of paths
    print("Unqualified Image Paths:")
    print(json.dumps(unqualified_paths, indent=4, ensure_ascii=False))
    
    # Save to a file
    output_file = "unqualified_images_list.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unqualified_paths, f, indent=4, ensure_ascii=False)
    print(f"\nList saved to {output_file}")

if __name__ == "__main__":
    main()
