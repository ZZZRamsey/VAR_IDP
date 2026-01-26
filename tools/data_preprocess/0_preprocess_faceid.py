import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
import concurrent.futures

# Add project root to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from tools.face_utils import FaceExtractor

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_embedding(path, embedding):
    np.save(path, embedding)

def save_image(path, img):
    img.save(path)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/fs-ift/atlas/zouyuefeng/zls/code/VAR_IDP/data/FaceID-6M/laion_512')
    # parser.add_argument('--data_root', type=str, default='/data1/zls/code/AR/VAR_IDP/assets')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    embeddings_dir = data_root / 'embeddings'
    square_face_dir = data_root / 'square_face'
    insightface_json_dir = data_root / 'insightface_json'
    
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    square_face_dir.mkdir(parents=True, exist_ok=True)
    insightface_json_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing FaceExtractor...")

    face_extractor = FaceExtractor()

    # Collect all image files
    image_files = []
    print(f"Scanning files in {data_root}...")
    
    # Only scan png files in the current directory, excluding subdirectories
    if data_root.exists() and data_root.is_dir():
        for file_path in data_root.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.png':
            # if file_path.is_file() and file_path.suffix.lower() == '.jpg':
                image_files.append(str(file_path))
    
    print(f"Found {len(image_files)} images.")

    metadata = {}
    
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    
    print("Processing images...")
    error_log_path = data_root / 'error_log.txt'

    for img_path in tqdm(image_files):
        try:
            img_name = os.path.basename(img_path)
            file_stem = os.path.splitext(img_name)[0]
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                with open(error_log_path, "a") as f:
                    f.write(f"{img_name}: {e}\n")
                continue
            
            results = face_extractor.extract_refs(image)
            
            if results[0] is None:
                continue
                
            res_list, ref_imgs, embeddings, bboxes, modified_bboxes, pose_list = results
            
            if not res_list:
                continue

            best_idx = -1
            max_area = -1
            
            for i, bbox in enumerate(bboxes):
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > max_area:
                    max_area = area
                    best_idx = i
            
            if best_idx == -1:
                continue
                
            best_res = res_list[best_idx]
            best_ref_img = ref_imgs[best_idx]
            best_embedding = embeddings[best_idx]
            best_pose = pose_list[best_idx]
            
            metadata_entry = best_res.copy()
            
            if 'embedding' in metadata_entry:
                metadata_entry.pop('embedding')
            metadata_entry['modified_bboxes'] = modified_bboxes[best_idx]
            is_side = face_extractor.is_side_face(best_pose)
            metadata_entry['is_side_face'] = 1 if is_side else 0
            
            emb_path = embeddings_dir / f"{file_stem}.npy"
            io_executor.submit(save_embedding, emb_path, best_embedding)
            
            img_save_path = square_face_dir / f"{file_stem}.png" 
            io_executor.submit(save_image, img_save_path, best_ref_img)
            
            json_save_path = insightface_json_dir / f"{file_stem}.json"
            io_executor.submit(save_json, json_save_path, metadata_entry)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            with open(error_log_path, "a") as f:
                f.write(f"{os.path.basename(img_path)}: {e}\n")
            continue

    print("Waiting for IO tasks to complete...")
    io_executor.shutdown(wait=True)
        
    print("Done.")

if __name__ == '__main__':
    main()