import os
import json
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import io
import argparse


def process_faceid_dataset(
    base_path="",
    output_path="",
    samples_per_tar=1024,
    sample_num=None,
    ):
    base_path = Path(base_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define subdirectories
    embeddings_dir = base_path / "embeddings"
    insightface_dir = base_path / "insightface_json"
    square_face_dir = base_path / "square_face" 
    captions_path = base_path / "captions.jsonl"
    recaption_dir = base_path / "mllm_rec_json"
    if not insightface_dir.exists():
        raise FileNotFoundError(f"insightface_json not found at {insightface_dir}")
    
    # Get list of json files
    print(f"正在查找 json 文件: {insightface_dir}")
    # json_files = sorted(list(insightface_dir.glob("*.json")))
    json_files = list(insightface_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个 json 文件")
    
    # Load captions
    captions_list = []
    if captions_path.exists():
        print(f"正在读取 captions: {captions_path}")
        with open(captions_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    captions_list.append(item)
                except Exception as e:
                    print(f"Error parsing line in captions.jsonl: {e}")
        print(f"已加载 {len(captions_list)} 条 caption 数据")
    else:
        print(f"Warning: captions.jsonl not found at {captions_path}")
    
    # Load unqualified images list
    unqualified_set = set()
    unqualified_path = Path(__file__).parent / "unqualified_images_list.json"
    if unqualified_path.exists():
        print(f"正在读取 unqualified images list: {unqualified_path}")
        with open(unqualified_path, 'r', encoding='utf-8') as f:
            try:
                unqualified_list = json.load(f)
                unqualified_set = set(unqualified_list)
                print(f"已加载 {len(unqualified_set)} 个不合格图像名")
            except Exception as e:
                print(f"Error parsing unqualified_images_list.json: {e}")
    else:
        print(f"Warning: unqualified_images_list.json not found at {unqualified_path}")
    
    # 准备输出
    tar_file = None
    tar_writer = None
    current_tar_idx = 0
    sample_count = 0
    total_samples = 0
    
    # 汇总信息 jsonl
    summary_jsonl_path = output_path / "summary.jsonl"
    summary_file = open(summary_jsonl_path, 'w', encoding='utf-8')
    
    # 存储 tar 文件信息用于生成 FaceID.txt
    tar_info = {}  # {tar_index: sample_count}
    
    def create_new_tar():
        nonlocal tar_file, tar_writer, current_tar_idx, sample_count
        if tar_writer is not None:
            tar_writer.close()
            tar_info[current_tar_idx] = sample_count
            print(f"\n已完成 tar 文件: wds_faceid_{current_tar_idx:04d}.tar，包含 {sample_count} 个样本")
        
        tar_filename = output_path / f"wds_faceid_{current_tar_idx:04d}.tar"
        tar_writer = tarfile.open(tar_filename, 'w')
        sample_count = 0
        current_tar_idx += 1
        return tar_writer
    
    # 创建第一个 tar 文件
    tar_writer = create_new_tar()
    
    # 处理每一行数据
    print("\n开始处理数据...")
    skipped_count = 0
    side_face_count = 0
    unqualified_count = 0
    
    for json_file in tqdm(json_files, desc="处理样本"):
        # 如果达到指定的样本数量，停止处理
        if sample_num is not None and total_samples >= sample_num:
            break
        
        stem = json_file.stem
        
        # 检查是否为不合格图像
        if stem in unqualified_set:
            unqualified_count += 1
            continue
        
        # 检查 is_side_face
        try:
            with open(json_file, 'r') as f:
                meta_data = json.load(f)
        except Exception as e:
            print(f"读取 {json_file} 出错: {e}")
            skipped_count += 1
            continue

        # 默认35度以上侧脸跳过
        # if meta_data.get('is_side_face', 0) == 1:
        #     side_face_count += 1
        #     continue
        pose = meta_data.get('pose', None)
        yaw_threshold = 60  # 侧脸阈值
        if pose is not None:
            pitch, yaw, roll = pose
            if abs(yaw) > yaw_threshold:
                side_face_count += 1
                continue

        # 定位文件
        # tgt_img: 尝试 jpg 和 png
        tgt_img_path = base_path / f"{stem}.jpg"
        if not tgt_img_path.exists():
            tgt_img_path = base_path / f"{stem}.png"
            
        square_face_path = square_face_dir / f"{stem}.png"
        if not square_face_path.exists():
             square_face_path = square_face_dir / f"{stem}.jpg"

        embedding_path = embeddings_dir / f"{stem}.npy"
        mllm_rec_json_path = recaption_dir / f"{stem}.json"

        # 检查文件是否存在
        if not (tgt_img_path.exists() and square_face_path.exists() and embedding_path.exists() and mllm_rec_json_path.exists()):
            skipped_count += 1
            continue
        
        try:
            # 读取文件
            with open(tgt_img_path, 'rb') as f:
                tgt_bytes = f.read()
            with open(square_face_path, 'rb') as f:
                square_bytes = f.read()
            with open(embedding_path, 'rb') as f:
                emb_bytes = f.read()
            
            # 验证图片是否可以正常打开
            Image.open(io.BytesIO(tgt_bytes)).convert('RGB')
            Image.open(io.BytesIO(square_bytes)).convert('RGB')
            
            # 准备 webdataset 样本
            sample_key = f"{total_samples:08d}"
            
            # 1. src_img (square_face)
            sq_ext = square_face_path.suffix.lower()
            sq_info = tarfile.TarInfo(name=f"{sample_key}.src{sq_ext}")
            sq_info.size = len(square_bytes)
            tar_writer.addfile(sq_info, io.BytesIO(square_bytes))
            
            # 2. tgt_img
            tgt_ext = tgt_img_path.suffix.lower()
            tgt_info = tarfile.TarInfo(name=f"{sample_key}.tgt{tgt_ext}")
            tgt_info.size = len(tgt_bytes)
            tar_writer.addfile(tgt_info, io.BytesIO(tgt_bytes))
            
            # 3. embedding (src_img)
            emb_info = tarfile.TarInfo(name=f"{sample_key}.npy")
            emb_info.size = len(emb_bytes)
            tar_writer.addfile(emb_info, io.BytesIO(emb_bytes))
            
            # 4. text (caption)
            text = 'one person'
            height = 0
            width = 0
            
            if stem.isdigit():
                idx = int(stem)
                if idx < len(captions_list):
                    caption_data = captions_list[idx]
                    text = caption_data.get('caption', 'one person')
                    height = caption_data.get('height', 0)
                    width = caption_data.get('width', 0)
                else:
                    print(f"Warning: No caption data for index {idx}")
            else:
                print(f"Warning: {stem} is not a digit, cannot retrieve caption")
            text_bytes = text.encode('utf-8')
            txt_info = tarfile.TarInfo(name=f"{sample_key}.txt")
            txt_info.size = len(text_bytes)
            tar_writer.addfile(txt_info, io.BytesIO(text_bytes))

            # 5. mllm_rec_json
            with open(mllm_rec_json_path, 'r', encoding='utf-8') as f:
                mllm_rec_bytes = f.read().encode('utf-8')
            mllm_rec_info = tarfile.TarInfo(name=f"{sample_key}.mllm_rec.json")
            mllm_rec_info.size = len(mllm_rec_bytes)
            tar_writer.addfile(mllm_rec_info, io.BytesIO(mllm_rec_bytes))

            # 6. filename (metadata) 
            meta = {
                "file_name": stem,
                "original_file": str(tgt_img_path.name),
                "H": height,
                "W": width,
                "bbox_before_resize": meta_data.get("modified_bboxes", []),
            }
            meta_bytes = json.dumps(meta).encode('utf-8')
            json_info = tarfile.TarInfo(name=f"{sample_key}.meta.json")
            json_info.size = len(meta_bytes)
            tar_writer.addfile(json_info, io.BytesIO(meta_bytes))

            # 写入汇总信息
            summary_entry = {
                'sample_id': total_samples,
                'file_name': stem,
                # 'tgt_path': str(tgt_img_path),
                # 'square_path': str(square_face_path),
                # 'embedding_path': str(embedding_path),
                'tar_file': f"wds_faceid_{current_tar_idx-1:04d}.tar"
            }
            summary_file.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')
            
            sample_count += 1
            total_samples += 1
            
            # 如果当前 tar 已满，创建新的
            if sample_count >= samples_per_tar:
                tar_writer = create_new_tar()
        
        except Exception as e:
            print(f"\n处理样本 {stem} 时出错: {e}")
            skipped_count += 1
            continue
    
    # 关闭最后一个 tar 文件
    if tar_writer is not None:
        tar_writer.close()
        tar_info[current_tar_idx - 1] = sample_count
        print(f"\n已完成 tar 文件: wds_faceid_{current_tar_idx-1:04d}.tar，包含 {sample_count} 个样本")
    
    summary_file.close()
    
    # 生成 FaceID.txt
    faceid_txt_path = output_path / "FaceID.txt"
    with open(faceid_txt_path, 'w') as f:
        total_tar_samples = sum(tar_info.values())
        num_tars = len(tar_info)
        # 格式: file_name\tlength\tshard_num
        f.write(f"faceid\t{total_tar_samples}\t{num_tars}\n")
    
    print(f"\n" + "="*50)
    print(f"处理完成！")
    print(f"总样本数: {total_samples}")
    print(f"跳过样本数: {skipped_count}")
    print(f"侧脸跳过数: {side_face_count}")
    print(f"不合格图像跳过数: {unqualified_count}")
    print(f"生成 tar 文件数: {len(tar_info)}")
    print(f"输出路径: {output_path}")
    print(f"FaceID.txt 已生成: {faceid_txt_path}")
    print(f"汇总信息已保存: {summary_jsonl_path}")
    print("="*50)


def main():
    BASE_ROOT = '/data1/zls/code/AR/VAR_IDP/assets'
    OUTPUT_ROOT = '/data1/zls/code/AR/VAR_IDP/assets/webdataset'
    NUM_SAMPLES_PER_TAR = 1024
    SAMPLE_NUM = None  # 设置为 None 则处理所有数据

    parser = argparse.ArgumentParser(description='处理 FaceID 数据集为 webdataset 格式')
    parser.add_argument('--base_path',type=str,default='',help='assets 文件夹路径')
    parser.add_argument('--output_path',type=str,default='',help='输出的 webdataset 文件夹路径')
    parser.add_argument('--samples_per_tar',type=int,default=1024,help='每个 tar 包中的样本数量')
    parser.add_argument('--sample_num',type=int,help='要打包的样本总数，不指定则处理所有数据')
    args = parser.parse_args()

    args.base_path = BASE_ROOT if args.base_path == '' else args.base_path
    args.output_path = OUTPUT_ROOT if args.output_path == '' else args.output_path
    args.samples_per_tar = NUM_SAMPLES_PER_TAR if args.samples_per_tar == 1024 else args.samples_per_tar
    args.sample_num = SAMPLE_NUM if args.sample_num is None else args.sample_num
    
    process_faceid_dataset(
        base_path=args.base_path,
        output_path=args.output_path,
        samples_per_tar=args.samples_per_tar,
        sample_num=args.sample_num
    )

if __name__ == "__main__":
    main()
    