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
    tgt_preresize_dir = base_path / "tgt_preresize"
    bbox_dir = base_path / "tgt_preresize_new_bbox"
    # captions_path = base_path / "captions.jsonl"
    recaption_dir = base_path / "mllm_rec_json"
    if not insightface_dir.exists():
        raise FileNotFoundError(f"insightface_json not found at {insightface_dir}")
    
    # Load qualified images list
    qualified_set = set()
    qualified_path = Path(__file__).parent / "qualified_files.txt"
    files_to_process = []
    
    if qualified_path.exists():
        print(f"正在读取 qualified images list: {qualified_path}")
        with open(qualified_path, 'r', encoding='utf-8') as f:
            try:
                qualified_list = [line.strip() for line in f if line.strip()]
                qualified_set = set(qualified_list)
                files_to_process = sorted(list(qualified_set))
                print(f"已加载 {len(qualified_set)} 个合格图像名")
            except Exception as e:
                print(f"Error parsing qualified_files.txt: {e}")
    else:
        print(f"Warning: qualified_files.txt not found at {qualified_path}")
        print(f"Fallback: 从 insightface_json 中获取文件列表")
        json_files = list(insightface_dir.glob("*.json"))
        files_to_process = sorted([f.stem for f in json_files])
        qualified_set = set(files_to_process)        
        print(f"准备处理 {len(files_to_process)} 个文件")
    
    # # Load captions
    # captions_list = []
    # if captions_path.exists():
    #     print(f"正在读取 captions: {captions_path}")
    #     with open(captions_path, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             try:
    #                 item = json.loads(line)
    #                 captions_list.append(item)
    #             except Exception as e:
    #                 print(f"Error parsing line in captions.jsonl: {e}")
    #     print(f"已加载 {len(captions_list)} 条 caption 数据")
    # else:
    #     print(f"Warning: captions.jsonl not found at {captions_path}")
    
    # Load all preprocess bboxes
    all_bboxes = {}
    if bbox_dir.exists():
        print(f"正在读取 bbox json files from: {bbox_dir}")
        for bbox_file in bbox_dir.glob("*.json"):
            try:
                with open(bbox_file, 'r', encoding='utf-8') as f:
                    cur_bboxes = json.load(f)
                    if isinstance(cur_bboxes, dict):
                        # Merge dictionaries
                        all_bboxes.update(cur_bboxes)
                    else:
                        print(f"Warning: Unexpected format in {bbox_file}, top level is not dict")
            except Exception as e:
                print(f"Error parsing bbox file {bbox_file}: {e}")
        print(f"已加载 {len(all_bboxes)} 条 bbox 数据")
    else:
        print(f"Warning: bbox directory not found at {bbox_dir}")
    
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
    print("\n开始处理数据(多线程)...")
    skipped_count = 0
    side_face_count = 0
    unqualified_count = 0

    import concurrent.futures

    def process_one_file(stem):
        # 检查是否为不合格图像
        # if stem not in qualified_set:
        #    return 'unqualified', None
        
        # 对 insightface 结果进行处理（比如侧脸过滤等）
        # try:
        #     with open(json_file, 'r') as f:
        #         meta_data = json.load(f)
        # except Exception as e:
        #     # print(f"读取 {json_file} 出错: {e}")
        #     return 'skipped', None

        # 默认35度以上侧脸跳过
        # pose = meta_data.get('pose', None)
        # yaw_threshold = 60  # 侧脸阈值
        # if pose is not None:
        #     pitch, yaw, roll = pose
        #     if abs(yaw) > yaw_threshold:
        #         return 'side_face', None

        # 定位文件
        # tgt_img: 尝试 jpg 和 png
        tgt_img_path = tgt_preresize_dir / f"{stem}.png"
        if not tgt_img_path.exists():
            tgt_img_path = tgt_preresize_dir / f"{stem}.jpg"
            
        embedding_path = embeddings_dir / f"{stem}.npy"
        mllm_rec_json_path = recaption_dir / f"{stem}.json"
        
        # 验证 insightface json 是否存在 (虽然目前未使用其内容)
        # insightface_path = insightface_dir / f"{stem}.json"
        # if not insightface_path.exists():
        #     return 'skipped', None

        # 检查文件是否存在
        # Check if stem is in all_bboxes
        if not (tgt_img_path.exists() and embedding_path.exists() and mllm_rec_json_path.exists()):
            return 'skipped', None
        
        # Get bbox from loaded dict
        bbox_list = all_bboxes.get(stem)
        if not bbox_list:
            # print(f"Warning: No bbox found for {stem}")
            return 'skipped', None
        
        try:
            # 1. tgt_img
            with open(tgt_img_path, 'rb') as f:
                tgt_bytes = f.read()
            # 2. embedding
            with open(embedding_path, 'rb') as f:
                emb_bytes = f.read()
            
            # 3. square_face
            # 验证图片是否可以正常打开并crop
            try:
                with Image.open(io.BytesIO(tgt_bytes)) as img:
                    img = img.convert('RGB')
                    
                    # bbox_list structure: [[x1, y1, x2, y2], ...] or just [x1, y1, x2, y2]
                    # User sample: "VCG...": [[357, 138, 876, 657]]
                    if isinstance(bbox_list, list) and len(bbox_list) > 0:
                        # Assuming first one if list of lists
                        bbox = bbox_list[0]
                        if isinstance(bbox, list) and len(bbox) >= 4:
                            pass
                        elif isinstance(bbox, (int, float)) and len(bbox_list) >= 4:
                             # It was a flat list [x,y,w,h]
                             bbox = bbox_list
                        else:
                             return 'skipped', None
                    else:
                         return 'skipped', None
                    
                    # 截取并 resize
                    crop = img.crop(tuple(bbox))
                    crop = crop.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    buf = io.BytesIO()
                    crop.save(buf, format='PNG')
                    square_bytes = buf.getvalue()
                    
            except Exception:
                return 'skipped', None

            # # 4. text (caption)
            # text = 'one person'
            # # height = 0
            # # width = 0
            
            # if stem.isdigit():
            #     idx = int(stem)
            #     if idx < len(captions_list):
            #         caption_data = captions_list[idx]
            #         text = caption_data.get('caption', 'one person')
            #         # height = caption_data.get('height', 0)
            #         # width = caption_data.get('width', 0)
            #     else:
            #         print(f"Warning: No caption data for index {idx}")
            # else:
            #     print(f"Warning: {stem} is not a digit, cannot retrieve caption")

            # 5. mllm_rec_json
            try:
                with open(mllm_rec_json_path, 'r', encoding='utf-8') as f:
                    rec_json_data = json.load(f)
                    result_str = rec_json_data.get('result', '')
                    
                    # 尝试解析 result 字段（它是一个 JSON 字符串）
                    result_val = ''
                    if result_str:
                        try:
                            # 清理可能的 markdown 代码块标记
                            clean_result = result_str.strip()
                            if clean_result.startswith("```json"):
                                clean_result = clean_result.replace("```json", "").replace("```", "").strip()
                            elif clean_result.startswith("```"):
                                clean_result = clean_result.replace("```", "").strip()
                            
                            # 解析为字典
                            result_dict = json.loads(clean_result)
                            # 提取 refined_caption
                            result_val = result_dict.get('refined_caption', '')
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            # 如果解析失败，尝试从字符串中提取
                            if "refined_caption" in result_str:
                                # 粗略提取：取 refined_caption 之后的部分，并去除可能的 JSON 标点
                                result_val = result_str.split("refined_caption")[-1].lstrip('": ').rstrip('"}')
                            else:
                                result_val = result_str
                    
                    mllm_rec_bytes = json.dumps(result_val, ensure_ascii=False).encode('utf-8')
            except Exception as e:
                print(f"Error reading mllm rec json {stem}: {e}")
                mllm_rec_bytes = b"{one person}"

            res_data = {
                'stem': stem,
                'tgt_bytes': tgt_bytes,
                'tgt_ext': tgt_img_path.suffix.lower(),
                'square_bytes': square_bytes,
                'sq_ext': '.png',
                'emb_bytes': emb_bytes,
                'mllm_rec_bytes': mllm_rec_bytes,
                # 'text': text,
                # 'height': height,
                # 'width': width,
                'reso_match_bbox': bbox,
                'original_file_name': tgt_img_path.name
            }
            return 'success', res_data

        except Exception as e:
            # print(f"\n处理样本 {stem} 时出错: {e}")
            return 'skipped', None

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # 使用 map 保持顺序，或者可以乱序？这里保持顺序比较好，虽然 WebDataset 会 shuffle
        results = executor.map(process_one_file, files_to_process)
        
        for status, data in tqdm(results, total=len(files_to_process), desc="处理样本"):
            # 如果达到指定的样本数量，停止处理
            if sample_num is not None and total_samples >= sample_num:
                break
            
            if status == 'unqualified':
                unqualified_count += 1
                continue
            elif status == 'side_face':
                side_face_count += 1
                continue
            elif status == 'skipped':
                skipped_count += 1
                continue
            elif status == 'success' and data is not None:
                try:
                    # 准备 webdataset 样本
                    sample_key = f"{total_samples:08d}"
                    stem = data['stem']
                    
                    # 1. src_img (square_face)
                    sq_info = tarfile.TarInfo(name=f"{sample_key}.src{data['sq_ext']}")
                    sq_info.size = len(data['square_bytes'])
                    tar_writer.addfile(sq_info, io.BytesIO(data['square_bytes']))
                    
                    # 2. tgt_img
                    tgt_info = tarfile.TarInfo(name=f"{sample_key}.tgt{data['tgt_ext']}")
                    tgt_info.size = len(data['tgt_bytes'])
                    tar_writer.addfile(tgt_info, io.BytesIO(data['tgt_bytes']))
                    
                    # 3. embedding (src_img)
                    emb_info = tarfile.TarInfo(name=f"{sample_key}.npy")
                    emb_info.size = len(data['emb_bytes'])
                    tar_writer.addfile(emb_info, io.BytesIO(data['emb_bytes']))
                    
                    # # 4. text (original caption)
                    # text_bytes = data['text'].encode('utf-8')
                    # txt_info = tarfile.TarInfo(name=f"{sample_key}.txt")
                    # txt_info.size = len(text_bytes)
                    # tar_writer.addfile(txt_info, io.BytesIO(text_bytes))

                    # 5. mllm_rec_json
                    mllm_rec_info = tarfile.TarInfo(name=f"{sample_key}.mllm_rec.json")
                    mllm_rec_info.size = len(data['mllm_rec_bytes'])
                    tar_writer.addfile(mllm_rec_info, io.BytesIO(data['mllm_rec_bytes']))

                    # 6. filename (metadata) 
                    meta = {
                        # "file_name": stem,
                        "original_file": data['original_file_name'],
                        # "H": data['height'],
                        # "W": data['width'],
                        "reso_match_bbox": data['reso_match_bbox'],
                    }
                    meta_bytes = json.dumps(meta).encode('utf-8')
                    json_info = tarfile.TarInfo(name=f"{sample_key}.meta.json")
                    json_info.size = len(meta_bytes)
                    tar_writer.addfile(json_info, io.BytesIO(meta_bytes))

                    # 写入汇总信息
                    summary_entry = {
                        'sample_id': total_samples,
                        'file_name': stem,
                        'tar_file': f"wds_faceid_{current_tar_idx-1:04d}.tar"
                    }
                    summary_file.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')
                    
                    sample_count += 1
                    total_samples += 1
                    
                    # 如果当前 tar 已满，创建新的
                    if sample_count >= samples_per_tar:
                        tar_writer = create_new_tar()

                except Exception as e:
                    print(f"\n写入 Tar 样本 {stem} 时出错: {e}")
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
    # 注意当前路径下如果有qualified_files.txt则只打包其中的文件名
    BASE_ROOT = '/fs-ift/atlas/zouyuefeng/zls/code/VAR_IDP/data/FaceID-6M/laion_512'
    OUTPUT_ROOT = '/fs-ift/atlas/zouyuefeng/zls/code/VAR_IDP/data/FaceID-6M/512_webdataset_81390'
    NUM_SAMPLES_PER_TAR = 2 ** 14   # 16384
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
    