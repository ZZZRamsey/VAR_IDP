import os
import json
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import io


def process_faceid_dataset(
    base_path="/data1/zls/code/AR/VAREdit_FaceID/data/FaceID-70K/laion_1024",
    output_path="/data1/zls/code/AR/VAREdit_FaceID/data/FaceID-70K/webdataset",
    samples_per_tar=1000,
    sample_num=200
):
    """
    将 FaceID-70K 数据集处理成 webdataset 格式
    
    Args:
        base_path: laion_1024 文件夹路径
        output_path: 输出的 webdataset 文件夹路径
        samples_per_tar: 每个 tar 包中的样本数量
        sample_num: 要打包的三元组总数，None 表示处理所有数据
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取 captions.jsonl
    captions_file = base_path / "captions.jsonl"
    if not captions_file.exists():
        raise FileNotFoundError(f"captions.jsonl not found at {captions_file}")
    
    print("正在读取 captions.jsonl...")
    captions_data = []
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                captions_data.append(json.loads(line))
    
    print(f"总共读取了 {len(captions_data)} 条 caption 数据")
    
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
            print(f"\n已完成 tar 文件: wds_laion_1024_{current_tar_idx:04d}.tar，包含 {sample_count} 个样本")
        
        tar_filename = output_path / f"wds_laion_1024_{current_tar_idx:04d}.tar"
        tar_writer = tarfile.open(tar_filename, 'w')
        sample_count = 0
        current_tar_idx += 1
        return tar_writer
    
    # 创建第一个 tar 文件
    tar_writer = create_new_tar()
    
    # 处理每一行数据
    print("\n开始处理数据...")
    skipped_count = 0
    
    for idx, caption_data in enumerate(tqdm(captions_data, desc="处理样本")):
        # 如果达到指定的样本数量，停止处理
        if sample_num is not None and total_samples >= sample_num:
            break
        
        # 获取对应的文件名
        tgt_png = base_path / f"{idx}.png"
        src_png = base_path / "face" / f"{idx}.png"
        npy_file = base_path / f"{idx}.npy"
        
        # 检查文件是否存在
        if not tgt_png.exists():
            skipped_count += 1
            continue
        
        if not src_png.exists():
            skipped_count += 1
            continue
        
        # 提取需要的字段
        caption = caption_data.get('caption', '')
        height = caption_data.get('height', 0)
        width = caption_data.get('width', 0)
        
        if not caption:
            skipped_count += 1
            continue
        
        try:
            # 读取图片
            with open(src_png, 'rb') as f:
                src_img_bytes = f.read()
            
            with open(tgt_png, 'rb') as f:
                tgt_img_bytes = f.read()
            
            # 验证图片是否可以正常打开
            Image.open(io.BytesIO(src_img_bytes)).convert('RGB')
            Image.open(io.BytesIO(tgt_img_bytes)).convert('RGB')
            
            # 准备 webdataset 样本
            sample_key = f"{total_samples:08d}"
            
            # 添加到 tar 文件
            # src.jpg
            src_info = tarfile.TarInfo(name=f"{sample_key}.src.jpg")
            src_info.size = len(src_img_bytes)
            tar_writer.addfile(src_info, io.BytesIO(src_img_bytes))
            
            # tgt.jpg
            tgt_info = tarfile.TarInfo(name=f"{sample_key}.tgt.jpg")
            tgt_info.size = len(tgt_img_bytes)
            tar_writer.addfile(tgt_info, io.BytesIO(tgt_img_bytes))
            
            # txt (caption)
            caption_bytes = caption.encode('utf-8')
            txt_info = tarfile.TarInfo(name=f"{sample_key}.txt")
            txt_info.size = len(caption_bytes)
            tar_writer.addfile(txt_info, io.BytesIO(caption_bytes))
            
            # 写入汇总信息
            summary_entry = {
                'sample_id': total_samples,
                'src_filename': f"{idx}.png",
                'tgt_filename': f"{idx}.png",
                'src_path': str(src_png),
                'tgt_path': str(tgt_png),
                'height': height,
                'width': width,
                'caption': caption,
                'tar_file': f"wds_laion_1024_{current_tar_idx-1:04d}.tar"
            }
            summary_file.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')
            
            sample_count += 1
            total_samples += 1
            
            # 如果当前 tar 已满，创建新的
            if sample_count >= samples_per_tar:
                tar_writer = create_new_tar()
        
        except Exception as e:
            print(f"\n处理样本 {idx} 时出错: {e}")
            skipped_count += 1
            continue
    
    # 关闭最后一个 tar 文件
    if tar_writer is not None:
        tar_writer.close()
        tar_info[current_tar_idx - 1] = sample_count
        print(f"\n已完成 tar 文件: wds_laion_1024_{current_tar_idx-1:04d}.tar，包含 {sample_count} 个样本")
    
    summary_file.close()
    
    # 生成 FaceID.txt
    faceid_txt_path = output_path / "FaceID.txt"
    with open(faceid_txt_path, 'w') as f:
        total_tar_samples = sum(tar_info.values())
        num_tars = len(tar_info)
        # 格式: file_name\tlength\tshard_num
        f.write(f"laion_1024\t{total_tar_samples}\t{num_tars}\n")
    
    print(f"\n" + "="*50)
    print(f"处理完成！")
    print(f"总样本数: {total_samples}")
    print(f"跳过样本数: {skipped_count}")
    print(f"生成 tar 文件数: {len(tar_info)}")
    print(f"输出路径: {output_path}")
    print(f"FaceID.txt 已生成: {faceid_txt_path}")
    print(f"汇总信息已保存: {summary_jsonl_path}")
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='处理 FaceID-70K 数据集为 webdataset 格式')
    parser.add_argument(
        '--base_path',
        type=str,
        default='/data1/zls/code/AR/VAREdit_FaceID/data/FaceID-70K/laion_1024',
        help='laion_1024 文件夹路径'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/data1/zls/code/AR/VAREdit_FaceID/data/FaceID-70K/webdataset',
        help='输出的 webdataset 文件夹路径'
    )
    parser.add_argument(
        '--samples_per_tar',
        type=int,
        default=1000,
        help='每个 tar 包中的样本数量'
    )
    parser.add_argument(
        '--sample_num',
        type=int,
        # default=200,
        help='要打包的三元组总数，不指定则处理所有数据'
    )
    
    args = parser.parse_args()
    
    process_faceid_dataset(
        base_path=args.base_path,
        output_path=args.output_path,
        samples_per_tar=args.samples_per_tar,
        sample_num=args.sample_num
    )
