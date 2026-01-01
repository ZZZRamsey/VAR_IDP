import os
import shutil
from pathlib import Path

def move_npy_files(source_dir: str = "", target_dir: str = "", recursive: bool = False):
    """
    将指定目录下的.npy文件移动到目标目录
    :param source_dir: 源目录（默认当前目录）
    :param target_dir: 目标目录（默认./landmark）
    :param recursive: 是否递归移动子文件夹中的.npy文件（默认False）
    """
    # 1. 初始化路径
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 2. 创建目标目录（不存在则创建，存在则不报错）
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 3. 查找所有.npy文件
    npy_files = []

    # 仅查找当前目录下的.npy文件
    npy_files = list(source_path.glob("*.npy"))
    
    if not npy_files:
        print("未找到任何.npy文件！")
        return
    
    # 4. 批量移动文件
    moved_count = 0
    skipped_count = 0
    for npy_file in npy_files:
        # 构造目标文件路径
        target_file = target_path / npy_file.name
        
        # 避免覆盖已存在的文件（可选：如需覆盖，直接删除该判断）
        if target_file.exists():
            print(f"跳过已存在的文件：{npy_file.name}")
            skipped_count += 1
            continue
        
        # 移动文件
        try:
            shutil.move(str(npy_file), str(target_file))
            print(f"成功移动：{npy_file.name}")
            moved_count += 1
        except Exception as e:
            print(f"移动失败：{npy_file.name}，错误信息：{e}")
            skipped_count += 1
    
    # 5. 打印统计信息
    print("=" * 50)
    print(f"任务完成！")
    print(f"找到.npy文件总数：{len(npy_files)}")
    print(f"成功移动数量：{moved_count}")
    print(f"跳过/失败数量：{skipped_count}")
    print("=" * 50)

# -------------------------- 用法示例 --------------------------
if __name__ == "__main__":
    # 示例1：仅移动当前目录下的.npy文件（默认配置）
    move_npy_files("/data1/zls/code/AR/VAR_IDP/data/FaceID-6M/laion_512","/data1/zls/code/AR/VAR_IDP/data/FaceID-6M/laion_512/landmark")
    
    # 示例2：递归移动当前目录及所有子文件夹中的.npy文件
    # move_npy_files(recursive=True)
    
    # 示例3：指定源目录和目标目录
    # move_npy_files(source_dir="./data", target_dir="./landmark_data")