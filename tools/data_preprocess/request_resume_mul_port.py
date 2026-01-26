from openai import OpenAI
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob

#####################################################
# TASK = 'check'  # 过滤低质量
TASK = 'recaption'    # 描述重打标
# TASK = 'chat'

MAX_NUM_REQUESTS = 0
# MAX_NUM_REQUESTS = 1000

PROC_RATIO = 1  # 处理剩余请求的比例 (0.0-1.0)

# 如果存在, 则只处理该列表中的文件
file_list_path = "/NAS0/slurm/home/huangyanwen259/monster/zls/code/vllm/analysis_res/missing_in_mllm.txt"

BASE_PORT = 8000

NUM_PORT = 4

# DATA_ROOT = "/NAS0/slurm/home/huangyanwen259/monster/zls/data/FaceID-6M/laion_512"
DATA_ROOT = "/NAS0/slurm/home/huangyanwen259/monster/zls/data/FaceID-6M/laion_512"

if TASK == 'check':
    OUTPUT_ROOT = os.path.join(DATA_ROOT, "mllm_json")
    with open("/NAS0/slurm/home/huangyanwen259/monster/zls/code/vllm/prompt_only_img.txt", "r") as f:
        prompt_tmplate = f.read()

if TASK == 'recaption':
    OUTPUT_ROOT = os.path.join(DATA_ROOT, "mllm_rec_json")
    with open("/NAS0/slurm/home/huangyanwen259/monster/zls/code/vllm/prompt_recaption.txt", "r") as f:
        prompt_tmplate = f.read()

# resize target size
H,W=512,512

######################################################

# client = OpenAI(
#     api_key="EMPTY",  # VLLM 无需认证密钥, 任意字符串均可
#     base_url="http://localhost:8000/v1"  # 与 VLLM 服务端口一致
# )


def process_single_check_item(json_file, resized_root, prompt_tmplate, H, W):
    try:
        # We assume file stem matches key in bboxes and image name
        file_stem = os.path.splitext(os.path.basename(json_file))[0]
        
        # Check for resized image (png or jpg)
        resized_img_path = os.path.join(resized_root, f"{file_stem}.png")
        if not os.path.exists(resized_img_path):
                resized_img_path_jpg = os.path.join(resized_root, f"{file_stem}.jpg")
                if os.path.exists(resized_img_path_jpg):
                    resized_img_path = resized_img_path_jpg
                else:
                    # print(f"Warning: Resized image not found for {file_stem}")
                    return None

        # img_hw = f"{H}x{W}"
        # p = prompt_tmplate.replace("<IMG_HW_HERE>", img_hw)
        p = prompt_tmplate

        return {
            "prompt": p,
            "image_path": resized_img_path,
            # "image_hw": img_hw,
        }
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None

def process_single_rec_item(json_file, resized_root, captions_list, prompt_tmplate):
    try:
        file_stem = os.path.splitext(os.path.basename(json_file))[0]
        img_path = os.path.join(resized_root, f"{file_stem}.png")
        
        if not os.path.exists(img_path):
            # print(f"Warning: Image not found: {img_path}")
            return None

        try:
            raw_caption = captions_list[int(file_stem)] if int(file_stem) < len(captions_list) else "one person"
        except Exception:
            # print(f"Error getting raw caption for {file_stem}: {e}")
            raw_caption = "one person"

        p = prompt_tmplate.replace("<RAW_CAPTION_HERE>", str(raw_caption))
        return {
            "prompt": p,
            "image_path": img_path,
            "raw_caption": raw_caption
        }
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None

def parse_data(data_root: str, output_root: Optional[str] = None, num_requests: int = 0, processing_ratio: float = 1.0) -> List[Dict[str, Any]]:
    list_ = []
    json_dir = os.path.join(data_root, "insightface_json")
    
    if not os.path.exists(json_dir):
        print(f"Error: json directory not found: {json_dir}")
        return list_

    # Modified to read from specific list if provided
    if os.path.exists(file_list_path):
        print(f"Reading file list from {file_list_path}")
        with open(file_list_path, 'r') as f:
            stems = [line.strip() for line in f if line.strip()]
        json_files = [os.path.join(json_dir, f"{stem}.json") for stem in stems]
    else:
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Found {len(json_files)} json files in {json_dir}")

    # Resume: Filter out already processed files
    json_files = get_pending_files_with_ratio(json_files, output_root, processing_ratio)

    resized_root = os.path.join(data_root, "tgt_preresize")
    if not os.path.exists(resized_root):
        print(f"Warning: Resized root not found: {resized_root}")

    ## Load existing bboxes 
    
    # new_bbox_dir = os.path.join(data_root, 'tgt_preresize_new_bbox')
    # new_bbox_json = os.path.join(new_bbox_dir, f'resized_bbox_{H}_{W}.json')
    # all_new_bboxes = {}
    # if os.path.exists(new_bbox_json):
    #     try:
    #         with open(new_bbox_json, 'r') as f:
    #             all_new_bboxes = json.load(f)
    #         print(f"Loaded {len(all_new_bboxes)} bboxes from {new_bbox_json}")
    #     except Exception as e:
    #         print(f"Error: Failed to load existing bbox json: {e}")
    #         return list_
    # else:
    #     print(f"Error: Bbox file not found: {new_bbox_json}")
    #     return list_

    if num_requests > 0:
        json_files = json_files[:num_requests]
        print(f"限制处理前 {num_requests} 个请求")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_check_item, json_file, resized_root, prompt_tmplate, H, W) for json_file in json_files]
        for future in tqdm(as_completed(futures), total=len(json_files), desc="Generating request data..."):
            result = future.result()
            if result:
                list_.append(result)

    return list_

def parse_data_rec(data_root: str, output_root: Optional[str] = None, num_requests: int = 0, processing_ratio: float = 1.0) -> List[Dict[str, Any]]:
    list_ = []
    captions_list = []
    captions_path = os.path.join(data_root, "captions.jsonl")
    json_dir = os.path.join(data_root, "insightface_json")
    resized_root = os.path.join(data_root, "tgt_preresize")
    if not os.path.exists(json_dir):
        print(f"Error: json directory not found: {json_dir}")
        return list_

    # Modified to read from specific list if provided
    if os.path.exists(file_list_path):
        print(f"Reading file list from {file_list_path}")
        with open(file_list_path, 'r') as f:
            stems = [line.strip() for line in f if line.strip()]
        json_files = [os.path.join(json_dir, f"{stem}.json") for stem in stems]
    else:
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Found {len(json_files)} json files in {json_dir}")

    # Resume: Filter out already processed files
    json_files = get_pending_files_with_ratio(json_files, output_root, processing_ratio)

    if os.path.exists(captions_path):
        print(f"正在读取 captions: {captions_path}")
        with open(captions_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    captions_list.append(item.get("caption", "one person"))
                except Exception as e:
                    print(f"Error parsing line in captions.jsonl: {e}")
        print(f"已加载 {len(captions_list)} 条 caption 数据")
    else:
        print(f"Warning: captions.jsonl not found at {captions_path}")
    
    if num_requests > 0:
        json_files = json_files[:num_requests]
        print(f"限制处理前 {num_requests} 个请求")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_rec_item, json_file, resized_root, captions_list, prompt_tmplate) for json_file in json_files]
        for future in tqdm(as_completed(futures), total=len(json_files), desc="Generating request data..."):
            result = future.result()
            if result:
                list_.append(result)

    return list_


def request_api(client: OpenAI, text: str, image_path: str) -> str:
    """
    调用本地 VLLM 服务 (符合 OpenAPI 规范)
    """
    try:
        response = client.chat.completions.create(
            model="/data/huangyanwen259/pretrain/Qwen/Qwen3-VL-32B-Instruct",  # 使用模型路径, 如通过--served-model-name指定名称需与 VLLM 服务启动时指定的名称一致
            messages= [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {"type": "text", "text": f"{text}"}
                ]}
            ],
            max_tokens=1024,  # 控制生成文本长度
            temperature=0.7,  # 控制生成随机性 (0-1, 越高越随机) 
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"API 调用失败: {str(e)}")


def get_pending_files_with_ratio(json_files: List[str], output_root: Optional[str], processing_ratio: float = 1.0) -> List[str]:
    # Resume: Filter out already processed files
    if output_root and os.path.exists(output_root):
        existing_stems = {os.path.splitext(f)[0] for f in os.listdir(output_root) if f.endswith('.json')}
        json_files = [f for f in json_files if os.path.splitext(os.path.basename(f))[0] not in existing_stems]
        print(f"Resuming... Skipped {len(existing_stems)} already processed files. Remaining: {len(json_files)}")
    
    if processing_ratio < 1.0 and processing_ratio > 0:
        total_files = len(json_files)
        keep_count = int(total_files * processing_ratio)
        
        files_to_process = json_files[:keep_count]
        files_to_save = json_files[keep_count:]
        
        save_path = "remain_request.json"
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(files_to_save, f, ensure_ascii=False, indent=4)
            print(f"Ratio split ({processing_ratio}): Processing {len(files_to_process)}, Saved {len(files_to_save)} to {save_path}")
        except Exception as e:
            print(f"Error saving remaining requests: {e}")
            
        return files_to_process

    return json_files


def process_batch(what_to_do: str, data_root: str, clients: List[OpenAI], output_root: Optional[str] = None, error_log_path: Optional[str] = None, num_requests: int = 0, processing_ratio: float = 1.0) -> list[Dict[str, Any]]:
    if what_to_do not in ['check', 'recaption', 'chat']:
        raise ValueError("Invalid task specified. Choose from 'check', 'recaption', or 'chat'.")
    if what_to_do == 'check':
        tasks = parse_data(data_root, output_root=output_root, num_requests=num_requests, processing_ratio=processing_ratio)
    elif what_to_do == 'recaption':
        tasks = parse_data_rec(data_root, output_root=output_root, num_requests=num_requests, processing_ratio=processing_ratio)
    else:
        raise NotImplementedError("Chat task is not implemented yet.")

            
    future_to_task = {}
    results = []

    with ThreadPoolExecutor(max_workers=64) as executor:
        # 提交任务时, 记录任务的索引 (即提交顺序)
        for i, task in enumerate(tasks):
            client = clients[i % len(clients)]
            future = executor.submit(request_api, client, task["prompt"], task["image_path"])
            # 保存: future -> (任务, 提交顺序索引)
            p = task.pop("prompt", "")
            future_to_task[future] = (task)

        # 提交任务到线程池, 并建立Future对象与任务的映射
        # future_to_task = {executor.submit(request_api, task): task for task in tasks}

        # 使用tqdm展示任务处理进度
        print("收集结果...")
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"doing {TASK}"):
            task = future_to_task[future]
            try:
                res = future.result()
                result = {
                    "original": task,
                    "result": res,
                    # "status": "success",
                    # "task_index": tasks.index(task)
                }
            except Exception as e:
                print(f"任务处理失败: {task['image_path']}")
                with open(error_log_path, "a") as f:
                    f.write(f"{task['image_path']}: {e}\n")
                continue

            file_stem = os.path.splitext(os.path.basename(task["image_path"]))[0]
            if output_root:
                output_file = os.path.join(output_root, f"{file_stem}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            # print(f"结果已保存到 {output_file}")
            results.append(result)

        # 按任务提交顺序(task_index) 重新排序结果
        # results.sort(key=lambda x: x["task_index"])

        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=f'{DATA_ROOT}')
    parser.add_argument('--output_root', type=str, default=f'{OUTPUT_ROOT}')
    parser.add_argument('--num_requests', type=int, default=MAX_NUM_REQUESTS, help='Max number of requests to process, 0 means all')
    parser.add_argument('--processing_ratio', type=float, default=PROC_RATIO, help='Ratio of remaining files to process (0.0-1.0)')
    parser.add_argument('--base_port', type=int, default=BASE_PORT, help='Base port for VLLM services')
    parser.add_argument('--num_ports', type=int, default=NUM_PORT, help='Number of VLLM ports to use')

    args = parser.parse_args()
    if args.output_root:
        os.makedirs(args.output_root, exist_ok=True)
    
    clients = [
        OpenAI(
            api_key="EMPTY", 
            base_url=f"http://localhost:{port}/v1"
        ) for port in range(args.base_port, args.base_port + args.num_ports)
    ]
    print(f"Initialized {len(clients)} clients starting from port {args.base_port}")

    error_log_path = './error_log.txt'
    try:
        results = process_batch(TASK, args.data_root, clients, args.output_root, error_log_path, num_requests=args.num_requests, processing_ratio=args.processing_ratio)
        print(f"处理完成, 共 {len(results)} 个任务")
    except Exception as e:
        print(f"执行失败: {str(e)}")

if __name__ == "__main__":
    main()