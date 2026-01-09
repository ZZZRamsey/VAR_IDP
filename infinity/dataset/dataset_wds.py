from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, get_h_div_w_template2indices, h_div_w_templates
from tools.face_utils import general_face_preserving_resize
import webdataset as wds
import torch
import json
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import numpy as np
import PIL.Image as PImage
import io
from PIL.ImageOps import exif_transpose
import os
from tqdm import tqdm

def pad_image_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    new_img = PImage.new("RGB", (max_side, max_side), (0, 0, 0))
    paste_position = ((max_side - width) // 2, (max_side - height) // 2)
    new_img.paste(img, paste_position)
    return new_img


def transform(pil_img, tgt_h, tgt_w, file_name):
    """
    Convert PIL Image to normalized tensor.
    Assumes input image is already properly sized (e.g., from general_face_preserving_resize).
    
    If resize/crop is needed, it will be performed.
    Normalize from [0, 1] to [-1, 1] range: (2*x) - 1
    """
    width, height = pil_img.size
    
    # Only resize/crop if needed
    if width != tgt_w or height != tgt_h:
        # Resize while preserving aspect ratio
        if width / height <= tgt_w / tgt_h:
            resized_width = tgt_w
            resized_height = int(tgt_w / (width / height))
        else:
            resized_height = tgt_h
            resized_width = int((width / height) * tgt_h)
        pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
        # pil_img.save(f"{os.path.dirname(__file__)}/debug_src_resize_{file_name}.png")  # For debugging
        # pil_img.save(f"{os.path.dirname(__file__)}/debug_tgt_resize_{file_name}.png")  # For debugging

        # Center crop to target size
        arr = np.array(pil_img)
        crop_y = (arr.shape[0] - tgt_h) // 2
        crop_x = (arr.shape[1] - tgt_w) // 2
        im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    else:
        # pil_img.save(f"{os.path.dirname(__file__)}/debug_src_{file_name}.png")  # For debugging
        # pil_img.save(f"{os.path.dirname(__file__)}/debug_tgt_{file_name}.png")  # For debugging

        # Already correct size, just convert to tensor
        im = to_tensor(np.array(pil_img))
    
    # Normalize to [-1, 1] range
    return im.add(im).add_(-1)

def preprocess(sample):
    src, tgt, txt, emb, mllm_rec, meta = sample
    h, w = dynamic_resolution_h_w[h_div_w_template][PN]['pixel']
    
    src_img = PImage.open(io.BytesIO(src)).convert('RGB')
    tgt_img = PImage.open(io.BytesIO(tgt)).convert('RGB')

    rec_json = json.loads(mllm_rec.decode('utf-8'))
    meta_json = json.loads(meta.decode('utf-8'))
    
    if tgt_img.size[0] == h and tgt_img.size[1] == w:
        pass
    else:
        # Resize tgt_img with face preservation
        bboxes = meta_json.get('bbox_before_resize', [])
        file_name = meta_json.get('file_name', 'unknown')
        if bboxes:
            if isinstance(bboxes[0], (int, float)): bboxes = [bboxes]
            tgt_img_res, new_bbox = general_face_preserving_resize(tgt_img, bboxes, target_size=h)
            if tgt_img_res is not None:
                tgt_img = tgt_img_res
            else:
                print(f"### {file_name} ### Warning: face preserving resize failed, using normal resize.")
                tgt_img = pad_image_to_square(tgt_img)
        else:
            tgt_img = pad_image_to_square(tgt_img)

    src_img = transform(src_img, h, w, file_name)
    tgt_img = transform(tgt_img, h, w, file_name)
    
    face_emb = np.load(io.BytesIO(emb))
    face_emb_tensor = torch.from_numpy(face_emb)
    
    instruction = txt.decode('utf-8')
    
    return src_img, tgt_img, face_emb_tensor, instruction, rec_json, meta_json

def WDS_Train_Dataset(
    data_path,
    buffersize,
    pn,
    batch_size,
):
    urls = []
    overall_length = 0

    with open(f"{data_path}/FaceID.txt", "r") as file:
        info_file = file.readlines()
    urls_base = f"{data_path}/<FILE>"
    data_file = []
    for item in info_file:
        file_name, length, shard_num = item.strip('\n').split('\t')
        length, shard_num = int(length), int(shard_num)
        for shard in range(shard_num):
            data_file.append(f"wds_{file_name}_{shard:=04d}.tar")
        overall_length += length
    urls += [urls_base.replace("<FILE>", file) for file in data_file]

    # with open(f"{data_path}/MultiID.txt", "r") as file:
    #     info_file = file.readlines()
    # urls_base = "MULTIID_DATA_SHARD_BASE"
    # data_file = []
    # for item in info_file:
    #     file_name, length, shard_num = item.strip('\n').split('\t')
    #     length, shard_num = int(length), int(shard_num)
    #     for shard in range(shard_num):
    #         data_file.append(f"wds_{file_name}_{shard:=04d}.tar")   # 将 shard 变量格式化为 4 位数字，不足 4 位时用零填充。 1 --> 0001  123 --> 0123 12345 --> 12345（超过4位不变）
    #     overall_length += length
    # urls += [urls_base.replace("<FILE>", file) for file in data_file]

    global PN
    PN = pn
    global h_div_w_template
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(1.0 - h_div_w_templates))]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    # 自动计算覆盖全量数据所需的步数
    steps_per_epoch = overall_length // (batch_size * world_size)
    dataset = wds.WebDataset(
        urls,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.split_by_worker,
        shardshuffle=True,
        resampled=True,
        handler=wds.handlers.warn_and_continue,
    ).with_length(overall_length).shuffle(buffersize) \
        .to_tuple("src.png", "tgt.png", "txt", "npy", "mllm_rec.json", "meta.json") \
        .map(preprocess) \
        .batched(batch_size, partial=False) \
        .with_epoch(steps_per_epoch)
    return dataset

def tgt_preresize(data_path, tgt_size=1024):
    from pathlib import Path
    import concurrent.futures

    h, w = tgt_size, tgt_size
    skipped_count = 0
    side_face_count = 0
    new_bbox_dict = {}
    insightface_dir = os.path.join(data_path, "insightface_json")
    if not os.path.exists(insightface_dir):
        print(f"Error: {insightface_dir} does not exist.")
        return
    json_files = list(Path(insightface_dir).glob("*.json"))

    tgt_resized_dir = os.path.join(data_path, 'tgt_preresize')
    os.makedirs(tgt_resized_dir, exist_ok=True)
    new_bbox_dir = os.path.join(data_path, 'tgt_preresize_new_bbox')
    os.makedirs(new_bbox_dir, exist_ok=True)
    new_bbox_json = os.path.join(new_bbox_dir, f'resized_bbox_{h}_{w}.json')

    def process_one_file(json_file):
        stem = json_file.stem
        local_skipped = 0
        local_side_face = 0
        local_new_bbox = None

        # 1. 检查 is_side_face
        try:
            with open(json_file, 'r') as f:
                meta_data = json.load(f)
        except Exception as e:
            print(f"读取 {json_file} 出错: {e}")
            local_skipped = 1
            return stem, local_skipped, local_side_face, local_new_bbox
            
        pose = meta_data.get('pose', None)
        yaw_threshold = 60  # 侧脸阈值
        if pose is not None:
            pitch, yaw, roll = pose
            if abs(yaw) > yaw_threshold:
                local_side_face = 1
                return stem, local_skipped, local_side_face, local_new_bbox

        # 2. resize
        tgt_img_path = os.path.join(data_path, f"{stem}.png")
        if not os.path.exists(tgt_img_path):
            tgt_img_path = os.path.join(data_path, f"{stem}.jpg")
        if not os.path.exists(tgt_img_path):
            local_skipped = 1
            return stem, local_skipped, local_side_face, local_new_bbox

        try:
            with PImage.open(tgt_img_path) as img:
                img = exif_transpose(img).convert('RGB')
                bboxes = meta_data.get('modified_bboxes', [])
                if bboxes:
                    if isinstance(bboxes[0], (int, float)): bboxes = [bboxes]
                    img_res, new_bbox = general_face_preserving_resize(img, bboxes, target_size=h)
                    # save new_bbox to new json
                    if new_bbox is not None:
                        local_new_bbox = new_bbox
                    if img_res is not None:
                        img = img_res
                    else:
                        print(f"### {stem} ### Warning: face preserving resize failed, using normal resize.")
                        img = pad_image_to_square(img)
                else:
                    img = pad_image_to_square(img)
                
                width, height = img.size
    
                # Only resize/crop if needed
                if width != w or height != h:
                    # Resize while preserving aspect ratio
                    if width / height <= w / h:
                        resized_width = w
                        resized_height = int(w / (width / height))
                    else:
                        resized_height = h
                        resized_width = int((width / height) * h)
                    img = img.resize((resized_width, resized_height), resample=PImage.LANCZOS)

                    # Center crop to target size
                    arr = np.array(img)
                    crop_y = (arr.shape[0] - h) // 2
                    crop_x = (arr.shape[1] - w) // 2
                    arr = arr[crop_y: crop_y + h, crop_x: crop_x + w]
                    img = PImage.fromarray(arr)
                else:
                    pass
                save_path = os.path.join(tgt_resized_dir, f"{stem}.png")
                img.save(save_path)
        except Exception as e:
            print(f"处理图像 {tgt_img_path} 出错: {e}")
            local_skipped = 1
            
        return stem, local_skipped, local_side_face, local_new_bbox

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_one_file, json_files), total=len(json_files), desc="处理JSON文件"))

    for stem, skipped, side_face, new_bbox in results:
        skipped_count += skipped
        side_face_count += side_face
        if new_bbox is not None:
            new_bbox_dict[stem] = new_bbox
    
    # save new_bbox_dict to json
    with open(new_bbox_json, 'w') as f:
        json.dump(new_bbox_dict, f, indent=4)

    print(f"预处理完成。跳过的图像数量: {skipped_count}，侧脸图像数量: {side_face_count}")

def main():
    # from infinity.utils.arg_util import Args
    # args = Args(explicit_bool=True).parse_args(known_only=True)
    # dataset = WDS_Train_Dataset(
    #         data_path="/data1/zls/code/AR/VAR_IDP/data/FaceID-70K/webdataset", 
    #         buffersize=args.iterable_data_buffersize,
    #         pn=args.pn,
    #         batch_size=args.batch_size,
    #     )
    
    dataset = WDS_Train_Dataset(
        # data_path='/data1/zls/code/AR/VAR_IDP/data/FaceID-70K/webdataset',
        data_path='/data1/zls/code/AR/VAR_IDP/assets/webdataset',
        buffersize=5000,
        pn='0.06M',
        batch_size=7,
    )
    print(len(dataset))
    ld_train = wds.WebLoader(dataset=dataset, num_workers=0, pin_memory=True, generator=None, batch_size=None, prefetch_factor=None)
    dataset = iter(ld_train)
    for src_img, tgt_img, face_emb_tensor, instruction, rec_json, meta_json in dataset:
        # print(src_img.shape, tgt_img.shape, face_emb_tensor.shape, instruction)
        pass
        # break

# PYTHONPATH=/data1/zls/code/AR/VAR_IDP python infinity/dataset/dataset_wds.py
if __name__ == '__main__':
    # main()
    
    data_path = '/data1/zls/code/AR/VAR_IDP/assets/'
    # data_path = '/data1/zls/code/AR/VAR_IDP/data/FaceID-6M/laion_512'   # 等 insightface 预处理完成后再运行
    # tgt_preresize(data_path)
    