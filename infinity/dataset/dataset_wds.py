from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, get_h_div_w_template2indices, h_div_w_templates

import webdataset as wds
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import numpy as np
import PIL.Image as PImage
import io
from PIL.ImageOps import exif_transpose

def pad_image_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    new_img = PImage.new("RGB", (max_side, max_side), (0, 0, 0))
    paste_position = ((max_side - width) // 2, (max_side - height) // 2)
    new_img.paste(img, paste_position)
    return new_img


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1)

def transform_face(pil_img, target_size=224):
    pil_img = pil_img.resize((target_size, target_size), resample=PImage.LANCZOS)   # 人脸会变形，我觉得还是要在原图中找到人脸区域然后外扩crop成正方形
    arr = np.array(pil_img)
    im = to_tensor(arr)
    return im.add(im).add_(-1)

def preprocess_idp(sample):
    # size = 224
    # image_transforms = transforms.Compose(
    #         [
    #             transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    #             transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ]
    #     )
    src, tgt, prompt = sample
    h, w = dynamic_resolution_h_w[h_div_w_template][PN]['pixel']
    src_img = PImage.open(io.BytesIO(src)).convert('RGB')
    tgt_img = PImage.open(io.BytesIO(tgt)).convert('RGB')
    # 处理EXIF信息，保证图像方向正确
    # src_img_t = exif_transpose(src_img)
    # tgt_img_t = exif_transpose(tgt_img)

    
    ################ debug ###########################
    if DEBUG_VISUALIZE:
        import os
        import time
        os.makedirs('debug_images', exist_ok=True)
        timestamp = int(time.time() * 1000000)
        src_img.save(f'debug_images/src_{timestamp}.jpg')
        tgt_img.save(f'debug_images/tgt_{timestamp}.jpg')
        print(f"Saved debug images: debug_images/src_{timestamp}.jpg")

    src_img = transform_face(src_img)
    tgt_img = transform(tgt_img, h, w)
    instruction = prompt.decode('utf-8')
    return src_img, tgt_img, instruction
    ######################################################

def preprocess(sample):
    src, tgt, prompt = sample
    h, w = dynamic_resolution_h_w[h_div_w_template][PN]['pixel']
    src_img = PImage.open(io.BytesIO(src)).convert('RGB')
    tgt_img = PImage.open(io.BytesIO(tgt)).convert('RGB').resize((src_img.size))
    src_img = transform(src_img, h, w)
    tgt_img = transform(tgt_img, h, w)
    instruction = prompt.decode('utf-8')
    return src_img, tgt_img, instruction

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
    dataset = wds.WebDataset(
        urls,
        nodesplitter=wds.shardlists.split_by_node,
        shardshuffle=True,
        resampled=True,
        cache_size=buffersize,
        handler=wds.handlers.warn_and_continue,
    ).with_length(overall_length).shuffle(100).to_tuple("src.jpg", "tgt.jpg", "txt").map(preprocess).batched(batch_size, partial=False).with_epoch(100000)
    # ).with_length(overall_length).shuffle(100).to_tuple("src.jpg", "tgt.jpg", "txt").map(preprocess_idp).batched(batch_size, partial=False).with_epoch(100000)
    return dataset

if __name__ == '__main__':
    # from infinity.utils.arg_util import Args
    # args = Args(explicit_bool=True).parse_args(known_only=True)
    # dataset = WDS_Train_Dataset(
    #         data_path="/data1/zls/code/AR/VAR_IDP/data/FaceID-70K/webdataset", 
    #         buffersize=args.iterable_data_buffersize,
    #         pn=args.pn,
    #         batch_size=args.batch_size,
    #     )
    DEBUG_VISUALIZE = True
    
    # 能否处理人脸位置偏僻的问题？
    from tools.face_utils import general_face_preserving_resize
    # 裁剪tgt：检测人脸，以人脸为中心裁剪
    
    dataset = WDS_Train_Dataset(
        data_path='/data1/zls/code/AR/VAR_IDP/data/FaceID-70K/webdataset',
        buffersize=10000,
        pn='0.06M',
        batch_size=4,
    )
    ld_train = wds.WebLoader(dataset=dataset, num_workers=0, pin_memory=True, generator=None, batch_size=None, prefetch_factor=None)
    dataset = iter(ld_train)
    for src_img, tgt_img, instruction in dataset:
        pass
        print(src_img.shape, tgt_img.shape, instruction)
        break