import random
import torch

import cv2
import numpy as np
from tools.run_infinity import *


def load_2B(pretrain_root, model_path):
    vae_path=f'{pretrain_root}/infinity/infinity_vae_d32reg.pth'
    text_encoder_ckpt=f'{pretrain_root}/flan-t5-xl'
    args=argparse.Namespace(
        pn='0.25M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_2b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=1,
        bf16=1,
        enable_model_cache=0,
    )

    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    return args, infinity, vae, text_tokenizer, text_encoder


def load_8B(pretrain_root, model_path):
    vae_path=f'{pretrain_root}/infinity/infinity_vae_d56_f8_14_patchify.pth'
    text_encoder_ckpt=f'{pretrain_root}/flan-t5-xl'
    args=argparse.Namespace(
        pn='0.25M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=14,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_8b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=1,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=0,
        bf16=1,
        enable_model_cache=0,
    )

    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    return args, infinity, vae, text_tokenizer, text_encoder


def load(pretrain_root, model_path, model_size='2B'):
    if model_size == '2B':
        return load_2B(pretrain_root, model_path)
    elif model_size == '8B':
        return load_8B(pretrain_root, model_path)
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    

def infer(args, src_img_path, tgt_img_path, instruction):
    from infinity.dataset.webdataset import transform
    
    h, w = 512, 512
    if args.pn == '0.06M':
        h, w = 256, 256

    with open(src_img_path, 'rb') as f:
        src_img: PImage.Image = PImage.open(f)
        src_img = src_img.convert('RGB')
        src_img_3HW = transform(src_img, h, w)

    cfg = 4
    tau = 0.5
    h_div_w = 1/1 # aspect ratio, height:width
    enable_positive_prompt=0
    save_file = tgt_img_path

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        instruction,
        src_img_3HW,
        g_seed=args.seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
        apply_spatial_patchify=args.apply_spatial_patchify,
    )
    os.makedirs(osp.dirname(osp.abspath(save_file)), exist_ok=True)
    cv2.imwrite(save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(save_file)}')


if __name__ == "__main__":
    pretrain_root = "PRETRAIN_ROOT"
    model_path = "MODEL_PATH"
    model_size = "MODEL_SIZE"  # "2B" or "8B"
    src_img_path = "SOURCE_IMG_PATH"
    tgt_img_path = "TARGET_IMG_PATH"
    instruction = "INSTRUCTION"

    args, infinity, vae, text_tokenizer, text_encoder = load(pretrain_root, model_path, model_size)
    infer(args, src_img_path, tgt_img_path, instruction)