import argparse
import os
import shutil
from datetime import datetime

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Generating T2I CompBench dataset using DeepFloyd")
    parser.add_argument(
        "--num_images",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--chunk_idx",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--compbench_category_name",
        type=str,
        default="color",
        choices=["color", "texture", "shape", "spatial"],
    )

    args = parser.parse_args()

    if args.num_images % args.batch_size != 0:
        raise ValueError("--num_images should be divisable by --batch_size")
    if args.chunk_idx < 0 or args.chunk_idx >= args.num_chunks:
        raise ValueError("--chunk_idx should be in range of (0, --num_chunks)")

    return args


def get_prompts_chunk(prompts, num_chunks, chunk_idx):
    prompts_len = len(prompts)

    chunk_size = (prompts_len + num_chunks - 1) // num_chunks

    start_index = chunk_size * chunk_idx
    end_index = min((chunk_idx + 1) * chunk_size, prompts_len)

    print(f"Choosing chunk ({start_index}:{end_index})")
    print(f"First prompt of the chunk: \"{prompts[start_index]}\"")
    print(f"Last prompt of the chunk: \"{prompts[end_index-1]}\"")

    return prompts[start_index:end_index]


if __name__ == '__main__':
    args = parse_args()

    with open(f'T2I-CompBench-dataset/{args.compbench_category_name}.txt', 'r') as f:
        prompts = f.read().splitlines()
        prompts = [p.strip('.') for p in prompts]
    
    prompts_chunk = get_prompts_chunk(prompts, args.num_chunks, args.chunk_idx)

    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
    stage_1.to('cuda')
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    )
    stage_2.to('cuda')

    base_output_dir = f"T2I-CompBench-dataset/{args.compbench_category_name}"

    for prompt in prompts_chunk:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start generation for prompt \"{prompt}\"")

        prompt_directory = os.path.join(base_output_dir, f"{prompt}/deepfloyd")
        if os.path.exists(prompt_directory):
            images_count = len([file for file in os.listdir(prompt_directory) if file.endswith(".png")])
            if images_count == args.num_images:
                print(f"Directory {prompt_directory} has already {args.num_images} images. Skipping...")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finish for prompt \"{prompt}\"")
                continue
            print(f"Removing existing directory [with {images_count} images]: {prompt_directory}")
            shutil.rmtree(prompt_directory)
        os.makedirs(prompt_directory)
        
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt, num_images_per_prompt=args.batch_size)
        for i in range(args.num_images // args.batch_size):
            stage_1_output = stage_1(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                output_type="pt",
            ).images
            stage_2_output = stage_2(
                image=stage_1_output,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                output_type="pt",
            ).images
            stage_2_out_imgs = pt_to_pil(stage_2_output)

            for j, img in enumerate(stage_2_out_imgs):
                img_number = i*args.batch_size + j
                img.save(os.path.join(prompt_directory, f'{prompt}_{img_number:06d}.png'))

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finish generation for prompt \"{prompt}\"")
