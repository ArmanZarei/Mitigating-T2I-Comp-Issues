import argparse
import os
from datetime import datetime
from typing import Tuple

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, EMAModel

from evaluation import generate_samples_and_evaluate_blip_vqa
from text_emb_projection_models import (
    CLIPTextEmbeddingLinearProjector,
    CLIPTextEmbeddingLinearSkipProjector,
    CLIPTextEmbeddingMLPProjector,
    WindowAwareLinearProjection
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained text embedding projectors on T2I CompBench dataset (validation set)")
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
        "--stable_diffusion_checkpoint",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        choices=["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"]
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--compbench_category_name",
        type=str,
        default="color",
        choices=["color", "texture", "shape"],
    )
    parser.add_argument(
        "--projection_checkpoint",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--early_guidance_timestep_threshold",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512
    )
    parser.add_argument(
        "--evaluation_batch_size",
        type=int,
        default=10
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    
    args = parser.parse_args()

    if args.chunk_idx < 0 or args.chunk_idx >= args.num_chunks:
        raise ValueError("--chunk_idx should be in range of (0, --num_chunks)")
    
    if args.early_guidance_timestep_threshold != -1 and (args.early_guidance_timestep_threshold < 0 or args.early_guidance_timestep_threshold > 1000):
        print("[Warning] --early_guidance_timestep_threshold should be in range of (0, 1000) or -1")
    
    if args.clip_checkpoint is None:
        print(f"args.clip_checkpoint is None. Default would use {args.stable_diffusion_checkpoint} subfolder=text_encoder")

    return args


def get_list_chunk(arr: list, num_chunks: int, chunk_idx: int) -> list:
    arr_len = len(arr)

    chunk_size = (arr_len + num_chunks - 1) // num_chunks

    start_index = chunk_size * chunk_idx
    end_index = min((chunk_idx + 1) * chunk_size, arr_len)

    print(f"Choosing chunk ({start_index}:{end_index})")
    print(f"First item of the chunk: \"{arr[start_index]}\"")
    print(f"Last item of the chunk: \"{arr[end_index-1]}\"", flush=True)

    return arr[start_index:end_index]


def get_text_embeddings(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel) -> torch.Tensor:
    text_input = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]
    
    return text_embeddings


def load_models(args) -> Tuple[AutoencoderKL, CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, PNDMScheduler]:
    vae = AutoencoderKL.from_pretrained(args.stable_diffusion_checkpoint, subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.stable_diffusion_checkpoint, subfolder="tokenizer")
    if args.clip_checkpoint is None:
        text_encoder = CLIPTextModel.from_pretrained(args.stable_diffusion_checkpoint, subfolder="text_encoder", use_safetensors=True)
    else:
        text_encoder = CLIPTextModel.from_pretrained(args.clip_checkpoint, use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(args.stable_diffusion_checkpoint, subfolder="unet", use_safetensors=True)
    scheduler = PNDMScheduler.from_pretrained(args.stable_diffusion_checkpoint, subfolder="scheduler")
        
    vae.to('cuda')
    text_encoder.to('cuda')
    unet.to('cuda');

    num_inference_steps = 25
    scheduler.set_timesteps(num_inference_steps)

    return vae, tokenizer, text_encoder, unet, scheduler


if __name__ == '__main__':
    args = parse_args()

    with open(f'T2I-CompBench-dataset/{args.compbench_category_name}_val.txt', 'r') as f:
        prompts = f.read().splitlines()
        prompts = [p.strip('.') for p in prompts]
        prompts = sorted(set(prompts))
    
    prompts_chunk = get_list_chunk(prompts, args.num_chunks, args.chunk_idx)

    # Initialization of the models
    vae, tokenizer, text_encoder, unet, scheduler = load_models(args)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False);

    text_embedding_projector = torch.load(args.projection_checkpoint).to('cuda')
  
    for prompt in prompts_chunk:
        print("="*100)
        print(f"[Start of Generation and Evaluation] prompt: {prompt}")
        print(f"[Data and Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        prompt_directory_path = os.path.join(args.output_dir, prompt)

        if os.path.isfile(os.path.join(prompt_directory_path, 'vqa_result.json')):
            print("!"*100)
            print(f"!!! Skipping prompt \"{prompt}\"")
            print("!!! Generation and Evaluation has already been done!")
            print("!"*100, flush=True)
            continue

        # Getting the CLIP's text embedding for the prompt 
        with torch.no_grad():
            fixed_text_embeddings = text_embedding_projector(get_text_embeddings(prompt, tokenizer, text_encoder)).detach()
            clean_fixed_text_embeddings = get_text_embeddings(prompt, tokenizer, text_encoder).detach() if args.early_guidance_timestep_threshold != -1 else None

        _, average_score = generate_samples_and_evaluate_blip_vqa(
            vae,
            unet,
            scheduler,
            tokenizer,
            text_encoder,
            prompt=[prompt],
            fixed_text_embeddings=fixed_text_embeddings,
            evaluation_path=prompt_directory_path,
            batch_size=args.evaluation_batch_size,
            num_evaluation_images=100,
            image_size=args.image_size,
            clean_fixed_text_embeddings=clean_fixed_text_embeddings,
            early_guidance_timestep_threshold=args.early_guidance_timestep_threshold,
            seed=args.seed
        )

        print(f"[Finished Generation and Evaluation] Prompt: {prompt}")
        print(f"[Finished Generation and Evaluation] Average Score: {average_score}")

        print(f"[Data and Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Finish] prompt: {prompt}")
        print("="*100, flush=True)
