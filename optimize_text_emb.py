import argparse
import json
import os
from datetime import datetime
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, EMAModel

from evaluation import generate_samples_and_evaluate_blip_vqa


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize text embedding for better compositionality")
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
        "--generator_directory_names",
        type=str,
        nargs='+',
        default=["syngen", "deepfloyd", "sd-v1-4"]
    )
    parser.add_argument(
        "--stable_diffusion_checkpoint",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        choices=["CompVis/stable-diffusion-v1-4"]
    )
    parser.add_argument(
        "--compbench_category_name",
        type=str,
        default="color",
        choices=["color", "texture", "shape"],
    )

    args = parser.parse_args()

    if args.chunk_idx < 0 or args.chunk_idx >= args.num_chunks:
        raise ValueError("--chunk_idx should be in range of (0, --num_chunks)")

    return args


def get_training_dataloader(prompt_directory_path: str, args: argparse.Namespace) -> DataLoader:
    missing_generators = set(args.generator_directory_names) - set(os.listdir(prompt_directory_path))
    if len(missing_generators) != 0:
        raise Exception(f"Some generators are missing in the directory [{', '.join(missing_generators)}]")
    
    generators_vqa_results = {}

    for generator_directory_name in args.generator_directory_names:
        vqa_result_path = os.path.join(prompt_directory_path, generator_directory_name, 'vqa_result.json')
        if not os.path.isfile(vqa_result_path):
            raise Exception(f"VQA Result is missing: \"{vqa_result_path}\"")
        
        with open(vqa_result_path) as f:
            d = json.load(f)
            for k in d:
                generators_vqa_results[os.path.join(generator_directory_name, k)] = float(d[k])
    
    sorted_generators_vqa_results = sorted(generators_vqa_results.items(), key=lambda x: -x[1])

    top_generators_vqa_results = sorted_generators_vqa_results[:30]

    dataset = load_dataset(
        prompt_directory_path, data_files={"train": [x[0] for x in top_generators_vqa_results]}, split='train'
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}


    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    return train_dataloader


def get_text_embeddings(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel) -> torch.Tensor:
    text_input = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]
    
    return text_embeddings


def train(
    train_dataloader: DataLoader,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: PNDMScheduler,
    learnable_text_embedding: torch.nn.Parameter,
    ema_learnable_text_embedding: EMAModel,
    optimizer: torch.optim.Adam,
    opt_scheduler: torch.optim.lr_scheduler.LRScheduler
) -> Tuple[torch.nn.Parameter, EMAModel]:
    for epoch in range(100):
        epoch_loss = 0.
        for batch in train_dataloader:
            latents = vae.encode(batch["images"].to('cuda')).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)

            batch_size = latents.shape[0]

            # timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size//2 + 1, ), device=latents.device)
            timesteps = torch.cat([timesteps, scheduler.config.num_train_timesteps - timesteps - 1], dim=0)[:batch_size]
            timesteps = timesteps.long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            target = noise # noise_scheduler.config.prediction_type = "epsilon"

            encoder_hidden_states = learnable_text_embedding.expand(batch_size, -1, -1)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learnable_text_embedding, 1.);
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size

        opt_scheduler.step()   
        
        ema_learnable_text_embedding.step(learnable_text_embedding)

        epoch_loss /= len(train_dataloader.dataset)

        print(f"[Epoch {epoch:2d}] Loss: {epoch_loss}", flush=True)
    
    return learnable_text_embedding, ema_learnable_text_embedding


def load_models(args) -> Tuple[AutoencoderKL, CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, PNDMScheduler]:
    vae = AutoencoderKL.from_pretrained(args.stable_diffusion_checkpoint, subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.stable_diffusion_checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.stable_diffusion_checkpoint, subfolder="text_encoder", use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(args.stable_diffusion_checkpoint, subfolder="unet", use_safetensors=True)
    scheduler = PNDMScheduler.from_pretrained(args.stable_diffusion_checkpoint, subfolder="scheduler")
        
    vae.to('cuda')
    text_encoder.to('cuda')
    unet.to('cuda');

    num_inference_steps = 25
    scheduler.set_timesteps(num_inference_steps)

    return vae, tokenizer, text_encoder, unet, scheduler


def get_list_chunk(arr: list, num_chunks: int, chunk_idx: int) -> list:
    arr_len = len(arr)

    chunk_size = (arr_len + num_chunks - 1) // num_chunks

    start_index = chunk_size * chunk_idx
    end_index = min((chunk_idx + 1) * chunk_size, arr_len)

    print(f"Choosing chunk ({start_index}:{end_index})")
    print(f"First item of the chunk: \"{arr[start_index]}\"")
    print(f"Last item of the chunk: \"{arr[end_index-1]}\"", flush=True)

    return arr[start_index:end_index]


if __name__ == '__main__':
    args = parse_args()

    dataset_base_path = f"./T2I-CompBench-dataset/{args.compbench_category_name}"

    with open(f'T2I-CompBench-dataset/{args.compbench_category_name}.txt', 'r') as f:
        prompts = f.read().splitlines()
        prompts = [p.strip('.') for p in prompts]
        prompts = sorted(set(prompts))
    
    assert len(set(os.listdir(dataset_base_path)).intersection(prompts)) == len(prompts)

    prompts_chunk = get_list_chunk(prompts, args.num_chunks, args.chunk_idx)

    # Initialization of the models
    vae, tokenizer, text_encoder, unet, scheduler = load_models(args)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False);

    for prompt in prompts_chunk:
        print("="*100)
        print(f"[Start of Training] prompt: {prompt}")
        print(f"[Data and Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        prompt_directory_path = os.path.join(dataset_base_path, prompt)

        if os.path.isfile(os.path.join(prompt_directory_path, 'learnable_text_embedding', 'vqa_result.json')):
            print("!"*100)
            print(f"!!! Skipping prompt \"{prompt}\"")
            print("!!! Training and Evaluation has already been done!")
            print("!"*100, flush=True)
            continue

        # Picking some "good" samples for training
        try:
            train_dataloader = get_training_dataloader(prompt_directory_path, args)
        except Exception as error:
            print("!"*100)
            print(f"!!! Failed for prompt \"{prompt}\". Error: {error}")
            print("!"*100, flush=True)
            continue

        print(f"[Training Setup] Training Dataset Size: {len(train_dataloader.dataset)}")
        print(f"[Training Setup] Batch Size: {train_dataloader.batch_size}")
        
        # Getting the CLIP's text embedding for the prompt 
        text_embeddings = get_text_embeddings(prompt, tokenizer, text_encoder)

        # Making the text embedding learnable + creating the EMA Model
        learnable_text_embedding = torch.nn.Parameter(text_embeddings.detach(), requires_grad=True)
        ema_learnable_text_embedding = EMAModel(learnable_text_embedding)
        ema_learnable_text_embedding.to('cuda')

        # Optimizer and Scheduler used for optimizing the learnable text embedding
        optimizer = torch.optim.Adam([learnable_text_embedding], lr=1e-1)
        opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 75], gamma=0.1)

        # Training the learnable text embedding
        learnable_text_embedding, ema_learnable_text_embedding = train(
            train_dataloader,
            vae,
            unet,
            scheduler,
            learnable_text_embedding,
            ema_learnable_text_embedding,
            optimizer,
            opt_scheduler
        )
        
        # Saving the learned text embedding + the EMA version
        torch.save(learnable_text_embedding, os.path.join(prompt_directory_path, 'learnable_text_embedding.pth'))
        torch.save(ema_learnable_text_embedding.state_dict(), os.path.join(prompt_directory_path, 'ema_learnable_text_embedding.pth'))

        print(f"[Start of Evaluation] prompt: {prompt}", flush=True)

        # Evaluation of the learned text embedding by generating 100 samples and calculating the BLIP VQA score
        _, average_score = generate_samples_and_evaluate_blip_vqa(
            vae,
            unet,
            scheduler,
            tokenizer,
            text_encoder,
            prompt=[prompt],
            fixed_text_embeddings=learnable_text_embedding.data.detach(),
            evaluation_path=os.path.join(prompt_directory_path, 'learnable_text_embedding'),
            batch_size=10,
            num_evaluation_images=100,
        )

        print(f"[Finished Evaluation] Prompt: {prompt}")
        print(f"[Finished Evaluation] Average Score: {average_score}")

        print(f"[Data and Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[Finish] prompt: {prompt}")
        print("="*100, flush=True)
