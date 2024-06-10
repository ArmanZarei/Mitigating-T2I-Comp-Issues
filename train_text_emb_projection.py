import argparse
import json
import math
import os
from datetime import datetime
from typing import Tuple

import PIL
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset, Image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDPMScheduler,
    EMAModel,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from evaluation import generate_samples_and_evaluate_blip_vqa
from text_emb_projection_models import (
    CLIPTextEmbeddingLinearProjector,
    CLIPTextEmbeddingMLPProjector,
    CLIPTextEmbeddingLinearSkipProjector, 
    WindowAwareLinearProjection
)


def parse_args():
    
    parser = argparse.ArgumentParser(description="Train MLP on top of CLIP text encoder")

    parser.add_argument(
        "--stable_diffusion_checkpoint",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        choices=["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"]
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=["a red backpack and a blue book", "a blue bench and a green cake", "a red book and a yellow vase", "a bathroom with green tile and a red shower curtain", "a white car and a red sheep", "a red dog and a brown orange", "a brown banana and a green cow", "a yellow apple and red bananas", "a brown frog and a green pond", "A green scooter is parked near a curb in front of a blue vintage car."],
        nargs="+",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--projector_type",
        type=str,
        default='linear',
        choices=['linear', 'linear_and_skip', 'mlp', 'window_aware_linear']
    )
    parser.add_argument(
        "--linear_projection_initialization_type",
        type=str,
        default='eye',
        help='used only if projector_type is linear',
        choices=['zeros', 'eye', 'default', 'xavier']
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
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
        "--train_batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--lr_scheduler_decay_steps",
        type=int,
        nargs="+",
        default=[10000, 16000],
    )
    parser.add_argument(
        "--projection_window_size",
        type=int,
        default=None
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True
    )

    args = parser.parse_args()

    if args.stable_diffusion_checkpoint == "CompVis/stable-diffusion-v1-4":
        assert args.image_size == 512
    elif args.stable_diffusion_checkpoint == "stabilityai/stable-diffusion-2-1":
        assert args.image_size == 768   
    else:
        raise Exception("Not handled yet!")
    
    if args.projector_type == 'window_aware_linear':
        assert args.projection_window_size is not None

    assert args.dataset_path.endswith('.json')

    return args


def get_text_embeddings(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel) -> torch.Tensor:
    text_input = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]
    
    return text_embeddings


def tokenize_captions(examples: dict, tokenizer: CLIPTokenizer) -> torch.Tensor:
    tokenized_captions = tokenizer(
        examples["prompt"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    return tokenized_captions.input_ids


def get_training_dataloader(tokenizer: CLIPTokenizer, args: argparse.Namespace) -> DataLoader:
    dataset = load_dataset('json', data_files=args.dataset_path, split='train').cast_column('image', Image())

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images, "input_ids": tokenize_captions(examples, tokenizer)}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True) 

    return train_dataloader


def log_validation(
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    text_embedding_projector: torch.nn.Module,
    unet: UNet2DConditionModel,
    args: argparse.Namespace,
    global_step: int,
):
    print("Running validation... ")

    scheduler = PNDMScheduler.from_pretrained(args.stable_diffusion_checkpoint, subfolder="scheduler")

    images_for_visualization = []
    prompt_scores = {}
    for prompt in args.validation_prompts:
        images_path = os.path.join(args.output_dir, "validation_images", f"{global_step:08d}", prompt)

        with torch.no_grad():
            fixed_text_embeddings = text_embedding_projector(get_text_embeddings(prompt, tokenizer, text_encoder)).detach()

        image_scores_dict, prompt_average_score = generate_samples_and_evaluate_blip_vqa(
            vae,
            unet,
            scheduler,
            tokenizer,
            text_encoder,
            [prompt],
            fixed_text_embeddings=fixed_text_embeddings,
            evaluation_path=images_path,
            batch_size=args.evaluation_batch_size,
            num_evaluation_images=50,
            image_size=args.image_size,
        )
                    
        prompt_scores[prompt] = prompt_average_score

        best_images_names = sorted(image_scores_dict.items(), key=lambda x: -float(x[1]))[:5]
        best_images = [PIL.Image.open(os.path.join(images_path, img_name)) for img_name, _ in best_images_names]

        for img in best_images:
            images_for_visualization.append((prompt, img))
        
    wandb.log(
        {
            "validation": [
                wandb.Image(image, caption=f"{i}: {caption}")
                for i, (caption, image) in enumerate(images_for_visualization)
            ],
            "prompt_scores": prompt_scores,
            "average_score": sum(prompt_scores.values()) / len(prompt_scores),
        }
    )

    torch.cuda.empty_cache()


def train(
    train_dataloader: DataLoader,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: PNDMScheduler,
    text_embedding_projector: torch.nn.Module,
    optimizer: torch.optim.Adam,
    opt_scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    args: argparse.Namespace,
) -> torch.nn.Module:
    num_epochs = math.ceil(args.train_steps / len(train_dataloader))

    progress_bar = tqdm(
        range(0, args.train_steps),
        desc="Steps",
        initial=global_step
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch in train_dataloader:
            latents = vae.encode(batch["images"].to('cuda')).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)

            batch_size = latents.shape[0]

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"].to('cuda'), return_dict=False)[0]
            encoder_hidden_states = text_embedding_projector(encoder_hidden_states)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_embedding_projector.parameters(), 1.); 
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size

            opt_scheduler.step()   
        
            global_step += 1

            logs = {
                "train_loss": loss.detach().item(),
                "lr": opt_scheduler.get_last_lr()[0]
            }
            wandb.log(logs)

            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            if global_step % args.validation_steps == 0 or global_step == 1 or global_step == args.train_steps:
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    text_embedding_projector,
                    unet,
                    args,
                    global_step,
                )
                CheckpointUtil.save_checkpoint(text_embedding_projector, optimizer, opt_scheduler, global_step, args)
            elif global_step % args.checkpoint_steps == 0 or (global_step + 1) % args.validation_steps == 0 or (global_step + 1) == args.train_steps: # if at the checkpoint_step or before (previous step) validation
                CheckpointUtil.save_checkpoint(text_embedding_projector, optimizer, opt_scheduler, global_step, args)

            if global_step == args.train_steps:
                return text_embedding_projector

    return text_embedding_projector


class CheckpointUtil:
    @staticmethod
    def save_checkpoint(projector, optimizer, scheduler, global_step, args):
        if not os.path.exists(os.path.join(args.output_dir, "checkpoint")):
            os.makedirs(os.path.join(args.output_dir, "checkpoint"))
            
        torch.save(
            {
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
            },
            os.path.join(args.output_dir, "checkpoint", f"checkpoint.pth"),
        )
        print(f"Checkpoint saved [Step: {global_step}]")

    @staticmethod
    def load_checkpoint(projector, optimizer, scheduler, args):
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint", f"checkpoint.pth"))

        projector.load_state_dict(checkpoint["projector"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]

        print(f"Checkpoint loaded [Step: {global_step}]")

        return projector, optimizer, scheduler, global_step


def load_models(args) -> Tuple[AutoencoderKL, CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, PNDMScheduler]:
    vae = AutoencoderKL.from_pretrained(args.stable_diffusion_checkpoint, subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.stable_diffusion_checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.stable_diffusion_checkpoint, subfolder="text_encoder", use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(args.stable_diffusion_checkpoint, subfolder="unet", use_safetensors=True)
    scheduler = DDPMScheduler.from_pretrained(args.stable_diffusion_checkpoint, subfolder="scheduler")
    # scheduler = PNDMScheduler.from_pretrained(args.stable_diffusion_checkpoint, subfolder="scheduler")
        
    vae.to('cuda')
    text_encoder.to('cuda')
    unet.to('cuda');

    num_inference_steps = 25 # TODO: Needed?
    scheduler.set_timesteps(num_inference_steps)

    return vae, tokenizer, text_encoder, unet, scheduler


if __name__ == '__main__':
    args = parse_args()

    wandb.login()

    checkpoint_exists = os.path.exists(os.path.join(args.output_dir, "checkpoint", "checkpoint.pth"))

    run = wandb.init(
        project=f"{args.projector_type}-projection-of-clip-text-emb",
        config=vars(args),
        name=f"{'Resume Checkpoint - ' if checkpoint_exists else ''}{args.wandb_run_name}"
    )

    if checkpoint_exists:
        print("************************************************")
        print("*********** Resuming from checkpoint ***********")
        print("************************************************")

    # Initialization of the models
    vae, tokenizer, text_encoder, unet, scheduler = load_models(args)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False);

    dim_text_embedding = text_encoder.text_model.config.hidden_size
    if args.projector_type == 'linear':
        text_embedding_projector = CLIPTextEmbeddingLinearProjector(dim=dim_text_embedding, initialization_type=args.linear_projection_initialization_type).to('cuda')
    if args.projector_type == 'linear_and_skip':
        text_embedding_projector = CLIPTextEmbeddingLinearSkipProjector(dim=dim_text_embedding).to('cuda')
    elif args.projector_type == 'mlp':
        text_embedding_projector = CLIPTextEmbeddingMLPProjector(dim=dim_text_embedding).to('cuda')
    elif args.projector_type == 'window_aware_linear':
        text_embedding_projector = WindowAwareLinearProjection(text_embeddings_dim=dim_text_embedding, window_size=args.projection_window_size).to('cuda')
    
    print(f"Using text embedding projector of type: {type(text_embedding_projector)}")

    optimizer = torch.optim.Adam(text_embedding_projector.parameters(), lr=args.lr)
    opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler_decay_steps, gamma=0.1) # TODO
    
    global_step = 0

    if checkpoint_exists:
        print(f"Loading from checkpoint...")
        text_embedding_projector, optimizer, opt_scheduler, global_step = CheckpointUtil.load_checkpoint(
            text_embedding_projector, optimizer, opt_scheduler, args
        )
        text_embedding_projector.train()

    print(f"[Start of Training]")
    print(f"[Data and Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    train_dataloader = get_training_dataloader(tokenizer, args)

    print(f"[Training Setup] Training Dataset Size: {len(train_dataloader.dataset)}")
    print(f"[Training Setup] Batch Size: {train_dataloader.batch_size}")

    text_embedding_projector = train(
        train_dataloader,
        vae,
        unet,
        scheduler,
        text_embedding_projector,
        optimizer,
        opt_scheduler,
        global_step,
        args
    )
        
    torch.save(text_embedding_projector, os.path.join(args.output_dir, f'text_embedding_projector_{args.projector_type}.pth'))

    print(f"[Data and Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Finished Training]")
