import os
import torch
import shutil
import string
import subprocess
import random
import json
import glob
import time
from diffusers import AutoencoderKL, UNet2DConditionModel, SchedulerMixin
from diffusers.utils import pt_to_pil
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


TEMP_DIRECTORIES_ROOT = 'temp_directories_for_blip_vqa_evaluation'

def make_temp_directories():
    """
    Creates temporary directories for BLIP VQA evaluation using the T2I-CompBench structure.

    Returns:
        evaluation_dir_path (str): 
            The path to the directory of evaluation. T2I-CompBench's BLIP VQA evaluation
            directories and files will be created under this directory.
        samples_dir_path (str): 
            The path to the directory for storing samples.
    """
    evaluation_dir_path = ''.join(random.choices(string.ascii_lowercase + string.digits, k=40)) + str(int(time.time() * 1000))
    evaluation_dir_path = os.path.join(TEMP_DIRECTORIES_ROOT, evaluation_dir_path)
    
    if os.path.exists(evaluation_dir_path):
        raise Exception("Something pretty rare happened!!!")

    samples_dir_path = os.path.join(evaluation_dir_path, 'samples')
    os.makedirs(samples_dir_path)

    return evaluation_dir_path, samples_dir_path


def evaluate_direcotry_using_blip_vqa(
        image_folder_path: str,
        print_log: bool = True
    ):
    """
    Evaluates images in a specified directory using the BLIP VQA model. The results will also be saved in
    a JSON file called 'vqa_result.json' in the same directory.

    Args:
        image_folder_path (str): The path to the directory containing the images to be evaluated.
        print_log (bool, optional): Whether to print log messages during evaluation. Defaults to True.

    Returns:
        dict: A dictionary containing the scores for each image and question pair.
    """
    assert os.path.exists(image_folder_path)

    image_folder_path = os.path.abspath(image_folder_path)

    images_in_folder_pattern = os.path.join(image_folder_path, '*.png')
    list_of_images = glob.glob(images_in_folder_pattern)

    if print_log:
        print(f"Evaluating {len(list_of_images)} images")
        
    evaluation_dir_path, samples_dir_path = make_temp_directories()

    for imgpath in list_of_images:
        shutil.copy2(imgpath, samples_dir_path)

    subprocess.call(["./t2i_compbench_vqa_evaluation.sh", os.path.abspath(evaluation_dir_path)])

    question_id_score_dict = {}
    with open(os.path.join(evaluation_dir_path, 'annotation_blip/vqa_result.json')) as f:
        vqa_result_json = json.load(f)
        for item in vqa_result_json:
            question_id_score_dict[item["question_id"]] = item["answer"]

    image_scores_dict = {}
    with open(os.path.join(evaluation_dir_path, 'annotation1_blip/vqa_test.json')) as f:
        vqa_test_json = json.load(f)
        for item in vqa_test_json:
            image_scores_dict[os.path.basename(item['image'])] = question_id_score_dict[item['question_id']]

    assert len(image_scores_dict) == len(list_of_images)
    assert set(image_scores_dict.keys()) == set([os.path.basename(f) for f in list_of_images])

    shutil.rmtree(evaluation_dir_path)

    result_path = os.path.join(image_folder_path, 'vqa_result.json')
    if os.path.exists(result_path) and print_log:
        print("Rewriting results of VQA")
    with open(result_path, 'w') as f:
        json.dump(image_scores_dict, f)
    
    return image_scores_dict
    

def generate_samples_and_evaluate_blip_vqa(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    fixed_text_embeddings: torch.Tensor,
    evaluation_path: str,
    batch_size: int = 10,
    num_evaluation_images: int = 30,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    image_size: int = 512,
    clean_fixed_text_embeddings: torch.Tensor = None,
    early_guidance_timestep_threshold: int = -1,
    seed: int = None,
):
    """
    Generates samples using components of T2I Stable Diffusion (with given text_embedding tensor) and evaluates them using the VQA metric.

    Args:
        vae (AutoencoderKL): 
            The VAE model.
        unet (UNet2DConditionModel): 
            The UNet model.
        scheduler (SchedulerMixin): 
            The noise scheduler for doing the backward process.
        tokenizer (CLIPTokenizer): 
            The tokenizer for CLIP model.
        text_encoder (CLIPTextModel): 
            The text encoder model.
        prompt (str): 
            The prompt for generating the samples.
        fixed_text_embeddings (torch.Tensor):
            The input text embeddings tensor to the UNet. This could be the output of the text_encoder or a modified version of it.
        evaluation_path (str): 
            The path to save the generated images and the evaluation results.
        batch_size (int, optional): 
            The batch size for generating samples. Defaults to 10.
        num_evaluation_images (int, optional): 
            The number of evaluation images to generate. Defaults to 30.
        guidance_scale (float, optional):
            The scale factor for guidance. Defaults to 7.5.
        num_inference_steps (int, optional):
            The number of inference steps. Defaults to 25.
        image_size (int, optional): 
            The size of the generated images. Defaults to 512.
        clean_fixed_text_embeddings (torch.Tensor, optional):
            The clean fixed text embeddings, generated by the text_encoder with the prompt as input. Defaults to None.
            This is used for early guidance if early_guidance_timestep_threshold is set to a value greater than 0.
        early_guidance_timestep_threshold (int, optional): 
            Specifies the threshold for initiating early guidance, with a default value of -1. When this threshold is set to
            a particular timestep t, the guidance will utilize `fixed_text_embeddings` instead of `clean_fixed_text_embeddings` for
            all timesteps greater than or equal to t.
        seed (int, optional): 
            The random seed for generating samples. Defaults to None.

    Returns:
        Tuple[Dict[str, float], float]:
            A tuple containing a dictionary of image scores and the average score.
    """
    
    assert num_evaluation_images % batch_size == 0, "just for now!!!"

    if os.path.exists(evaluation_path):
        print("Removing previous evaluation path ...")
        shutil.rmtree(evaluation_path)
    os.makedirs(evaluation_path)

    text_embeddings = fixed_text_embeddings.repeat(batch_size, 1, 1).clone()
    max_length = text_embeddings.shape[1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(unet.device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    if early_guidance_timestep_threshold != -1:
        text_embeddings_clean = clean_fixed_text_embeddings.repeat(batch_size, 1, 1).clone()
        text_embeddings_clean = torch.cat([uncond_embeddings, text_embeddings_clean])

    torch.cuda.empty_cache(); # TODO: ?

    f = 2 ** (len(vae.config.block_out_channels) - 1)

    for b_idx in range(num_evaluation_images // batch_size):
        latents = torch.randn(
            (batch_size, unet.config.in_channels, image_size // f, image_size // f),
            device=unet.device,
            generator=None if seed is None else torch.Generator(device='cuda').manual_seed(seed*100 + b_idx),
        )
        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings if t > early_guidance_timestep_threshold else text_embeddings_clean
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / vae.scaling_factor * latents
        with torch.no_grad():
            images = vae.decode(latents).sample
        
        for idx, pil_img in enumerate(pt_to_pil(images)):
            pil_img.save(os.path.join(evaluation_path, f'{prompt[0]}_{(b_idx*batch_size + idx):06d}.png'))
        
    prev_device = vae.device
    vae.to('cpu')
    text_encoder.to('cpu')
    unet.to('cpu');
    
    torch.cuda.empty_cache(); # TODO: ?

    image_scores_dict = evaluate_direcotry_using_blip_vqa(image_folder_path=evaluation_path)

    torch.cuda.empty_cache(); # TODO: ?

    vae.to(prev_device)
    text_encoder.to(prev_device)
    unet.to(prev_device)

    average_score = sum(map(lambda x: float(x), image_scores_dict.values())) / len(image_scores_dict)

    return image_scores_dict, average_score
