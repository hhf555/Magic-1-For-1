import argparse
import os
import imageio
import fire
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import glob
from model_dit.utils.util import instantiate
import sys
from datetime import datetime
import warnings 
from PIL import Image
import torchvision.transforms as transforms

from datasets.test_image import TestImageDataset
from model_dit.utils.image import tensor2image


def t2v(config: DictConfig):
    """Main entry point for inference."""
    # 0. device
    device = torch.device("cuda")

    # 1. Check missing keys
    missing_keys = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    # 2. Model loading
    module = instantiate(config.model, instantiate_module=False)
    model = module(config=config, device=device)
    
    assert os.path.exists(config.resume_ckpt) and os.path.isfile(config.resume_ckpt)
    print(f"Loading model from checkpoint: {config.resume_ckpt}")
    ckpt = torch.load(config.resume_ckpt, map_location="cpu")

    if config.inference.num_inference_steps == 4:
        m, u = model.model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        m, u = model.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt
    model.eval()

    # 3. Quantization
    if config.inference.quantization:
        from model_dit.utils.quant import quantization_interface
        model.model = quantization_interface(model.model, save=False, model_path="outputs/quant_new/model.pth", quant_type=config.inference.quantization_level)
        model.text_encoder = quantization_interface(model.text_encoder, save=False, model_path="outputs/quant_new/text_encoder.pth")
        model.text_encoder_2 = quantization_interface(model.text_encoder_2, save=False, model_path="outputs/quant_new/text_encoder_2.pth")
        model.text_encoder_vlm = quantization_interface(model.text_encoder_vlm, save=False, model_path="outputs/quant_new/text_encoder_vlm.pth")
        print("Quantization is enabled")
        
    # 4. Data loading
    print("Loading data")
    if config.inference.cli:
        print("=== Text-to-Image-to-Video CLI Mode ===")
        print("Enter your prompt with image tags. Example:")
        print("A cat playing piano")
        print("Type 'exit' to quit")
        
        prompt = input("\nEnter your prompt: ").strip()
        image_path = input("Enter the image path: ").strip()
        config.test_data.image_paths_and_scales = [[image_path, 21, 1.2, prompt]]

    test_dataset = TestImageDataset(**config.test_data,resize=True,crop=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # 5. Test
    output_dir = config.inference.output_dir

    # get timestamp and use as folder name
    timestamp = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = os.path.join(output_dir, timestamp)

    for idx, batch in enumerate(test_dataloader):
        img_name = batch["img_name"][0]
        video_length = batch["video_length"][0]
        if video_length > 1:
            config.inference.repeat_times = 1

        for i in tqdm(range(config.inference.repeat_times)):
            # Seed everything
            seed_everything(config.seed + i)
            # Generate images
            outputs = model.predict_step(
                batch=batch, 
                guidance_scale=config.inference.guidance_scale,
                low_memory=config.inference.low_memory,
                downsample_ratio=2
            )
            output_name = f"{img_name}_repeat-{i}.mp4"
            # save_dir = os.path.join(output_dir, img_name) # For Image save only, Image name as dir and Image item
            save_dir = output_dir
            os.makedirs(save_dir, exist_ok=True)

            # outputs shape: 1 c f h w [0, 1]
            img_arrays = []
            for j in range(outputs.size(2)):
                tensor = outputs[0, :, j].cpu()
                img_array = tensor2image(
                    tensor=tensor,
                )
                img_arrays.append(img_array)
            imageio.mimwrite(os.path.join(save_dir, output_name), img_arrays, fps=24)
            print('=========video save to =========', os.path.join(save_dir, output_name))

    print("finish validation!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-q", "--quantization", type=bool, default=False)
    parser.add_argument("-l", "--quantization_level", type=str, default="int8")
    parser.add_argument("-m", "--low_memory", type=bool, default=False)
    parser.add_argument("--cli", type=bool, default=False)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.inference.output_dir = os.path.join("./outputs/inference", os.path.basename(args.config)[:-5])
    cfg.inference.quantization = args.quantization
    cfg.inference.quantization_level = args.quantization_level
    cfg.inference.low_memory = args.low_memory
    cfg.inference.cli = args.cli
    # output the config to the save dir.
    # output the config to the save dir.
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    with open(os.path.join(cfg.inference.output_dir, "infer_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        t2v(cfg)