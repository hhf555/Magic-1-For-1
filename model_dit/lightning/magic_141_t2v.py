import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import torch
import copy
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat
from lightning import LightningModule, seed_everything
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm

from model_dit.utils.loss import compute_snr, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from model_dit.utils.util import instantiate
from model_dit.models.magic_141_video.text_encoder import TextEncoder
from model_dit.models.magic_141_video.text_encoder.text_encoder_vlm import TextEncoderVLM, LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT
from model_dit.models.magic_141_video.constants import PROMPT_TEMPLATE

@torch.no_grad()
def noise_inversion(model_infer,
                    noise_tensor,
                    encoder_hidden_states,
                    text_states,
                    text_mask,
                    text_states_2,
                    reference_image, 
                    cur_sigma_t,
                    device,
                    guidance_scale=3.0,
                    free_init=True):
    reference_image = reference_image.repeat(1, 1, noise_tensor.shape[2], 1, 1)
    if guidance_scale != 1.0:
        _reference_image = reference_image[:1]
    else:
        _reference_image = reference_image 
    if guidance_scale != 1.0:
        noisy_latents = noisy_latents.repeat(2, 1, 1, 1, 1)
    else:
        noisy_latents = noisy_latents.repeat(1, 1, 1, 1, 1)
    print(f"{noisy_latents.shape=}")
    print(f"{ts.shape=}")
    print(f"{encoder_hidden_states.shape=}")
    print(f"{text_states.shape=}")
    print(f"{text_mask.shape=}")
    print(f"{text_states_2.shape=}")
    with torch.autocast(device_type="cuda", dtype=model_infer.dtype, enabled=True):
        print("==============>Text Cond Inversion=================")
        if guidance_scale != 1.0:
            noise_pred_uncond = model_infer(
                hidden_states=noisy_latents[:1].to(dtype=torch.float32),
                encoder_hidden_states=encoder_hidden_states[:1].to(dtype=torch.float32),
                text_states=text_states[:1].to(device=device, dtype=torch.float32),
                text_mask=text_mask[:1] if text_mask is not None else None,
                text_states_2=text_states_2[:1].to(device=device, dtype=torch.float32) if text_states_2 is not None else None,
                timestep=ts[:1],
                ).sample
            noise_pred_text = model_infer(
                hidden_states=noisy_latents[1:].to(dtype=torch.float32),
                encoder_hidden_states=encoder_hidden_states[1:].to(dtype=torch.float32),
                text_states=text_states[1:].to(device=device, dtype=torch.float32),
                text_mask=text_mask[1:] if text_mask is not None else None,
                text_states_2=text_states_2[1:].to(device=device, dtype=torch.float32) if text_states_2 is not None else None,
                timestep=ts[1:],
                ).sample
            noise_pred = noise_pred_uncond - guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred_text = model_infer(
                hidden_states=noisy_latents.to(dtype=torch.float32),
                encoder_hidden_states=encoder_hidden_states.to(dtype=torch.float32),
                text_states=text_states.to(device=device, dtype=torch.float32),
                text_mask=text_mask if text_mask is not None else None,
                text_states_2=text_states_2.to(device=device, dtype=torch.float32) if text_states_2 is not None else None,
                timestep=ts,
                ).sample
            noise_pred = noise_pred_text
        H,W = noise_pred_text.shape[-2:]
        initial_noise = ori_noisy_latents + noise_pred * (1 - cur_sigma_t)
    return initial_noise

# This LightningModule support rectify flow denoising
# During Training: reference image condition will be the first frame of current video
class EmoLitModule(LightningModule):
    def __init__(
        self,
        config: DictConfig,
        device=None,
    ) -> None:
        super().__init__()
        self.config = config

        self._get_scheduler()  # load scheduler
        self.to(device)
        self._init_model(device)
        # self.save_hyperparameters(config)  # save hyperparameters for resuming the training

    def _init_model(self, device):
        config = self.config
        model = instantiate(config.model.denoising_model, instantiate_module=False)
        model_additional_kwargs = config.model.get("model_additional_kwargs", {})
        if not isinstance(model_additional_kwargs, dict):
            model_additional_kwargs = OmegaConf.to_container(model_additional_kwargs)
        model_additional_kwargs["device"] = device if device is not None else "cpu"
        model_additional_kwargs["dtype"] = eval(config.model.dtype)

        self.model = model.from_pretrained(
            config.model.base_model_path,
            from_scratch=config.model.base_model_from_scratch,
            model_additional_kwargs=model_additional_kwargs,
        )
        self.model.requires_grad_(False)
        self.model.to(dtype=eval(config.model.dtype))

        # clip
        if config.model.clip_name is not None:
            pass  # TODO

        # VAE
        if config.model.vae_model_path is not None:
            from ..models.magic_141_video.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
            self.vae = AutoencoderKLCausal3D.from_pretrained(config.model.vae_model_path)
            # freeze vae
            self.vae.requires_grad_(False)
            self.vae.to(device=device, dtype=eval(config.model.vae_dtype))
            
        if config.is_inference:
            prompt_template = PROMPT_TEMPLATE["dit-llm-encode"]
            prompt_template_video = PROMPT_TEMPLATE["dit-llm-encode-video"]
            max_length = 256 + prompt_template_video.get(
                "crop_start", 0
            )
            self.text_encoder = TextEncoder(
                text_encoder_type="llm",
                max_length=max_length,
                text_encoder_path=config.model.text_encoder_path,
                text_encoder_precision=eval(config.model.text_encoder_dtype),
                tokenizer_type="llm",
                prompt_template=prompt_template,
                prompt_template_video=prompt_template_video,
                hidden_state_skip_layer=2,
                device=device,
            )
            # self.text_encoder.to("cpu")

            self.text_encoder_2 = TextEncoder(
                text_encoder_type="clipL",
                max_length=77,
                text_encoder_path=config.model.text_encoder2_path,
                text_encoder_precision=eval(config.model.text_encoder2_dtype),
                tokenizer_type="clipL",
                device=device,
            )
            # self.text_encoder_2.to("cpu")
            
            self.text_encoder_vlm = TextEncoderVLM(
                text_encoder_type="vlm",
                max_length=512,
                text_encoder_path=config.model.text_encoder_vlm_path,
            ).to(dtype=eval(config.model.text_encoder_vlm_dtype))
            # self.text_encoder_vlm.to("cpu")
            # torch.cuda.empty_cache()

    def _get_scheduler(self) -> Any:
        if self.config.get("noise_scheduler", "flow") == "flow":
            from ..models.magic_141_video.diffusion.schedulers import FlowMatchDiscreteScheduler
            self.train_noise_scheduler = FlowMatchDiscreteScheduler(
                shift=self.config.scheduler.flow_shift,
                reverse=self.config.scheduler.flow_reverse,
                solver=self.config.scheduler.flow_solver,
            )
        else:
            raise ValueError(f"Invalid denoise type when finetune model")
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / self.vae.config.scaling_factor * latents
        # b c f h w
        image = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample # first frame is the single image.
        return (image / 2.0 + 0.5).clamp(0, 1)
    
    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "video",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # max_length = prompt_embeds.shape[1]
            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type, device=device
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )
    
    @torch.no_grad()
    def predict_step(
        self, 
        batch: Dict, 
        guidance_scale: float, 
        low_memory: bool = False,
        downsample_ratio: int = 1,
    ) -> torch.Tensor:
        model_infer = self.model

        if low_memory:
            model_infer = model_infer.to("cpu")
            torch.cuda.empty_cache()
        else:
            model_infer.to(self.device)

        # 1. get data
        image = batch["image"].to(self.device).unsqueeze(1)
        video_length = batch["video_length"][0].item()
        prompt = batch["prompt"][0]
        neg_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
        ref_image_path = batch["ref_image_path"][0]
        do_classifier_free_guidance = False
        if guidance_scale > 1.0:
            do_classifier_free_guidance = True

        self.text_encoder.to(self.device)
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=neg_prompt,
            text_encoder=self.text_encoder
        )

        if low_memory:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        self.text_encoder_2.to(self.device)
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = self.encode_prompt(
            prompt,
            self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=neg_prompt,
            text_encoder=self.text_encoder_2,
        )

        if low_memory:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        # Get Image text encoder
        self.text_encoder_vlm.to(self.device)
        ref_image = Image.open(ref_image_path)
        image_embeds = self.text_encoder_vlm.forward(text=LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT, image1=ref_image).hidden_state

        prompt_embeds = torch.cat([image_embeds, prompt_embeds], dim=1)
        image_masks = torch.ones((image_embeds.shape[0], image_embeds.shape[1]), dtype=prompt_mask.dtype)
        prompt_mask = torch.cat([image_masks.to(prompt_mask.device), prompt_mask], dim=1)
        if do_classifier_free_guidance:
            # zero_image = Image.fromarray(np.zeros_like(np.array(ref_image)))
            # zero_image_embeds = self.text_encoder_vlm.forward(text=LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT, image1=zero_image).hidden_state
            negative_prompt_embeds = torch.cat([image_embeds, negative_prompt_embeds], dim=1)
            negative_prompt_mask = torch.cat([image_masks.to(negative_prompt_mask.device), negative_prompt_mask], dim=1)
       
        if low_memory:
            self.text_encoder_vlm.to("cpu")
            torch.cuda.empty_cache()
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # vae
        image = rearrange(image, "b f c h w -> b c f h w")
        image_latents = (
            self.vae.encode(image.to(dtype=self.vae.dtype) * 2.0 - 1.0).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        if low_memory:
            self.vae.to("cpu")
        B, C, F, H, W = image_latents.shape

        # face_mask b 1 f h w
        # 2. set timesteps
        self.train_noise_scheduler.set_timesteps(
            self.config.inference.num_inference_steps,
            device=self.device,
        )
        timesteps = self.train_noise_scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.config.inference.num_inference_steps * self.train_noise_scheduler.order

        # 3.set inference latent
        generator = torch.manual_seed(torch.randint(0, 100000, (1,)).item())
        latent = (
            randn_tensor((B, C, video_length, H, W), generator=generator, device=self.device, dtype=image_latents.dtype)
        )

        image_latents = (
            torch.cat([image_latents, image_latents])
            if do_classifier_free_guidance
            else image_latents
        )

        # print(f"{image_latents.shape=}")
        # if downsample_ratio > 1:
        #     image_latents = image_latents[:, :, :, ::downsample_ratio, ::downsample_ratio]

        if getattr(self.config.inference, "inversion", False):
            latent = noise_inversion(
                model_infer,
                latent,
                encoder_hidden_states=image_latents,
                text_states=prompt_embeds,
                text_mask=prompt_mask,
                text_states_2=prompt_embeds_2,
                reference_image = image_latents,
                cur_sigma_t = 0.9999,
                device = self.device,
                guidance_scale=guidance_scale,
            )
        
        # 4. inference
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            if i < num_warmup_steps:
                continue

            latent_input = latent.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
            ts = torch.tensor([t] * latent_input.shape[0], device=self.device, dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=model_infer.dtype, enabled=True):
                if do_classifier_free_guidance:
                    print("==============> Uncond =================")
                    noise_pred_uncond = model_infer(
                        hidden_states=latent_input[:1].to(dtype=torch.float32),
                        encoder_hidden_states=image_latents[:1].to(dtype=torch.float32),
                        text_states=prompt_embeds[:1].to(device=self.device, dtype=torch.float32),  # [2, 256, 4096]
                        text_mask=prompt_mask[:1],  # [2, 256]
                        text_states_2=prompt_embeds_2[:1].to(device=self.device, dtype=torch.float32),  # [2, 768]
                        timestep=ts[:1],
                    ).sample
                    
                    print("==============>Text Cond =================")
                    noise_pred_text = model_infer(
                        hidden_states=latent_input[1:].to(dtype=torch.float32),
                        encoder_hidden_states=image_latents[1:].to(dtype=torch.float32),
                        text_states=prompt_embeds[1:].to(device=self.device, dtype=torch.float32),  # [2, 256, 4096]
                        text_mask=prompt_mask[1:],  # [2, 256]
                        text_states_2=prompt_embeds_2[1:].to(device=self.device, dtype=torch.float32),  # [2, 768]
                        timestep=ts[1:],
                    ).sample

                    print("CFG:",guidance_scale," Guidance Value Mean:", (guidance_scale  * (noise_pred_text - noise_pred_uncond)).mean()) 
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                else:
                    noise_pred = model_infer(
                        hidden_states=latent_input.to(dtype=torch.float32),
                        encoder_hidden_states=image_latents.to(dtype=torch.float32),
                        text_states=prompt_embeds.to(device=self.device, dtype=torch.float32),  # [2, 256, 4096]
                        text_mask=prompt_mask,  # [2, 256]
                        text_states_2=prompt_embeds_2.to(device=self.device, dtype=torch.float32),  # [2, 768]
                        timestep=ts,
                    ).sample
                
            latent = self.train_noise_scheduler.step(
                noise_pred,
                t,
                latent,
            ).prev_sample  # outputs are prev_sample and pred_original_sample
        
        if low_memory:
            self.vae.to(self.device)
            model_infer.to("cpu")

        return self.decode_latents(latent)