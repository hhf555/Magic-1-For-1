import os
from pathlib import Path
from safetensors.torch import load_file, save_file
from optimum.quanto import quantization_map, requantize
import json
import torch
import logging

def quantization_interface(
    model,
    model_path: str = None,
    save: bool = False,
    quant_type: str = "int8"
):
    if model_path == "outputs/quant_new/model.pth":
        quant_type = "int4"
        
    if quant_type == "int8":
        return quanto(model, model_path, save)
    elif quant_type == "int4":
        return torch_ao(model, model_path, save)
    else:
        raise ValueError(f"Quantization type {quant_type} not supported")

def fp8_linear_forward(cls, inputs, original_dtype=torch.bfloat16):
    if inputs.dtype != torch.bfloat16:
        return cls.original_forward(inputs.to(original_dtype))
    else:
        return cls.original_forward(inputs)

def torch_ao(
    model,
    model_path: str = None,
    save: bool = False
):  
    import torchao
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int4_weight_only

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    group_size = 32
    model = model.to("cuda")
    use_hqq = True

    try:
        if hasattr(model, 'single_blocks'):
            logger.info("Applying INT4 quantization to single_blocks...")
            quantize_(model.single_blocks, int4_weight_only(group_size=group_size, use_hqq=use_hqq))
        else:
            logger.warning("Model does not have single_blocks")
        
        if hasattr(model, 'double_blocks'):
            logger.info("Applying INT4 quantization to double_blocks...")
            quantize_(model.double_blocks, int4_weight_only(group_size=group_size, use_hqq=use_hqq))
        else:
            logger.warning("Model does not have double_blocks")
        
        remaining_modules = []
        for name, module in model.named_children():
            if name not in ['single_blocks', 'double_blocks']:
                remaining_modules.append(module)
        
        logger.info("Applying INT8 quantization to remaining modules...")
        for module in remaining_modules:
            quantize_(module, int8_dynamic_activation_int8_weight())
            
        torch.compile(model, mode='max-autotune')
        logger.info("Model quantization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        raise

    return model

def quanto(
    model,
    model_path: str = None,
    save: bool = False
):  
    # Quantize model if no path provided
    from optimum.quanto import freeze, qint8, quantize
    quantize(model, qint8)
    freeze(model)
    if save:
        print("saving model in", model_path)
        # Save state dict instead of full model
        torch.save(model.state_dict(), model_path)
    elif model_path is not None:
        print(f"Loading quantized model from {model_path}")
        loaded_model = torch.load(model_path)
        if hasattr(loaded_model, 'state_dict'):
            state_dict = loaded_model.state_dict()
            # Print some diagnostics
            print(f"Loaded model type: {type(loaded_model)}")
            print(f"Number of parameters in loaded state: {len(state_dict)}")
            print(f"Number of parameters in target model: {len(model.state_dict())}")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Number of missing keys: {len(missing_keys)}")
            print(f"Number of unexpected keys: {len(unexpected_keys)}")
        else:
            state_dict = loaded_model
            missing_keys, unexpected_keys = model.load_state_dict(loaded_model, strict=False)
            print(f"Number of missing keys: {len(missing_keys)}")
            print(f"Number of unexpected keys: {len(unexpected_keys)}")
        del loaded_model


    # Apply fp8 forward hook to linear layers
    for key, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            original_forward = layer.forward
            setattr(layer, "original_forward", original_forward)
            setattr(layer, "forward", lambda inputs, m=layer: fp8_linear_forward(m, inputs))
    
    return model