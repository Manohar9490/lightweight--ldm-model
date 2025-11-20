import torch
from diffusers import DDPMPipeline

def load_ddpm_cifar32(device: str):
    model_id = "google/ddpm-cifar10-32"
    pipe = DDPMPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    return pipe, model_id
