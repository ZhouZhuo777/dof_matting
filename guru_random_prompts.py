from datetime import datetime
import os

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

torch_dtype = torch.float32 # torch.float16 只支持GPU
