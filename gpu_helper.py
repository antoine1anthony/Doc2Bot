# gpu_helper.py
import torch

DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu" # "cuda" if torch.cuda.is_available() else "cpu" or  ("mps" if torch.backends.mps.is_available() else "cpu") if on Apple Silicon
