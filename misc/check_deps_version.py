import torch
import sys

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Check if cuDNN is available
    if torch.backends.cudnn.is_available():
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("cuDNN is not available.")
else:
    print("CUDA is not available.")
