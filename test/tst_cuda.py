import torch

# Check if CUDA is available
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.version.cuda)         # Should return the CUDA version if installed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")