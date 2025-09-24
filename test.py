import torch
import torchvision.models as models

# Load ResNet-152 pretrained on ImageNet
resnet152 = models.resnet152(pretrained=True)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet152.to(device)

# Number of parameters
num_params = sum(p.numel() for p in resnet152.parameters())
print(f"Number of parameters: {num_params / 1e6:.2f}M")

# Example: estimate GPU memory for batch size 32, 224x224 images, FP32
batch_size = 32
img_size = (3, 224, 224)
x = torch.randn((batch_size, *img_size), device=device)

# Forward pass to roughly check memory usage
with torch.no_grad():
    out = resnet152(x)

print("Forward pass done! Check your GPU memory usage with nvidia-smi.")
