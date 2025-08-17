import os
from torchvision import datasets, transforms
from PIL import Image

# Directory to save the images
output_dir = "mnist_images"
os.makedirs(output_dir, exist_ok=True)

# Create subfolders for digits 0-9
for i in range(10):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

# Load MNIST dataset
mnist_train = datasets.MNIST(root="dataset/", train=True, download=True, transform=transforms.ToTensor())

# Save each image
for idx, (img_tensor, label) in enumerate(mnist_train):
    # Convert tensor to PIL Image
    img = transforms.ToPILImage()(img_tensor)
    
    # Save path: mnist_images/label/label_idx.png
    save_path = os.path.join(output_dir, str(label), f"{label}_{idx}.png")
    img.save(save_path)

print(f"Saved {len(mnist_train)} images to '{output_dir}'")
