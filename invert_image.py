from PIL import Image, ImageOps
import os
import numpy as np

# Paths
input_folder = "images/minus"
output_folder = "new_images/minus"

os.makedirs(output_folder, exist_ok=True)

# Threshold for black and white
threshold = 110

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("L")  # Grayscale

        # Convert to numpy array for thresholding
        img_np = np.array(img)
        bw_np = np.where(img_np < threshold, 0, 255).astype(np.uint8)  # Black/white

        # Invert image
        inverted_np = 255 - bw_np

        # Convert back to PIL image and resize to 28x28
        final_img = Image.fromarray(inverted_np).resize((28, 28))

        save_path = os.path.join(output_folder, filename)
        final_img.save(save_path)

print(f"Processed images saved to {output_folder}")

