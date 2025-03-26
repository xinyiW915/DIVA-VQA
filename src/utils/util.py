import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_tensor_frames(tensor, output_dir, tensor_name, num_frames=5):
    os.makedirs(output_dir, exist_ok=True)

    n, c, t, h, w = tensor.shape
    tensor = tensor.squeeze(0).permute(1, 0, 2, 3)  # Change to [T, C, H, W]

    for i in range(min(t, num_frames)):
        frame = tensor[i].cpu().numpy()  # Convert to NumPy
        frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
        frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize to [0, 1]

        # Save or display the frame
        output_path = os.path.join(output_dir, f"{tensor_name}_frame_{i + 1}.png")
        plt.imsave(output_path, frame)
        print(f"Saved frame {i + 1} of {tensor_name} to {output_path}")
