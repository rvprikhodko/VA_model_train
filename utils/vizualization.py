import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_dataset(dataset, num_images=5):
    fig, axes = plt.subplots(num_images, 2, figsize=(15, 5 * num_images))

    for i in range(num_images):
        image, mask = dataset[i]
        image_tensor = image.unsqueeze(0)
        mask = mask.numpy()

        axes[i, 0].imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_results(model, dataset, device, num_images=10):
    model.eval()
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    for i in range(num_images):
        image, mask = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)
        mask = mask.numpy()

        with torch.no_grad():
            pred = model(image_tensor)["out"]
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)

        axes[i, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_binary, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()