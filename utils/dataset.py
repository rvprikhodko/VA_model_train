from utils.preprocess import denoise_with_ants, denoise_with_sitk
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, pairs, transform=None, denoise=None):
        self.pairs = pairs
        self.transform = transform
        self.denoise = denoise

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.denoise == "ants":
            image = denoise_with_ants(image)
            image = Image.fromarray(image)
            mask = Image.fromarray(np.array(mask))

        elif self.denoise == "sitk":
            image = denoise_with_sitk(image)
            image = Image.fromarray(image)
            mask = Image.fromarray(np.array(mask))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = mask.squeeze(0)

        return image, mask