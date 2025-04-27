import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import fcn_resnet101
import os

class SegmentationModel:
    def __init__(self, train_dataset, val_dataset, criterion, batch_size=8, lr=1e-4, num_epochs=25, device="cuda"):
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = criterion

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        self.model = fcn_resnet101(pretrained=True)
        self.model.backbone['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        for images, masks in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(images)["out"]

            loss = self.criterion(outputs, masks)
            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        dice_score = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device).unsqueeze(1)

                outputs = self.model(images)["out"]

                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")