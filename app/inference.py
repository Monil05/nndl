"""Inference matching both notebooks' held-out test preprocessing."""

from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0


PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "efficientnet_b0_best.pth"
CNN_CHECKPOINT_PATH = PROJECT_DIR / "checkpoints_cnn_baseline" / "cnn_baseline_best.pth"
EXPECTED_CLASS_TO_IDX = {"fake": 0, "real": 1}
EXPECTED_TARGET_CONVENTION = "FAKE=1, REAL=0"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeepfakeCNN(nn.Module):
    """Custom CNN architecture from the baseline training notebook."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 32, 64, 128, 256, 256]
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.4), nn.Linear(256, 1))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def evaluation_transform(image_size):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(checkpoint_path=CHECKPOINT_PATH):
    """Load a trusted local training checkpoint without downloading ImageNet weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The original training checkpoint contains Python/NumPy RNG state, so it
    # requires weights_only=False. Only load the checkpoint produced by this project.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_to_idx") != EXPECTED_CLASS_TO_IDX:
        raise ValueError("Checkpoint class mapping differs from fake=0, real=1.")
    config = checkpoint.get("config", {})
    if config.get("target_convention") != EXPECTED_TARGET_CONVENTION:
        raise ValueError("Checkpoint target convention differs from FAKE=1, REAL=0.")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    image_size = int(config.get("image_size", 224))
    return model, evaluation_transform(image_size), device


def load_cnn_model(checkpoint_path=CNN_CHECKPOINT_PATH):
    """Load the trusted custom CNN training checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_to_idx") != EXPECTED_CLASS_TO_IDX:
        raise ValueError("CNN checkpoint class mapping differs from fake=0, real=1.")
    if checkpoint.get("architecture") != "DeepfakeCNN_custom_baseline":
        raise ValueError("Unexpected CNN checkpoint architecture.")
    config = checkpoint.get("config", {})
    if config.get("target_convention") != EXPECTED_TARGET_CONVENTION:
        raise ValueError("CNN checkpoint target convention differs from FAKE=1, REAL=0.")

    model = DeepfakeCNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    image_size = int(config.get("image_size", 224))
    return model, evaluation_transform(image_size), device


def predict_image(image: Image.Image, model, transform, device):
    """Return the class, FAKE probability, REAL probability and class confidence."""
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        fake_probability = torch.sigmoid(model(x)).item()

    real_probability = 1.0 - fake_probability
    prediction = "FAKE" if fake_probability >= 0.5 else "REAL"
    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "confidence": max(fake_probability, real_probability),
    }
