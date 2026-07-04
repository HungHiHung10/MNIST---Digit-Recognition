from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover
    datasets = None
    transforms = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.pt'


class MNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _read_idx_images(file_path: Path) -> np.ndarray:
    import struct

    with file_path.open('rb') as file:
        magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid IDX image file magic number: {magic}')
        image_data = np.frombuffer(file.read(), dtype=np.uint8)
    return image_data.reshape(num_images, rows, cols)


def _read_idx_labels(file_path: Path) -> np.ndarray:
    import struct

    with file_path.open('rb') as file:
        magic, num_labels = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid IDX label file magic number: {magic}')
        label_data = np.frombuffer(file.read(), dtype=np.uint8)
    return label_data


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    return (image - 0.1307) / 0.3081


class MNISTIdxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
    ) -> None:
        self.images = _read_idx_images(images_path)
        self.labels = _read_idx_labels(labels_path)
        if len(self.images) != len(self.labels):
            raise ValueError('Số lượng ảnh và nhãn không khớp.')

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index].astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        image = _normalize_image(image)
        label = int(self.labels[index])
        return image, label


def get_data_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    local_data_dir = BASE_DIR.parent / 'assets' / 'dataset'
    train_images_path = local_data_dir / 'train-images.idx3-ubyte'
    train_labels_path = local_data_dir / 'train-labels.idx1-ubyte'
    test_images_path = local_data_dir / 't10k-images.idx3-ubyte'
    test_labels_path = local_data_dir / 't10k-labels.idx1-ubyte'

    if all(path.exists() for path in [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
    ]):
        train_dataset = MNISTIdxDataset(train_images_path, train_labels_path)
        test_dataset = MNISTIdxDataset(test_images_path, test_labels_path)
    else:
        if datasets is None or transforms is None:
            raise ImportError(
                'Không tìm thấy torchvision. Cài đặt torchvision hoặc cung cấp dữ liệu MNIST tại assets/dataset.'
            )
        train_dataset = datasets.MNIST(
            BASE_DIR,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
        )
        test_dataset = datasets.MNIST(
            BASE_DIR,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(data_loader.dataset)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item() * images.size(0)
            predictions = output.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()

    dataset_size = len(data_loader.dataset)
    return running_loss / dataset_size, correct / dataset_size


def save_model(model: nn.Module, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path = MODEL_PATH, device: torch.device | None = None) -> MNISTClassifier:
    device = device or torch.device('cpu')
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(batch_size=128)

    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=3,
        min_lr=1e-5,
        verbose=True,
    )

    best_accuracy = 0.0
    for epoch in range(1, 11):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        print(
            f'Epoch {epoch:02d}: train_loss={train_loss:.4f}, '
            f'test_loss={test_loss:.4f}, test_accuracy={test_accuracy*100:.2f}%'
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_model(model)
            print(f'  → Lưu model tốt nhất với accuracy={best_accuracy*100:.2f}%')

    print(f'Hoàn tất đào tạo. Best test accuracy: {best_accuracy*100:.2f}%')
    print(f'Model đã lưu tại: {MODEL_PATH}')


if __name__ == '__main__':
    main()
