import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# REEMPLAZA EL 42 POR LOS 2 ÚLTIMOS DÍGITOS
# DE TU CÓDIGO DE MATRÍCULA
# ==========================================
SEED = 97

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def set_seed(seed: int = SEED):
    """Aplica la semilla individual obligatoria en PyTorch y NumPy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_transforms():
    """Pipeline de Data Augmentation, conversión a tensor y normalización RGB."""
    # Constantes estándar de normalización para CIFAR-10
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return train_transform, test_transform

def get_dataloaders(data_dir='./data', batch_size=64, val_ratio=0.15, seed=SEED):
    """Descarga, particiona y crea los DataLoaders reproducibles."""
    set_seed(seed)
    train_transform, test_transform = get_transforms()

    # Descarga automática si no existe en la carpeta ./data
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Partición explícita de Train y Validación
    total_train = len(full_train_dataset)
    val_len = int(total_train * val_ratio)
    train_len = total_train - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [train_len, val_len], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[Dataset] Semilla individual fijada: {seed}")
    print(f"[Dataset] Partición -> Train: {train_len} imágenes | Val: {val_len} imágenes | Test: {len(test_dataset)} imágenes")

    return train_loader, val_loader, test_loader

def visualize_samples(dataset, num_images=16):
    """Muestra una cuadrícula de 16 imágenes con sus clases reales."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=9)
        ax.axis('off')
    plt.suptitle("Muestras de CIFAR-10 con Clases Reales", fontsize=12)
    plt.show()

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders()
    sample_imgs, sample_labels = next(iter(train_loader))
    print(f"[Batch Tensor 4D] Formato: {sample_imgs.shape} (Batch, Canales, Alto, Ancho)")
    print(f"[Batch Etiquetas] Formato: {sample_labels.shape}")
    print("\nAbriendo cuadrícula de visualización...")
    visualize_samples(train_loader.dataset, num_images=16)