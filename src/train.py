import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders, CLASSES
from model import ConvNetCIFAR

# Semilla individual derivada de tu código de matrícula
SEED = 97

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        
    return running_loss / total, 100.0 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    return running_loss / total, 100.0 * correct / total, all_targets, all_preds

def run_experiment(seed=SEED, epochs=12, lr=0.001, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('outputs', exist_ok=True)
    
    print(f"Dispositivo de ejecución: {device}")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, seed=seed)
    
    model = ConvNetCIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nIniciando entrenamiento con Semilla={seed} por {epochs} épocas...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%")

    # Evaluación final con el set de prueba
    test_loss, test_acc, test_targets, test_preds = evaluate(model, test_loader, criterion, device)
    print(f"\n[Evaluación Final Test] Loss: {test_loss:.4f} - Accuracy: {test_acc:.2f}%")

    # 1. Generar Curvas de Aprendizaje (loss_accuracy.png)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', marker='s')
    plt.title('Curvas de Pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy', marker='s')
    plt.title('Curvas de Precisión (Accuracy %)')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/loss_accuracy.png', dpi=300)
    plt.close()
    print("[Artefacto Creado] outputs/loss_accuracy.png")

    # 2. Generar Matriz de Confusión (confusion_matrix.png)
    cm = confusion_matrix(test_targets, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f'Matriz de Confusión - Test Set (Seed {seed})')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()
    print("[Artefacto Creado] outputs/confusion_matrix.png")

if __name__ == '__main__':
    run_experiment(seed=SEED, epochs=12)