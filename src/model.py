import torch
import torch.nn as nn

class ConvNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNetCIFAR, self).__init__()
        
        # Bloque Convolucional 1: [B, 3, 32, 32] -> [B, 32, 16, 16]
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque Convolucional 2: [B, 32, 16, 16] -> [B, 64, 8, 8]
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloque Convolucional 3: [B, 64, 8, 8] -> [B, 128, 4, 4]
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Capa de Aplanado (Flatten)
        self.flatten = nn.Flatten()
        
        # Clasificador Denso: 128 * 4 * 4 = 2048 neuronas de entrada
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        out = self.classifier(x)
        return out

def validate_tensor_flow():
    """Valida el paso hacia adelante imprimiendo las dimensiones y comprobando aserciones."""
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    model = ConvNetCIFAR(num_classes=10)
    model.eval()

    print("--- Verificación de Flujo de Tensores 4D ---")
    print(f"Shape Entrada (4D): {dummy_input.shape}")

    with torch.no_grad():
        out1 = model.block1(dummy_input)
        print(f"Post-Bloque 1: {out1.shape}")
        assert out1.shape == (batch_size, 32, 16, 16), "Error dimensional en Bloque 1"

        out2 = model.block2(out1)
        print(f"Post-Bloque 2: {out2.shape}")
        assert out2.shape == (batch_size, 64, 8, 8), "Error dimensional en Bloque 2"

        out3 = model.block3(out2)
        print(f"Post-Bloque 3: {out3.shape}")
        assert out3.shape == (batch_size, 128, 4, 4), "Error dimensional en Bloque 3"

        flat = model.flatten(out3)
        print(f"Post-Flatten: {flat.shape}")
        assert flat.shape == (batch_size, 128 * 4 * 4), "Error dimensional en Flatten"

        logits = model.classifier(flat)
        print(f"Logits Salida: {logits.shape}")
        assert logits.shape == (batch_size, 10), "Error dimensional en salida de Clasificador"

    print("Validación completada: Todas las dimensiones cumplen el contrato sin errores.")

if __name__ == '__main__':
    validate_tensor_flow()