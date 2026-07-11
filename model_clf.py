import torch
import torch.nn as nn

# clasificador
class Clasificador(nn.Module):
    # Creacion de arquitectura del modelo
    def __init__(self):
        super().__init__()
        # Entrada: [Batch, 1, 28, 28]
        self.features = nn.Sequential(
            # Bloque Convolucional 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'), # -> [Batch, 32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> [Batch, 32, 14, 14]

            # Bloque Convolucional 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'), # -> [Batch, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> [Batch, 64, 7, 7]
        )
        
        # Clasificador MLP
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.20), # Regularización para evitar sobreajuste
            nn.Linear(128, 10) # 10 clases de salida (números del 0 al 9)
        )

    # Forward del modelo
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) # Aplana las dimensiones espaciales: [Batch, 64 * 7 * 7]
        x = self.classifier(x)
        return x
    
    # Guarda los pesos
    def save(self, ruta="clasificador.pt"):
        """
        Guarda los pesos del modelo de forma segura, asegurando 
        que se almacenen en la CPU para máxima compatibilidad.
        """
        # 1. Asegurar la compatibilidad de hardware (CPU/GPU)
        pesos = {k: v.cpu() for k, v in self.state_dict().items()}
        
        # 2. Guardar el archivo
        torch.save(pesos, ruta)
        return 
    
    def load(self, ruta="clasificador.pt", device="cpu"):
        """
        Carga los pesos en el modelo, asegurando compatibilidad 
        independientemente del hardware donde se guardaron.
        """
        pesos = torch.load(ruta, map_location=device)
        self.load_state_dict(pesos)
        return
