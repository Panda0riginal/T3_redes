import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader
from download_data import descargar_y_preparar
from model_clf import Clasificador

def fijar_semilla(seed):
    """
    Fija la semilla en todos los generadores de números aleatorios
    para garantizar la reproducibilidad del experimento.
    """
    # 1. Semilla nativa de Python
    random.seed(seed)
    
    # 2. Semilla de Numpy (muchas librerías internas lo usan)
    np.random.seed(seed)
    
    # 3. Semilla de PyTorch en CPU
    torch.manual_seed(seed)
    
    # 4. Semilla de PyTorch en GPU y configuraciones deterministas
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

def train(model, data_train, n_epochs, batch_size, lr, device, seed):
    #fijamos la semilla
    fijar_semilla(seed)
    
    # Configurar el dispositivo 
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    model.to(device)

    # Cargar datos
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    
    # Configuracion entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train() # Activa el modo de entrenamiento (activa Dropout y BatchNorm)
    for epoch in range(n_epochs):
        running_loss = 0.0
        for imagenes, etiquetas in train_loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            
            # Paso hacia adelante (Forward pass)
            predicciones = model(imagenes)
            loss = criterion(predicciones, etiquetas)
            
            # Paso hacia atrás y optimización (Backward pass)
            optimizer.zero_grad() 
            loss.backward()     
            optimizer.step()      
            
            # Estadísticas métricas
            running_loss += loss.item() * imagenes.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Época [{epoch+1}/{n_epochs}] -> Pérdida: {epoch_loss:.4f}")
    return


# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clasificador.")    
    # ARGUMENTOS: n_epochs, batch_size, lr, device, seed
    # Epocas
    parser.add_argument('--n_epochs', type=int, default=2,
        help='Numero de epocas del entrenamiento'
    )    
    # Batch
    parser.add_argument('--batch_size', type=int, default=60, 
        help='Número del tamano de los batches'
    )
    # Semilla
    parser.add_argument('--seed', type=int, default=69, 
        help='seed de aleatoriedad para reproducibilidad'
    )
    # Learning rate
    parser.add_argument('--lr', type=float, default=0.001, 
        help='Learning rate del alg. de optimizacion'
    )
    # Device
    parser.add_argument('--device',type=str,default='cpu', 
        help='Dispositivo de computo'
    )
    # Empaquetar argumentos
    args = vars(parser.parse_args())
    fijar_semilla(args['seed'])

    # Descargar datos y inicializar modelo
    dataset_train, dataset_test, _ = descargar_y_preparar()
    modelo = Clasificador()

    # Comenzar el entrenamiento
    print("\nIniciando entrenamiento del clasificador...")
    print("Parametros:")
    print(args)
    train(model=modelo, data_train=dataset_train, **args)
    print("Entrenamiento finalizado. Guardando pesos...")
    modelo.save()

        