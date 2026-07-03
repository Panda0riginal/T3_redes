import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from download_data import descargar_y_preparar_mnist
from model_clf import Clasificador

# =====================================================================
# 2. ALGORITMO DE ENTRENAMIENTO
# =====================================================================
def train_classifier(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train() # Activa el modo de entrenamiento (activa Dropout y BatchNorm)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imagenes, etiquetas in train_loader:
            # Mover tensores al hardware seleccionado (CPU o GPU)
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            
            # Paso hacia adelante (Forward pass)
            predicciones = model(imagenes)
            loss = criterion(predicciones, etiquetas)
            
            # Paso hacia atrás y optimización (Backward pass)
            optimizer.zero_grad() # Resetea los gradientes acumulados
            loss.backward()       # Calcula los nuevos gradientes
            optimizer.step()      # Actualiza los pesos de la red
            
            # Estadísticas métricas
            running_loss += loss.item() * imagenes.size(0)
            _, predicted = torch.max(predicciones, 1)
            total += etiquetas.size(0)
            correct += (predicted == etiquetas).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100
        print(f"Época [{epoch+1}/{epochs}] -> Pérdida: {epoch_loss:.4f} | Precisión: {epoch_acc:.2f}%")

# =====================================================================
# 3. ALGORITMO DE TEST (EVALUACIÓN)
# =====================================================================
def evaluate_classifier(model, test_loader, device):
    model.eval() # Activa el modo de evaluación (desactiva Dropout y congela BatchNorm)
    correct = 0
    total = 0
    
    # torch.no_grad() evita que PyTorch consuma memoria calculando gradientes (no los necesitamos para testear)
    with torch.no_grad():
        for imagenes, etiquetas in test_loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            predicciones = model(imagenes)
            
            _, predicted = torch.max(predicciones, 1)
            total += etiquetas.size(0)
            correct += (predicted == etiquetas).sum().item()
            
    accuracy = (correct / total) * 100
    print(f"\n=========================================")
    print(f"🎯 Precisión Final en el Set de Test: {accuracy:.2f}%")
    print(f"=========================================")
    return accuracy



# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    # Configurar el dispositivo de hardware de forma inteligente
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ejecutando en el dispositivo: {device}\n")
    
    # Cargar los datos usando tu script modular anterior
    dataset_train, dataset_test, _ = descargar_y_preparar_mnist()
    
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    
    # Inicializar el modelo, función de pérdida y optimizador
    model = Clasificador().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Comenzar el entrenamiento
    print("\nIniciando entrenamiento del clasificador...")
    train_classifier(model, train_loader, criterion, optimizer, device, epochs=5)
    
    # Evaluar con datos no vistos
    evaluate_classifier(model, test_loader, device)
    
    # Guardar los pesos del modelo (se guardan en la raíz, pero el .gitignore los filtrará)
    torch.save(model.state_dict(), "mnist_classifier.pt")
    print("Pesos del clasificador guardados como 'mnist_classifier.pt'")