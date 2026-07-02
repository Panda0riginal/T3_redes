import torchvision
import torchvision.transforms as transforms

def descargar_y_preparar_mnist():
    """
    Descarga el dataset MNIST y define las transformaciones.
    """
    print(f"Verificando/Descargando dataset MNIST")

    # El preprocesamiento: 
    # 1. ToTensor() convierte la imagen a tensor y escala a [0.0, 1.0]
    # 2. Normalize() escala los valores al rango [-1.0, 1.0]
    transformacion = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Descargar el conjunto de entrenamiento
    dataset_train = torchvision.datasets.MNIST(
        root='./', 
        train=True, 
        download=True, 
        transform=transformacion
    )
    
    # Descargar el conjunto de prueba (útil para validación/evaluación)
    dataset_test = torchvision.datasets.MNIST(
        root='./', 
        train=False, 
        download=True, 
        transform=transformacion
    )

    print("¡Descarga completada con éxito!")
    print(f"Total imágenes entrenamiento: {len(dataset_train)}")
    print(f"Total imágenes prueba: {len(dataset_test)}")
    
    # Retornamos los datasets por si este script es importado desde train.py
    return dataset_train, dataset_test

if __name__ == "__main__":
    # Al ejecutar este script directamente en la consola, descargará los datos.
    descargar_y_preparar_mnist()