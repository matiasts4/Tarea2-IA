"""
Script para preparar y guardar el dataset MNIST
Ejecutar UNA SOLA VEZ antes de los experimentos
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preparar_y_guardar_dataset(output_file='mnist_prepared.npz', n_samples=15000, test_size=0.2):
    """
    Descarga MNIST, prepara los datos y los guarda localmente
    
    Args:
        output_file: Nombre del archivo de salida
        n_samples: Número de muestras a tomar
        test_size: Proporción del conjunto de test
    """
    logger.info("="*60)
    logger.info("PREPARACIÓN DE DATASET MNIST")
    logger.info("="*60)
    
    # 1. Descargar MNIST
    logger.info("Descargando MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data
    y = mnist.target.astype(int)
    logger.info(f"Dataset original: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # 2. Tomar muestra aleatoria
    logger.info(f"Tomando muestra de {n_samples} registros...")
    np.random.seed(42)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
    y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    
    # Convertir a numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Muestra seleccionada: {X.shape}")
    logger.info(f"Clases únicas: {sorted(np.unique(y))}")
    
    # 3. Split train/test
    logger.info(f"Dividiendo en train ({(1-test_size)*100:.0f}%) y test ({test_size*100:.0f}%)...")
    n_train = int(n_samples * (1 - test_size))
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    logger.info(f"Train: {X_train.shape[0]} muestras")
    logger.info(f"Test: {X_test.shape[0]} muestras")
    
    # 4. Normalizar con StandardScaler
    logger.info("Normalizando con StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Media train: {X_train_scaled.mean():.4f}, Std train: {X_train_scaled.std():.4f}")
    logger.info(f"Media test: {X_test_scaled.mean():.4f}, Std test: {X_test_scaled.std():.4f}")
    
    # 5. Guardar todo en un archivo
    logger.info(f"Guardando datos en '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
    )
    
    logger.info("="*60)
    logger.info("✓ DATASET PREPARADO Y GUARDADO EXITOSAMENTE")
    logger.info("="*60)
    logger.info(f"Archivo: {output_file}")
    logger.info(f"Tamaño: {np.round(os.path.getsize(output_file) / 1024 / 1024, 2)} MB")
    logger.info("\nAhora puedes ejecutar tarea2_clustering.py y tarea2_supervisado.py")
    logger.info("sin necesidad de descargar el dataset cada vez.")

if __name__ == "__main__":
    preparar_y_guardar_dataset()
