"""
Tarea 2 - Parte 1: Clustering
Comparación de K-Means, K-Means++ y MeanShift
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
import os
from datetime import datetime

# Configuración
np.random.seed(42)

# Configurar logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'clustering_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cargar_datos(archivo_local=None):
    """
    Carga el dataset desde archivo local o descarga MNIST
    
    Formato esperado del archivo CSV:
    - Columnas: pixel0, pixel1, ..., pixel783, label
    - O cualquier dataset con features y una columna 'label' o 'target'
    """
    logger.info("="*60)
    logger.info("CARGANDO DATASET")
    logger.info("="*60)
    
    if archivo_local and os.path.exists(archivo_local):
        logger.info(f"Cargando dataset desde archivo local: {archivo_local}")
        try:
            df = pd.read_csv(archivo_local)
            logger.info(f"Archivo cargado exitosamente: {df.shape}")
            
            # Detectar columna de etiquetas
            label_col = None
            for col in ['label', 'target', 'y', 'class']:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                # Asumir que la última columna es la etiqueta
                label_col = df.columns[-1]
                logger.warning(f"No se encontró columna de etiquetas, usando última columna: {label_col}")
            
            y = df[label_col].astype(int)
            X = df.drop(columns=[label_col])
            
            logger.info(f"Features: {X.shape[1]} columnas")
            logger.info(f"Etiquetas: {label_col}")
            logger.info(f"Clases únicas: {sorted(y.unique())}")
            
            # Validar requisitos
            if X.shape[0] < 10000:
                logger.warning(f"Dataset tiene {X.shape[0]} filas, se requieren al menos 10,000")
            if X.shape[1] < 7:
                logger.warning(f"Dataset tiene {X.shape[1]} columnas, se requieren al menos 7")
            if len(y.unique()) <= 2:
                logger.warning(f"Dataset tiene {len(y.unique())} clases, se requieren más de 2")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error al cargar archivo local: {e}")
            logger.info("Intentando descargar MNIST...")
    
    # Descargar MNIST si no hay archivo local
    logger.info("Descargando dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data
    y = mnist.target.astype(int)
    
    # Tomar una muestra de 15,000 para eficiencia
    logger.info("Tomando muestra de 15,000 registros...")
    indices = np.random.choice(len(X), 15000, replace=False)
    X = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
    y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    
    logger.info(f"Dataset cargado: {X.shape[0]} filas, {X.shape[1]} columnas")
    logger.info(f"Clases únicas: {sorted(np.unique(y))}")
    
    return X, y

def preparar_datos(X, y, test_size=0.2):
    """
    Divide los datos en train/test y normaliza
    """
    logger.info("\n" + "="*60)
    logger.info("PREPARACIÓN DE DATOS")
    logger.info("="*60)
    
    # Split train/test
    n_samples = len(X)
    n_train = int(n_samples * (1 - test_size))
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
    X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
    y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
    y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
    
    logger.info(f"Train set: {X_train.shape[0]} muestras ({(1-test_size)*100:.0f}%)")
    logger.info(f"Test set: {X_test.shape[0]} muestras ({test_size*100:.0f}%)")
    
    # Normalizar
    logger.info("Normalizando datos con StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Media train: {X_train_scaled.mean():.4f}, Std train: {X_train_scaled.std():.4f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def configuraciones_clustering(X_train_sample=None):
    """
    Define 4 configuraciones para cada algoritmo (12 total)
    Para MeanShift, estima el bandwidth óptimo basado en los datos
    """
    # Estimar bandwidth para MeanShift si se proporciona muestra
    if X_train_sample is not None:
        logger.info("Estimando bandwidth óptimo para MeanShift...")
        # Usar una muestra pequeña para estimar (más rápido)
        sample_size = min(2000, len(X_train_sample))
        sample_indices = np.random.choice(len(X_train_sample), sample_size, replace=False)
        X_sample = X_train_sample[sample_indices]
        
        bandwidth_estimado = estimate_bandwidth(X_sample, quantile=0.2, n_samples=500)
        logger.info(f"Bandwidth estimado: {bandwidth_estimado:.2f}")
        
        # Crear configuraciones alrededor del valor estimado
        meanshift_configs = [
            {'bandwidth': bandwidth_estimado * 0.8, 'bin_seeding': True},
            {'bandwidth': bandwidth_estimado * 1.0, 'bin_seeding': True},
            {'bandwidth': bandwidth_estimado * 1.2, 'bin_seeding': False},
            {'bandwidth': bandwidth_estimado * 1.5, 'bin_seeding': True},
        ]
    else:
        # Valores por defecto más altos para MNIST
        meanshift_configs = [
            {'bandwidth': 10.0, 'bin_seeding': True},
            {'bandwidth': 15.0, 'bin_seeding': True},
            {'bandwidth': 20.0, 'bin_seeding': False},
            {'bandwidth': 25.0, 'bin_seeding': True},
        ]
    
    configs = {
        'KMeans': [
            {'n_clusters': 10, 'init': 'random', 'n_init': 10, 'max_iter': 300},
            {'n_clusters': 15, 'init': 'random', 'n_init': 20, 'max_iter': 300},
            {'n_clusters': 20, 'init': 'random', 'n_init': 10, 'max_iter': 500},
            {'n_clusters': 25, 'init': 'random', 'n_init': 15, 'max_iter': 300},
        ],
        'KMeans++': [
            {'n_clusters': 10, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300},
            {'n_clusters': 15, 'init': 'k-means++', 'n_init': 20, 'max_iter': 300},
            {'n_clusters': 20, 'init': 'k-means++', 'n_init': 10, 'max_iter': 500},
            {'n_clusters': 25, 'init': 'k-means++', 'n_init': 15, 'max_iter': 300},
        ],
        'MeanShift': meanshift_configs
    }
    return configs

def entrenar_clustering(X_train, configs):
    """
    Entrena todos los modelos de clustering y evalúa con Silhouette Score
    """
    resultados = []
    
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO DE MODELOS DE CLUSTERING")
    logger.info("="*60)
    
    for algoritmo, config_list in configs.items():
        logger.info(f"\n{algoritmo}:")
        for i, config in enumerate(config_list, 1):
            logger.info(f"  Configuración {i}: {config}")
            
            try:
                if algoritmo in ['KMeans', 'KMeans++']:
                    modelo = KMeans(**config, random_state=42)
                else:  # MeanShift
                    modelo = MeanShift(**config)
                
                # Entrenar
                inicio = datetime.now()
                labels = modelo.fit_predict(X_train)
                tiempo = (datetime.now() - inicio).total_seconds()
                
                # Evaluar con Silhouette Score
                score = silhouette_score(X_train, labels, sample_size=5000)
                
                resultados.append({
                    'algoritmo': algoritmo,
                    'config_num': i,
                    'config': config,
                    'modelo': modelo,
                    'labels_train': labels,
                    'silhouette_score': score,
                    'n_clusters': len(np.unique(labels)),
                    'tiempo_entrenamiento': tiempo
                })
                
                logger.info(f"    ✓ Silhouette Score: {score:.4f}")
                logger.info(f"    ✓ Clusters encontrados: {len(np.unique(labels))}")
                logger.info(f"    ✓ Tiempo: {tiempo:.2f}s")
                
            except Exception as e:
                logger.error(f"    ✗ Error en {algoritmo} config {i}: {e}")
    
    return resultados

def seleccionar_mejores(resultados, top_k=3):
    """
    Selecciona las mejores configuraciones según Silhouette Score
    """
    logger.info("\n" + "="*60)
    logger.info(f"SELECCIÓN DE TOP {top_k} CONFIGURACIONES")
    logger.info("="*60)
    
    resultados_sorted = sorted(resultados, key=lambda x: x['silhouette_score'], reverse=True)
    
    for i, r in enumerate(resultados_sorted[:top_k], 1):
        logger.info(f"{i}. {r['algoritmo']} - Config {r['config_num']}")
        logger.info(f"   Silhouette Score: {r['silhouette_score']:.4f}")
        logger.info(f"   Clusters: {r['n_clusters']}")
        logger.info(f"   Tiempo: {r['tiempo_entrenamiento']:.2f}s")
    
    return resultados_sorted[:top_k]

def evaluar_en_test(mejores_modelos, X_test, y_test, y_train):
    """
    Evalúa los mejores modelos en el conjunto de test
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUACIÓN EN CONJUNTO DE TEST")
    logger.info("="*60)
    
    for i, resultado in enumerate(mejores_modelos, 1):
        logger.info(f"\n{i}. {resultado['algoritmo']} - Config {resultado['config_num']}")
        logger.info(f"   Silhouette Score (train): {resultado['silhouette_score']:.4f}")
        
        modelo = resultado['modelo']
        labels_test = modelo.predict(X_test)
        
        # Mapear clusters a etiquetas dominantes usando datos de train
        labels_train = resultado['labels_train']
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Para cada cluster, encontrar la etiqueta dominante en train
        cluster_to_label = {}
        for cluster_id in np.unique(labels_train):
            mask = labels_train == cluster_id
            labels_en_cluster = y_train_array[mask]
            if len(labels_en_cluster) > 0:
                # Etiqueta más frecuente en este cluster
                etiqueta_dominante = Counter(labels_en_cluster).most_common(1)[0][0]
                cluster_to_label[cluster_id] = etiqueta_dominante
                logger.info(f"   Cluster {cluster_id} -> Etiqueta {etiqueta_dominante} ({np.sum(mask)} muestras)")
        
        # Asignar etiquetas predichas
        y_pred = np.array([cluster_to_label.get(c, 0) for c in labels_test])
        
        # Calcular accuracy
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        accuracy = np.mean(y_pred == y_test_array)
        logger.info(f"\n   ✓ Accuracy en test: {accuracy:.4f}")
        
        # Análisis de clusters
        logger.info(f"   ✓ Clusters únicos en test: {len(np.unique(labels_test))}")
        
        # Distribución de predicciones
        logger.info(f"   ✓ Distribución de predicciones: {Counter(y_pred)}")

def analisis_final():
    """
    Análisis sobre la razonabilidad del método
    """
    logger.info("\n" + "="*60)
    logger.info("ANÁLISIS FINAL")
    logger.info("="*60)
    logger.info("""
El procedimiento de asignar etiquetas mediante clustering presenta las siguientes
características:

VENTAJAS:
- No requiere etiquetas durante el entrenamiento
- Puede descubrir patrones no supervisados en los datos
- Útil cuando las etiquetas son costosas de obtener

LIMITACIONES:
- Los clusters no necesariamente corresponden a las clases reales
- Un cluster puede contener muestras de múltiples clases
- La asignación de etiqueta dominante puede ser arbitraria
- El número de clusters puede no coincidir con el número de clases

CONCLUSIÓN:
Este método es razonable como técnica exploratoria o para pre-etiquetar datos,
pero no debe usarse como único método de clasificación cuando se requiere
precisión. Es más útil para:
1. Análisis exploratorio de datos
2. Detección de anomalías
3. Pre-procesamiento antes de etiquetado manual
4. Reducción de dimensionalidad
    """)

def main(archivo_dataset=None):
    """
    Función principal
    
    Args:
        archivo_dataset: Ruta al archivo CSV del dataset (opcional)
                        Si no se proporciona, descarga MNIST
    """
    logger.info("="*60)
    logger.info("TAREA 2 - PARTE 1: CLUSTERING")
    logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    try:
        # Cargar datos
        X, y = cargar_datos(archivo_dataset)
        
        # Preparar datos
        X_train, X_test, y_train, y_test = preparar_datos(X, y)
        
        # Configuraciones (pasar muestra de datos para estimar bandwidth)
        configs = configuraciones_clustering(X_train)
        
        # Justificación de parámetros
        logger.info("\n" + "="*60)
        logger.info("JUSTIFICACIÓN DE PARÁMETROS")
        logger.info("="*60)
        logger.info("""
K-Means y K-Means++:
- n_clusters: Variamos entre 10-25 para explorar diferentes granularidades
- init: 'random' vs 'k-means++' para comparar estrategias de inicialización
- n_init: Múltiples inicializaciones para evitar mínimos locales
- max_iter: Suficientes iteraciones para convergencia

MeanShift:
- bandwidth: Controla el tamaño de la ventana de búsqueda. Se estima automáticamente
  basado en los datos y se prueban variaciones (0.8x, 1.0x, 1.2x, 1.5x)
- bin_seeding: Acelera el algoritmo mediante discretización
        """)
        
        # Entrenar modelos
        resultados = entrenar_clustering(X_train, configs)
        
        # Seleccionar mejores
        mejores = seleccionar_mejores(resultados, top_k=3)
        
        # Evaluar en test
        evaluar_en_test(mejores, X_test, y_test, y_train)
        
        # Análisis final
        analisis_final()
        
        logger.info("\n" + "="*60)
        logger.info("EJECUCIÓN COMPLETADA EXITOSAMENTE")
        logger.info(f"Log guardado en: {log_file}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR FATAL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Puedes especificar un archivo de dataset aquí
    # Ejemplo: main('mi_dataset.csv')
    main()
