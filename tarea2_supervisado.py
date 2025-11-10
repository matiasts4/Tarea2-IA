"""
Tarea 2 - Parte 2: Aprendizaje Supervisado
Entrenamiento paralelo de Regresión Logística y SVM con eliminación progresiva
"""

import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
import os
from datetime import datetime

np.random.seed(42)

# Configurar logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'supervisado_{timestamp}.log')

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
    - Columnas: features..., label/target
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
    
    # Muestra de 15,000 para eficiencia
    logger.info("Tomando muestra de 15,000 registros...")
    indices = np.random.choice(len(X), 15000, replace=False)
    X = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
    y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    
    logger.info(f"Dataset cargado: {X.shape[0]} filas, {X.shape[1]} columnas")
    logger.info(f"Clases únicas: {sorted(np.unique(y))}")
    
    return X, y

def preparar_datos(X, y, test_size=0.2):
    """Divide y normaliza los datos"""
    logger.info("\n" + "="*60)
    logger.info("PREPARACIÓN DE DATOS")
    logger.info("="*60)
    
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
    
    logger.info("Normalizando datos con StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Media train: {X_train_scaled.mean():.4f}, Std train: {X_train_scaled.std():.4f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def cargar_configuraciones(archivo='config.json'):
    """Carga configuraciones desde archivo JSON"""
    with open(archivo, 'r') as f:
        configs = json.load(f)
    return configs

class ModeloEntrenable:
    """Clase para entrenar modelos con evaluación periódica"""
    
    def __init__(self, tipo_modelo, config, X_train, y_train):
        self.tipo_modelo = tipo_modelo
        self.config = config
        self.nombre = config['name']
        self.X_train = X_train
        self.y_train = y_train
        self.historial = []
        self.activo = True
        
    def entrenar_epoca(self, epoca):
        """Entrena una época y evalúa"""
        if not self.activo:
            return None
        
        try:
            if self.tipo_modelo == 'logistic_regression':
                modelo = LogisticRegression(
                    max_iter=self.config['max_iter'] // 10,  # Dividir en épocas
                    C=self.config['C'],
                    solver=self.config['solver'],
                    random_state=42,
                    warm_start=True
                )
            else:  # SVM
                modelo = SVC(
                    C=self.config['C'],
                    kernel=self.config['kernel'],
                    gamma=self.config['gamma'],
                    max_iter=self.config['max_iter'] // 10,
                    random_state=42
                )
            
            # Entrenar
            inicio = datetime.now()
            modelo.fit(self.X_train, self.y_train)
            tiempo = (datetime.now() - inicio).total_seconds()
            
            # Evaluar en train
            y_pred = modelo.predict(self.X_train)
            accuracy = accuracy_score(self.y_train, y_pred)
            
            self.historial.append({
                'epoca': epoca,
                'accuracy': accuracy,
                'modelo': modelo,
                'tiempo': tiempo
            })
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error en {self.nombre} época {epoca}: {e}")
            return None
    
    def desactivar(self):
        """Marca el modelo como inactivo"""
        self.activo = False
    
    def obtener_mejor_modelo(self):
        """Retorna el mejor modelo entrenado"""
        if not self.historial:
            return None
        mejor = max(self.historial, key=lambda x: x['accuracy'])
        return mejor['modelo']

def entrenar_modelo_paralelo(args):
    """Función para entrenar un modelo (usada en paralelo)"""
    tipo, config, X_train, y_train, epoca = args
    modelo_entrenable = ModeloEntrenable(tipo, config, X_train, y_train)
    accuracy = modelo_entrenable.entrenar_epoca(epoca)
    return modelo_entrenable.nombre, accuracy, modelo_entrenable

def entrenar_con_eliminacion(configs, X_train, y_train, epocas_totales=50, evaluar_cada=5):
    """
    Entrena múltiples modelos con eliminación progresiva
    """
    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO CON ELIMINACIÓN PROGRESIVA")
    logger.info("="*60)
    
    # Crear modelos entrenables
    modelos = []
    
    for config in configs['logistic_regression']:
        modelo = ModeloEntrenable('logistic_regression', config, X_train, y_train)
        modelos.append(modelo)
        logger.info(f"Creado: {config['name']} (Regresión Logística)")
    
    for config in configs['svm']:
        modelo = ModeloEntrenable('svm', config, X_train, y_train)
        modelos.append(modelo)
        logger.info(f"Creado: {config['name']} (SVM)")
    
    logger.info(f"\nTotal de configuraciones: {len(modelos)}")
    logger.info(f"Épocas totales: {epocas_totales}")
    logger.info(f"Evaluación cada: {evaluar_cada} épocas")
    
    # Entrenamiento por épocas
    for epoca in range(1, epocas_totales + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ÉPOCA {epoca}/{epocas_totales}")
        logger.info(f"{'='*60}")
        
        # Entrenar modelos activos EN PARALELO
        modelos_activos = [m for m in modelos if m.activo]
        logger.info(f"Modelos activos: {len(modelos_activos)}")
        
        # Preparar argumentos para entrenamiento paralelo
        args_paralelo = [
            (m.tipo_modelo, m.config, m.X_train, m.y_train, epoca)
            for m in modelos_activos
        ]
        
        # Entrenar en paralelo usando ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(len(modelos_activos), 4)) as executor:
            # Enviar todas las tareas
            futures = {executor.submit(entrenar_modelo_paralelo, args): args[1]['name'] 
                      for args in args_paralelo}
            
            # Recolectar resultados
            for future in as_completed(futures):
                nombre_modelo = futures[future]
                try:
                    nombre, accuracy, modelo_actualizado = future.result()
                    if accuracy is not None:
                        # Actualizar el modelo en la lista con el historial actualizado
                        for m in modelos_activos:
                            if m.nombre == nombre:
                                m.historial = modelo_actualizado.historial
                                break
                        logger.info(f"  ✓ {nombre}: {accuracy:.4f}")
                except Exception as e:
                    logger.error(f"  ✗ Error en {nombre_modelo}: {e}")
        
        # Cada 5 épocas, eliminar el peor
        if epoca % evaluar_cada == 0 and len(modelos_activos) > 2:
            logger.info(f"\n  Evaluando para eliminación...")
            # Obtener accuracy actual de cada modelo
            accuracies = []
            for modelo in modelos_activos:
                if modelo.historial:
                    acc = modelo.historial[-1]['accuracy']
                    accuracies.append((modelo, acc))
            
            # Encontrar el peor
            if accuracies:
                peor_modelo, peor_acc = min(accuracies, key=lambda x: x[1])
                peor_modelo.desactivar()
                logger.info(f"  ❌ Eliminado: {peor_modelo.nombre} (accuracy: {peor_acc:.4f})")
                logger.info(f"  Modelos restantes: {len([m for m in modelos if m.activo])}")
    
    # Retornar los 2 mejores
    modelos_activos = [m for m in modelos if m.activo]
    modelos_con_score = [(m, m.historial[-1]['accuracy']) for m in modelos_activos if m.historial]
    modelos_con_score.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("\n" + "="*60)
    logger.info("MODELOS FINALES SELECCIONADOS")
    logger.info("="*60)
    for i, (m, acc) in enumerate(modelos_con_score[:2], 1):
        logger.info(f"{i}. {m.nombre}: {acc:.4f}")
    
    return modelos_con_score[:2], modelos

def evaluar_mejores(mejores_modelos, X_test, y_test):
    """Evalúa los mejores modelos en el conjunto de test"""
    logger.info("\n" + "="*60)
    logger.info("EVALUACIÓN EN CONJUNTO DE TEST")
    logger.info("="*60)
    
    resultados = []
    
    for i, (modelo_entrenable, train_acc) in enumerate(mejores_modelos, 1):
        logger.info(f"\n{i}. {modelo_entrenable.nombre}")
        logger.info(f"   Tipo: {modelo_entrenable.tipo_modelo}")
        logger.info(f"   Configuración: {modelo_entrenable.config}")
        logger.info(f"   Accuracy en train: {train_acc:.4f}")
        
        # Obtener mejor modelo
        mejor_modelo = modelo_entrenable.obtener_mejor_modelo()
        
        if mejor_modelo is None:
            logger.warning("   No hay modelo entrenado")
            continue
        
        # Predecir en test
        inicio = datetime.now()
        y_pred = mejor_modelo.predict(X_test)
        tiempo_pred = (datetime.now() - inicio).total_seconds()
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"\n   Métricas en test:")
        logger.info(f"   - Accuracy:  {accuracy:.4f}")
        logger.info(f"   - Precision: {precision:.4f}")
        logger.info(f"   - Recall:    {recall:.4f}")
        logger.info(f"   - F1-Score:  {f1:.4f}")
        logger.info(f"   - Tiempo predicción: {tiempo_pred:.4f}s")
        
        resultados.append({
            'nombre': modelo_entrenable.nombre,
            'tipo': modelo_entrenable.tipo_modelo,
            'config': modelo_entrenable.config,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tiempo_prediccion': tiempo_pred
        })
    
    return resultados

def analizar_hiperparametros(resultados, todos_modelos):
    """Analiza el impacto de los hiperparámetros"""
    logger.info("\n" + "="*60)
    logger.info("ANÁLISIS DE HIPERPARÁMETROS")
    logger.info("="*60)
    
    logger.info("""
REGRESIÓN LOGÍSTICA:
- C (regularización): Valores más altos (menos regularización) pueden mejorar
  el ajuste en datos complejos, pero aumentan el riesgo de overfitting.
- solver: 'lbfgs' es eficiente para datasets pequeños/medianos, 'saga' 
  soporta regularización L1 y es mejor para datasets grandes.
- learning_rate: Afecta la velocidad de convergencia. Valores muy altos
  pueden causar inestabilidad, valores muy bajos ralentizan el entrenamiento.

SVM:
- C (parámetro de penalización): Controla el trade-off entre margen y
  errores de clasificación. C alto = menos errores pero margen pequeño.
- kernel: 'rbf' captura relaciones no lineales, 'linear' es más simple
  y rápido para datos linealmente separables.
- gamma: Controla la influencia de cada punto de entrenamiento. Valores
  altos pueden causar overfitting.

OBSERVACIONES DE LOS RESULTADOS:
    """)
    
    # Análisis específico de los mejores modelos
    for r in resultados:
        logger.info(f"\n{r['nombre']} ({r['tipo']}):")
        logger.info(f"  Accuracy: {r['accuracy']:.4f}")
        logger.info(f"  F1-Score: {r['f1']:.4f}")
        if r['tipo'] == 'logistic_regression':
            logger.info(f"  - C={r['config']['C']}: " + 
                       ("Baja regularización, mayor capacidad" if r['config']['C'] > 1 
                        else "Alta regularización, más generalización"))
            logger.info(f"  - Solver: {r['config']['solver']}")
        else:
            logger.info(f"  - Kernel={r['config']['kernel']}, C={r['config']['C']}")
            logger.info(f"    Configuración {'compleja' if r['config']['kernel']=='rbf' else 'simple'}")

def visualizar_progreso(todos_modelos):
    """Visualiza el progreso del entrenamiento"""
    print("\n" + "="*60)
    print("PROGRESO DEL ENTRENAMIENTO")
    print("="*60)
    
    for modelo in todos_modelos:
        if modelo.historial:
            epocas = [h['epoca'] for h in modelo.historial]
            accuracies = [h['accuracy'] for h in modelo.historial]
            estado = "✓ Activo" if modelo.activo else "✗ Eliminado"
            print(f"\n{modelo.nombre} ({estado}):")
            print(f"  Épocas entrenadas: {len(epocas)}")
            print(f"  Accuracy final: {accuracies[-1]:.4f}")
            print(f"  Mejora: {accuracies[-1] - accuracies[0]:.4f}")

def visualizar_progreso(todos_modelos):
    """Visualiza el progreso del entrenamiento"""
    logger.info("\n" + "="*60)
    logger.info("PROGRESO DEL ENTRENAMIENTO")
    logger.info("="*60)
    
    for modelo in todos_modelos:
        if modelo.historial:
            epocas = [h['epoca'] for h in modelo.historial]
            accuracies = [h['accuracy'] for h in modelo.historial]
            tiempos = [h['tiempo'] for h in modelo.historial]
            estado = "✓ Activo" if modelo.activo else "✗ Eliminado"
            logger.info(f"\n{modelo.nombre} ({estado}):")
            logger.info(f"  Épocas entrenadas: {len(epocas)}")
            logger.info(f"  Accuracy inicial: {accuracies[0]:.4f}")
            logger.info(f"  Accuracy final: {accuracies[-1]:.4f}")
            logger.info(f"  Mejora: {accuracies[-1] - accuracies[0]:.4f}")
            logger.info(f"  Tiempo total: {sum(tiempos):.2f}s")

def main(archivo_dataset=None):
    """
    Función principal
    
    Args:
        archivo_dataset: Ruta al archivo CSV del dataset (opcional)
    """
    logger.info("="*60)
    logger.info("TAREA 2 - PARTE 2: APRENDIZAJE SUPERVISADO")
    logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    try:
        # Cargar datos
        X, y = cargar_datos(archivo_dataset)
        X_train, X_test, y_train, y_test = preparar_datos(X, y)
        
        # Cargar configuraciones
        configs = cargar_configuraciones()
        
        logger.info("\nConfiguraciones cargadas:")
        logger.info(f"  Regresión Logística: {len(configs['logistic_regression'])} configs")
        logger.info(f"  SVM: {len(configs['svm'])} configs")
        
        # Entrenar con eliminación progresiva
        mejores, todos = entrenar_con_eliminacion(configs, X_train, y_train)
        
        # Visualizar progreso
        visualizar_progreso(todos)
        
        # Evaluar mejores en test
        resultados = evaluar_mejores(mejores, X_test, y_test)
        
        # Analizar hiperparámetros
        analizar_hiperparametros(resultados, todos)
        
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
