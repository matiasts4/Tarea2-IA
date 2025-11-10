"""
Script para generar gráficos de análisis de Clustering y Supervisado
Genera visualizaciones profesionales para el informe de la Tarea 2
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Crear directorio para gráficos
os.makedirs('graficos', exist_ok=True)


def extraer_datos_clustering(log_path):
    """Extrae datos de clustering desde el log"""
    datos = {
        'configuraciones': [],
        'silhouette_scores': [],
        'clusters': [],
        'tiempos': [],
        'algoritmos': [],
        'top3': [],
        'test_accuracy': []
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    # Extraer configuraciones y métricas
    patron_config = r'Configuración \d+: ({.*?})'
    patron_silhouette = r'Silhouette Score: ([-\d.]+)'
    patron_clusters = r'Clusters encontrados: (\d+)'
    patron_tiempo = r'Tiempo: ([\d.]+)s'
    
    lineas = contenido.split('\n')
    algoritmo_actual = None
    
    for i, linea in enumerate(lineas):
        # Detectar algoritmo
        if 'KMeans:' in linea and 'KMeans++' not in linea:
            algoritmo_actual = 'KMeans'
        elif 'KMeans++:' in linea:
            algoritmo_actual = 'KMeans++'
        elif 'MeanShift:' in linea:
            algoritmo_actual = 'MeanShift'
        
        # Extraer métricas
        if 'Silhouette Score:' in linea and algoritmo_actual:
            silhouette = float(re.search(patron_silhouette, linea).group(1))
            
            # Buscar clusters y tiempo en líneas cercanas
            clusters = None
            tiempo = None
            for j in range(i, min(i+5, len(lineas))):
                if 'Clusters encontrados:' in lineas[j]:
                    clusters = int(re.search(patron_clusters, lineas[j]).group(1))
                if 'Tiempo:' in lineas[j]:
                    tiempo = float(re.search(patron_tiempo, lineas[j]).group(1))
            
            if clusters and tiempo:
                datos['silhouette_scores'].append(silhouette)
                datos['clusters'].append(clusters)
                datos['tiempos'].append(tiempo)
                datos['algoritmos'].append(algoritmo_actual)
                datos['configuraciones'].append(f"{algoritmo_actual}_{len([a for a in datos['algoritmos'] if a == algoritmo_actual])}")
        
        # Extraer top 3
        if 'TOP 3 CONFIGURACIONES' in linea:
            for j in range(i+1, min(i+10, len(lineas))):
                if 'Silhouette:' in lineas[j]:
                    match = re.search(r'Silhouette: ([-\d.]+)', lineas[j])
                    if match:
                        datos['top3'].append(float(match.group(1)))
        
        # Extraer accuracy en test
        if 'Accuracy en test:' in linea:
            match = re.search(r'Accuracy en test: ([\d.]+)', linea)
            if match:
                datos['test_accuracy'].append(float(match.group(1)))
    
    return datos


def extraer_datos_supervisado(log_path):
    """Extrae datos de supervisado desde el log"""
    datos = {
        'modelos': defaultdict(lambda: {'epocas': [], 'accuracy': [], 'tiempos': []}),
        'eliminados': [],
        'finales': {},
        'metricas_test': {}
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    lineas = contenido.split('\n')
    epoca_actual = 0
    
    for i, linea in enumerate(lineas):
        # Detectar época
        if match := re.search(r'ÉPOCA (\d+)/\d+', linea):
            epoca_actual = int(match.group(1))
        
        # Extraer accuracy por modelo
        if match := re.search(r'✓ (LR_config_\d+|SVM_config_\d+): ([\d.]+)', linea):
            modelo = match.group(1)
            accuracy = float(match.group(2))
            datos['modelos'][modelo]['epocas'].append(epoca_actual)
            datos['modelos'][modelo]['accuracy'].append(accuracy)
        
        # Modelos eliminados
        if '❌ Eliminado:' in linea:
            match = re.search(r'Eliminado: (.*?) \(accuracy: ([\d.]+)\)', linea)
            if match:
                datos['eliminados'].append({
                    'modelo': match.group(1),
                    'accuracy': float(match.group(2)),
                    'epoca': epoca_actual
                })
        
        # Métricas en test - buscar patrón "- Accuracy:"
        if '- Accuracy:' in linea:
            # Buscar el modelo en líneas anteriores
            for j in range(max(0, i-10), i):
                if 'LR_config' in lineas[j] or 'SVM_config' in lineas[j]:
                    modelo_match = re.search(r'(LR_config_\d+|SVM_config_\d+)', lineas[j])
                    if modelo_match:
                        modelo = modelo_match.group(1)
                        # Extraer métricas con el formato "- Metrica:  valor"
                        acc = re.search(r'-\s*Accuracy:\s+([\d.]+)', linea)
                        
                        # Buscar las siguientes métricas en las líneas siguientes
                        prec = None
                        rec = None
                        f1 = None
                        
                        for k in range(i+1, min(i+5, len(lineas))):
                            if '- Precision:' in lineas[k]:
                                prec = re.search(r'-\s*Precision:\s+([\d.]+)', lineas[k])
                            elif '- Recall:' in lineas[k]:
                                rec = re.search(r'-\s*Recall:\s+([\d.]+)', lineas[k])
                            elif '- F1-Score:' in lineas[k]:
                                f1 = re.search(r'-\s*F1-Score:\s+([\d.]+)', lineas[k])
                        
                        if acc and modelo not in datos['metricas_test']:
                            datos['metricas_test'][modelo] = {
                                'accuracy': float(acc.group(1)),
                                'precision': float(prec.group(1)) if prec else 0,
                                'recall': float(rec.group(1)) if rec else 0,
                                'f1': float(f1.group(1)) if f1 else 0
                            }
                        break
    
    return datos


def grafico_clustering_silhouette(datos, output_path):
    """Gráfico de Silhouette Scores por algoritmo"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Agrupar por algoritmo
    algoritmos_unicos = ['KMeans', 'KMeans++', 'MeanShift']
    colores = {'KMeans': '#3498db', 'KMeans++': '#e74c3c', 'MeanShift': '#2ecc71'}
    
    x_pos = []
    scores = []
    colors = []
    labels = []
    
    for alg in algoritmos_unicos:
        indices = [i for i, a in enumerate(datos['algoritmos']) if a == alg]
        for idx in indices:
            x_pos.append(len(x_pos))
            scores.append(datos['silhouette_scores'][idx])
            colors.append(colores[alg])
            labels.append(f"{alg}\nConfig {len([i for i in indices if i <= idx])}")
    
    bars = ax.bar(x_pos, scores, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Configuración', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Silhouette Scores - Clustering', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colores[alg], label=alg, alpha=0.7) for alg in algoritmos_unicos]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico guardado: {output_path}")


def grafico_clustering_clusters(datos, output_path):
    """Gráfico de número de clusters encontrados"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algoritmos_unicos = ['KMeans', 'KMeans++', 'MeanShift']
    colores = {'KMeans': '#3498db', 'KMeans++': '#e74c3c', 'MeanShift': '#2ecc71'}
    
    for alg in algoritmos_unicos:
        indices = [i for i, a in enumerate(datos['algoritmos']) if a == alg]
        configs = list(range(1, len(indices) + 1))
        clusters = [datos['clusters'][i] for i in indices]
        ax.plot(configs, clusters, marker='o', linewidth=2, markersize=8, 
                label=alg, color=colores[alg])
    
    ax.set_xlabel('Configuración', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Clusters', fontsize=12, fontweight='bold')
    ax.set_title('Número de Clusters Encontrados por Algoritmo', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico guardado: {output_path}")


def grafico_clustering_tiempos(datos, output_path):
    """Gráfico de tiempos de ejecución"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algoritmos_unicos = ['KMeans', 'KMeans++', 'MeanShift']
    colores = {'KMeans': '#3498db', 'KMeans++': '#e74c3c', 'MeanShift': '#2ecc71'}
    
    x_pos = []
    tiempos = []
    colors = []
    labels = []
    
    for alg in algoritmos_unicos:
        indices = [i for i, a in enumerate(datos['algoritmos']) if a == alg]
        for idx in indices:
            x_pos.append(len(x_pos))
            tiempos.append(datos['tiempos'][idx])
            colors.append(colores[alg])
            labels.append(f"{alg}\nC{len([i for i in indices if i <= idx])}")
    
    bars = ax.bar(x_pos, tiempos, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Configuración', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo (segundos)', fontsize=12, fontweight='bold')
    ax.set_title('Tiempo de Ejecución por Configuración - Clustering', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colores[alg], label=alg, alpha=0.7) for alg in algoritmos_unicos]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico guardado: {output_path}")


def grafico_supervisado_progreso(datos, output_path):
    """Gráfico de progreso de entrenamiento"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colores_lr = ['#3498db', '#2980b9', '#1f618d']
    colores_svm = ['#e74c3c', '#c0392b', '#a93226']
    
    for i, (modelo, info) in enumerate(sorted(datos['modelos'].items())):
        if not info['epocas']:
            continue
        
        if 'LR' in modelo:
            color = colores_lr[int(modelo.split('_')[-1]) - 1]
            linestyle = '-'
        else:
            color = colores_svm[int(modelo.split('_')[-1]) - 1]
            linestyle = '--'
        
        ax.plot(info['epocas'], info['accuracy'], 
                marker='o', linewidth=2, markersize=4,
                label=modelo, color=color, linestyle=linestyle, alpha=0.8)
    
    # Marcar eliminaciones
    for elim in datos['eliminados']:
        ax.axvline(x=elim['epoca'], color='red', linestyle=':', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (Train)', fontsize=12, fontweight='bold')
    ax.set_title('Progreso de Entrenamiento - Aprendizaje Supervisado', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico guardado: {output_path}")


def grafico_supervisado_metricas_test(datos, output_path):
    """Gráfico de métricas en test"""
    if not datos['metricas_test']:
        print("No hay métricas de test disponibles")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modelos = list(datos['metricas_test'].keys())
    metricas = ['accuracy', 'precision', 'recall', 'f1']
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(modelos))
    width = 0.2
    
    colores = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metrica, nombre) in enumerate(zip(metricas, metricas_nombres)):
        valores = [datos['metricas_test'][m][metrica] for m in modelos]
        ax.bar(x + i*width, valores, width, label=nombre, color=colores[i], alpha=0.8)
    
    ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Métricas en Test - Modelos Finales', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modelos, rotation=0)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico guardado: {output_path}")


def grafico_supervisado_comparacion_lr_svm(datos, output_path):
    """Comparación entre LR y SVM"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy final por tipo
    lr_models = {k: v for k, v in datos['modelos'].items() if 'LR' in k and v['accuracy']}
    svm_models = {k: v for k, v in datos['modelos'].items() if 'SVM' in k and v['accuracy']}
    
    # Gráfico 1: Accuracy final
    lr_final = [v['accuracy'][-1] for v in lr_models.values() if v['accuracy']]
    svm_final = [v['accuracy'][-1] for v in svm_models.values() if v['accuracy']]
    
    bp1 = ax1.boxplot([lr_final, svm_final], labels=['Regresión Logística', 'SVM'],
                       patch_artist=True, showmeans=True)
    bp1['boxes'][0].set_facecolor('#3498db')
    bp1['boxes'][1].set_facecolor('#e74c3c')
    
    ax1.set_ylabel('Accuracy Final (Train)', fontsize=11, fontweight='bold')
    ax1.set_title('Distribución de Accuracy por Técnica', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Épocas sobrevividas
    lr_epocas = [len(v['epocas']) for v in lr_models.values() if v['epocas']]
    svm_epocas = [len(v['epocas']) for v in svm_models.values() if v['epocas']]
    
    x = np.arange(2)
    width = 0.35
    
    ax2.bar(x[0], np.mean(lr_epocas) if lr_epocas else 0, width, 
            label='LR', color='#3498db', alpha=0.8)
    ax2.bar(x[1], np.mean(svm_epocas) if svm_epocas else 0, width,
            label='SVM', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Épocas Promedio', fontsize=11, fontweight='bold')
    ax2.set_title('Épocas Sobrevividas por Técnica', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Regresión Logística', 'SVM'])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico guardado: {output_path}")


def grafico_comparacion_final_detallado(output_path):
    """Gráfico de comparación final detallado para Diapositiva 10"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Accuracy
    enfoques = ['Clustering', 'Supervisado']
    accuracies = [15.43, 94.67]
    colores = ['#e74c3c', '#2ecc71']
    
    bars1 = ax1.bar(enfoques, accuracies, color=colores, alpha=0.8, 
                    edgecolor='black', linewidth=2, width=0.6)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy en Test', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Gráfico 2: Ventajas/Desventajas
    categorias = ['Requiere\nEtiquetas', 'Precisión', 'Exploración', 'Costo']
    clustering_scores = [5, 2, 5, 5]  # Escala 1-5 (5=mejor para clustering)
    supervisado_scores = [1, 5, 3, 2]  # Escala 1-5 (5=mejor para supervisado)
    
    x = np.arange(len(categorias))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, clustering_scores, width, 
                    label='Clustering', color='#e74c3c', alpha=0.7)
    bars3 = ax2.bar(x + width/2, supervisado_scores, width,
                    label='Supervisado', color='#2ecc71', alpha=0.7)
    
    ax2.set_ylabel('Score (1-5)', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de Características', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categorias, fontsize=10)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 6])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Gráfico guardado: {output_path}")


def main():
    """Función principal"""
    print("="*70)
    print("GENERADOR DE GRÁFICOS - TAREA 2")
    print("="*70)
    
    # Buscar logs más recientes
    logs_dir = 'logs'
    
    clustering_logs = sorted([f for f in os.listdir(logs_dir) if f.startswith('clustering_')],
                            reverse=True)
    supervisado_logs = sorted([f for f in os.listdir(logs_dir) if f.startswith('supervisado_')],
                             reverse=True)
    
    if not clustering_logs:
        print("No se encontraron logs de clustering")
    else:
        print(f"\nCLUSTERING - Procesando: {clustering_logs[0]}")
        clustering_path = os.path.join(logs_dir, clustering_logs[0])
        datos_clustering = extraer_datos_clustering(clustering_path)
        
        if datos_clustering['silhouette_scores']:
            print(f"   Configuraciones encontradas: {len(datos_clustering['silhouette_scores'])}")
            grafico_clustering_silhouette(datos_clustering, 'graficos/clustering_silhouette.png')
            grafico_clustering_clusters(datos_clustering, 'graficos/clustering_num_clusters.png')
            grafico_clustering_tiempos(datos_clustering, 'graficos/clustering_tiempos.png')
        else:
            print("   No se pudieron extraer datos del log")
    
    if not supervisado_logs:
        print("\nNo se encontraron logs de supervisado")
    else:
        # Buscar el log más reciente con contenido
        supervisado_path = None
        for log in supervisado_logs:
            path = os.path.join(logs_dir, log)
            if os.path.getsize(path) > 1000:  # Al menos 1KB
                supervisado_path = path
                print(f"\nSUPERVISADO - Procesando: {log}")
                break
        
        if not supervisado_path:
            print("\nNo se encontraron logs de supervisado con contenido")
            return
        
        datos_supervisado = extraer_datos_supervisado(supervisado_path)
        
        if datos_supervisado['modelos']:
            print(f"   Modelos encontrados: {len(datos_supervisado['modelos'])}")
            grafico_supervisado_progreso(datos_supervisado, 'graficos/supervisado_progreso.png')
            grafico_supervisado_metricas_test(datos_supervisado, 'graficos/supervisado_metricas_test.png')
            grafico_supervisado_comparacion_lr_svm(datos_supervisado, 'graficos/supervisado_comparacion.png')
        else:
            print("   No se pudieron extraer datos del log")
    
    # Generar gráfico de comparación final
    print("\nCOMPARACIÓN FINAL - Generando gráfico detallado...")
    grafico_comparacion_final_detallado('graficos/comparacion_final_detallado.png')
    
    print("\n" + "="*70)
    print("GENERACIÓN DE GRÁFICOS COMPLETADA")
    print(f"Gráficos guardados en: ./graficos/")
    print("="*70)
    print("\nGráficos generados:")
    print("  Clustering (3): silhouette, num_clusters, tiempos")
    print("  Supervisado (3): progreso, metricas_test, comparacion")
    print("  Comparación Final (1): comparacion_final_detallado")


if __name__ == "__main__":
    main()
