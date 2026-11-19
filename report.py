"""Informe reproducible del experimento de clasificación de sábila.

Uso: python report.py --config configs/fit.yaml
No entrena modelos ni realiza inferencia: solo lee resultados guardados.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

MODELOS = ['dinov3_vit_s_16', 'mobilenetv4_conv_small', 'yolo26n_cls']
ESTRATEGIAS = ['desde_cero', 'transferencia']
SEMILLAS = [7, 42]


def guardar_figura(fig, ruta):
    fig.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close(fig)


def validacion(entrenamientos, salida):
    filas = []
    for modelo in MODELOS:
        for estrategia in ESTRATEGIAS:
            for semilla in SEMILLAS:
                ruta = entrenamientos / modelo / estrategia / f'semilla_{semilla}' / 'resultado.json'
                if not ruta.is_file():
                    raise FileNotFoundError(f'Falta resultado de entrenamiento: {ruta}')
                with ruta.open(encoding='utf-8') as archivo:
                    registro = json.load(archivo)
                filas.append({'modelo': modelo, 'estrategia': estrategia,
                              'semilla': semilla, 'f1_validacion': float(registro['mejor_f1_validacion'])})
    detalle = pd.DataFrame(filas)
    detalle.to_csv(salida / 'validacion_por_ejecucion.csv', index=False)
    resumen = detalle.groupby(['modelo', 'estrategia'], sort=False).agg(
        f1_promedio=('f1_validacion', 'mean'),
        f1_desviacion=('f1_validacion', 'std')).reset_index()
    resumen.to_csv(salida / 'resumen_validacion.csv', index=False)
    print('\nVALIDACIÓN: F1 MACRO (PROMEDIO Y DESVIACIÓN)')
    print(resumen.round(4).to_string(index=False))
    return resumen


def comparativa(resumen_val, resumen_test, salida):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for eje, tabla, titulo in [(ejes[0], resumen_val, 'Validación'),
                               (ejes[1], resumen_test, 'Prueba')]:
        etiquetas = [f'{m.replace("_", " ")}\n{e.replace("_", " ")}'
                     for m, e in zip(tabla['modelo'], tabla['estrategia'])]
        x = np.arange(len(tabla))
        eje.bar(x, tabla['f1_promedio'], yerr=tabla['f1_desviacion'], capsize=3)
        eje.set_xticks(x, etiquetas, rotation=55, ha='right', fontsize=8)
        eje.set_ylim(0, 1.08)
        eje.set_title(f'F1 macro: {titulo}')
        eje.grid(axis='y', alpha=0.25)
    ejes[0].set_ylabel('F1 macro')
    fig.suptitle('Desempeño por arquitectura y estrategia (2 semillas)')
    guardar_figura(fig, salida / 'comparacion_f1.png')


def curvas(entrenamientos, salida):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print('Curvas omitidas: tensorboard no está instalado.')
        return
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    metricas = [('train/loss', 'Pérdida de entrenamiento'),
                ('val/loss', 'Pérdida de validación'),
                ('val/f1_macro', 'F1 macro de validación')]
    halladas = 0
    for modelo in MODELOS[:2]:
        for estrategia in ESTRATEGIAS:
            carpeta = entrenamientos / modelo / estrategia / 'semilla_42' / 'tensorboard'
            if not carpeta.is_dir():
                continue
            eventos = list(carpeta.rglob('events.out.tfevents.*'))
            if not eventos:
                continue
            # TensorBoardLogger escribe en .../tensorboard/entrenamiento.
            directorios = sorted({p.parent for p in eventos})
            for directorio in directorios:
                try:
                    acc = EventAccumulator(str(directorio), size_guidance={'scalars': 0})
                    acc.Reload()
                    tags = acc.Tags().get('scalars', [])
                    if not any(t in tags for t, _ in metricas):
                        continue
                    fila = MODELOS.index(modelo)
                    col = ESTRATEGIAS.index(estrategia)
                    eje = ejes[fila, col]
                    for tag, nombre in metricas:
                        if tag in tags:
                            puntos = acc.Scalars(tag)
                            eje.plot([p.step for p in puntos], [p.value for p in puntos], label=nombre)
                    eje.set_title(f'{modelo} | {estrategia} | semilla 42')
                    eje.set_xlabel('Paso de entrenamiento')
                    eje.grid(alpha=0.2)
                    eje.legend(fontsize=7)
                    halladas += 1
                    break
                except Exception as exc:
                    print(f'No se pudo leer {directorio}: {exc}')
    if halladas:
        fig.suptitle('Curvas registradas en TensorBoard (semilla 42)')
        guardar_figura(fig, salida / 'curvas.png')
    else:
        plt.close(fig)
        print('No se encontraron curvas legibles en los registros de TensorBoard.')


def matrices(ruta_test, salida):
    # Una matriz por combinación, semilla 42, para no saturar la presentación.
    fig, ejes = plt.subplots(3, 2, figsize=(12, 15))
    for i, modelo in enumerate(MODELOS):
        for j, estrategia in enumerate(ESTRATEGIAS):
            archivo = ruta_test / f'matriz_{modelo}_{estrategia}_semilla_42.csv'
            if not archivo.is_file():
                raise FileNotFoundError(f'Falta matriz de confusión: {archivo}')
            matriz = pd.read_csv(archivo, index_col=0)
            eje = ejes[i, j]
            imagen = eje.imshow(matriz.to_numpy(), cmap='Blues', vmin=0)
            eje.set_xticks(range(len(matriz.columns)), [s.removeprefix('pred_') for s in matriz.columns],
                           rotation=30, ha='right', fontsize=8)
            eje.set_yticks(range(len(matriz.index)), [s.removeprefix('real_') for s in matriz.index], fontsize=8)
            eje.set_title(f'{modelo} | {estrategia} | semilla 42')
            eje.set_xlabel('Predicción')
            eje.set_ylabel('Clase real')
            for (y, x), numero in np.ndenumerate(matriz.to_numpy()):
                eje.text(x, y, str(int(numero)), ha='center', va='center', fontsize=10,
                         color='white' if numero > matriz.to_numpy().max() / 2 else 'black')
    fig.suptitle('Matrices de confusión en prueba (semilla 42)')
    guardar_figura(fig, salida / 'matrices_confusion.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fit.yaml')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as archivo:
        config = yaml.safe_load(archivo)
    base = Path(config['resultados']['ruta']) / 'experimento_una_fase_2026'
    entrenamientos = base / 'entrenamientos'
    ruta_test = base / 'evaluacion_test'
    salida = base / 'informe'
    salida.mkdir(parents=True, exist_ok=True)
    resumen_val = validacion(entrenamientos, salida)
    archivo_test = ruta_test / 'resumen_prueba.csv'
    if not archivo_test.is_file():
        raise FileNotFoundError(f'Falta evaluación final: {archivo_test}')
    resumen_test = pd.read_csv(archivo_test)
    esperadas = {(m, e) for m in MODELOS for e in ESTRATEGIAS}
    if set(zip(resumen_test['modelo'], resumen_test['estrategia'])) != esperadas:
        raise ValueError('El resumen de prueba no contiene las seis combinaciones esperadas')
    resumen_test.to_csv(salida / 'resumen_prueba.csv', index=False)
    print('\nPRUEBA: F1 MACRO (PROMEDIO Y DESVIACIÓN)')
    print(resumen_test[['modelo', 'estrategia', 'f1_promedio', 'f1_desviacion']].round(4).to_string(index=False))
    comparativa(resumen_val, resumen_test, salida)
    curvas(entrenamientos, salida)
    matrices(ruta_test, salida)
    print(f'\nInforme generado en: {salida}')
    print('Figuras: comparacion_f1.png, matrices_confusion.png y curvas.png (si hay registros).')


if __name__ == '__main__':
    main()
