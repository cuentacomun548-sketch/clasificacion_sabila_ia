"""Informe de clasificación de sábila. Solo lee resultados guardados.
Uso: python report.py --config configs/fit.yaml
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
NOMBRES = {'dinov3_vit_s_16': 'DINOv3', 'mobilenetv4_conv_small': 'MobileNetV4', 'yolo26n_cls': 'YOLO26n-cls'}
ESTRATEGIAS = ['desde_cero', 'transferencia']
SEMILLAS = [7, 42]
CLASES = {'Fresh Aloe Vera': 'Sana', 'Rot Aloe Vera': 'Podredumbre', 'Rust Aloe Vera': 'Roya'}


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def estilo():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def leer_validacion(entrenamientos, salida):
    filas = []
    for modelo in MODELOS:
        for estrategia in ESTRATEGIAS:
            for semilla in SEMILLAS:
                ruta = entrenamientos / modelo / estrategia / f'semilla_{semilla}' / 'resultado.json'
                if not ruta.is_file():
                    raise FileNotFoundError(f'Falta resultado de entrenamiento: {ruta}')
                with ruta.open(encoding='utf-8') as f:
                    registro = json.load(f)
                filas.append({'modelo': modelo, 'estrategia': estrategia, 'semilla': semilla,
                              'f1_validacion': float(registro['mejor_f1_validacion'])})
    detalle = pd.DataFrame(filas)
    detalle.to_csv(salida / 'validacion_por_ejecucion.csv', index=False)
    resumen = detalle.groupby(['modelo', 'estrategia'], sort=False).agg(
        f1_promedio=('f1_validacion', 'mean'),
        f1_desviacion=('f1_validacion', 'std')).reset_index()
    resumen.to_csv(salida / 'resumen_validacion.csv', index=False)
    return resumen


def imprimir_tabla(tabla, titulo):
    vista = tabla[['modelo', 'estrategia', 'f1_promedio', 'f1_desviacion']].copy()
    vista['modelo'] = vista['modelo'].replace(NOMBRES)
    vista['estrategia'] = vista['estrategia'].replace({'desde_cero': 'Desde cero', 'transferencia': 'Transferencia'})
    vista.columns = ['Arquitectura', 'Estrategia', 'F1 macro medio', 'Desv. estándar']
    print('\n' + titulo + ' | semillas 7 y 42')
    print(vista.round(4).to_string(index=False))


def comparativa(val, prueba, salida):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True, layout='constrained')
    for ax, tabla, titulo in zip(axes, [val, prueba], ['Validación (347 imágenes)', 'Prueba (349 imágenes)']):
        tabla = tabla.set_index(['modelo', 'estrategia']).reindex(
            pd.MultiIndex.from_product([MODELOS, ESTRATEGIAS])).reset_index()
        valores = tabla['f1_promedio'].astype(float).to_numpy()
        desv = tabla['f1_desviacion'].fillna(0).astype(float).to_numpy()
        x = np.arange(6)
        colores = ['#7898ad' if i % 2 == 0 else '#326e83' for i in x]
        ax.bar(x, valores, color=colores, width=.66, yerr=desv, capsize=3, ecolor='#333333')
        ax.set_xticks(x, [f'{NOMBRES[MODELOS[i // 2]]}\n{"Cero" if i % 2 == 0 else "Transfer."}' for i in x], fontsize=8)
        ax.set_ylim(0, 1.10)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.set_title(titulo, fontsize=11)
        ax.grid(axis='y', alpha=.17)
        ax.set_axisbelow(True)
        for xi, y, sd in zip(x, valores, desv):
            ax.text(xi, min(y + sd + .018, 1.065), f'{y:.4f}', ha='center', fontsize=7.5)
    axes[0].set_ylabel('F1 macro')
    fig.suptitle('Desempeño medio ± desviación estándar | semillas 7 y 42', fontsize=12)
    guardar(fig, salida / 'comparacion_f1.png')


def leer_tensorboard(carpeta):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print('AVISO: tensorboard no está instalado; se omiten curvas de DINOv3/MobileNetV4.')
        return {}
    eventos = sorted(carpeta.rglob('events.out.tfevents.*')) if carpeta.exists() else []
    por_tag = {}
    for directorio in sorted({p.parent for p in eventos}):
        try:
            acc = EventAccumulator(str(directorio), size_guidance={'scalars': 0})
            acc.Reload()
            for tag in ['train/loss', 'val/loss', 'val/f1_macro']:
                if tag in acc.Tags().get('scalars', []) and tag not in por_tag:
                    puntos = acc.Scalars(tag)
                    if puntos:
                        por_tag[tag] = ([p.step for p in puntos], [p.value for p in puntos])
        except Exception as exc:
            print(f'AVISO: no se pudo leer {directorio}: {exc}')
    return por_tag


def leer_yolo(carpeta):
    archivos = sorted(carpeta.rglob('results.csv')) if carpeta.exists() else []
    if not archivos:
        return {}
    df = pd.read_csv(archivos[0])
    df.columns = df.columns.str.strip()
    x = df['epoch'].to_numpy() if 'epoch' in df else np.arange(1, len(df) + 1)
    return {col: (x, pd.to_numeric(df[col], errors='coerce').to_numpy())
            for col in ['train/loss', 'val/loss', 'metrics/accuracy_top1'] if col in df}


def curvas(entrenamientos, salida):
    # Una figura por arquitectura: 2 estrategias x 2 semillas; cada panel separa pérdida y métrica.
    for modelo in MODELOS:
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), layout='constrained')
        encontradas = 0
        for fila, estrategia in enumerate(ESTRATEGIAS):
            for col, semilla in enumerate(SEMILLAS):
                ax = axes[fila, col]
                carpeta = entrenamientos / modelo / estrategia / f'semilla_{semilla}'
                if modelo == 'yolo26n_cls':
                    datos = leer_yolo(carpeta)
                    metrica = 'metrics/accuracy_top1'
                    etiqueta_metrica = 'Exactitud top-1 (validación)'
                    eje_x = 'Época'
                else:
                    datos = leer_tensorboard(carpeta / 'tensorboard')
                    metrica = 'val/f1_macro'
                    etiqueta_metrica = 'F1 macro (validación)'
                    eje_x = 'Paso de entrenamiento'
                if not datos:
                    ax.text(.5, .5, 'Registro no disponible', transform=ax.transAxes,
                            ha='center', va='center', color='#666666')
                else:
                    encontradas += 1
                    for tag, label, color in [('train/loss', 'Entrenamiento', '#326e83'),
                                               ('val/loss', 'Validación', '#c37d43')]:
                        if tag in datos:
                            ax.plot(*datos[tag], label=f'Pérdida: {label}', color=color, lw=1.8)
                    if metrica in datos:
                        ax2 = ax.twinx()
                        ax2.plot(*datos[metrica], label=etiqueta_metrica, color='#66936d', lw=1.6, linestyle='--')
                        ax2.set_ylim(0, 1.05)
                        ax2.set_ylabel(etiqueta_metrica, fontsize=8, color='#426d49')
                        ax2.tick_params(axis='y', labelsize=7)
                        ax2.spines['top'].set_visible(False)
                ax.set_title(f'{"Desde cero" if estrategia == "desde_cero" else "Transferencia"} | semilla {semilla}', fontsize=10)
                ax.set_xlabel(eje_x, fontsize=8)
                ax.set_ylabel('Pérdida', fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=.15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8,
                       bbox_to_anchor=(.5, -.015))
        fig.suptitle(f'{NOMBRES[modelo]} | evolución del entrenamiento', fontsize=12)
        if encontradas:
            guardar(fig, salida / f'curvas_{modelo}.png')
        else:
            plt.close(fig)
            print(f'AVISO: sin curvas para {modelo}')


def matrices(ruta_test, salida):
    for semilla in SEMILLAS:
        fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.3), layout='constrained')
        # Escala común en los seis paneles, para comparar los conteos.
        matrices_leidas = {}
        maximo = 0
        for modelo in MODELOS:
            for estrategia in ESTRATEGIAS:
                ruta = ruta_test / f'matriz_{modelo}_{estrategia}_semilla_{semilla}.csv'
                if not ruta.is_file():
                    raise FileNotFoundError(f'Falta matriz de confusión: {ruta}')
                tabla = pd.read_csv(ruta, index_col=0)
                matrices_leidas[(modelo, estrategia)] = tabla
                maximo = max(maximo, int(tabla.to_numpy().max()))
        for i, modelo in enumerate(MODELOS):
            for j, estrategia in enumerate(ESTRATEGIAS):
                ax = axes[i, j]
                tabla = matrices_leidas[(modelo, estrategia)]
                valores = tabla.to_numpy()
                ax.imshow(valores, cmap='Blues', vmin=0, vmax=maximo)
                etiquetas_x = [CLASES.get(str(s).removeprefix('pred_'), str(s).removeprefix('pred_')) for s in tabla.columns]
                etiquetas_y = [CLASES.get(str(s).removeprefix('real_'), str(s).removeprefix('real_')) for s in tabla.index]
                ax.set_xticks(range(len(etiquetas_x)), etiquetas_x, fontsize=8)
                ax.set_yticks(range(len(etiquetas_y)), etiquetas_y, fontsize=8)
                ax.set_title(f'{NOMBRES[modelo]} | {"Desde cero" if estrategia == "desde_cero" else "Transferencia"}', fontsize=10)
                ax.set_xlabel('Predicción', fontsize=8)
                ax.set_ylabel('Clase real', fontsize=8)
                for (r, c), n in np.ndenumerate(valores):
                    ax.text(c, r, str(int(n)), ha='center', va='center', fontsize=11,
                            color='white' if n > maximo * .52 else 'black')
        fig.suptitle(f'Matrices de confusión | prueba (349 imágenes) | semilla {semilla}', fontsize=12)
        guardar(fig, salida / f'matrices_confusion_semilla_{semilla}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fit.yaml')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    base = Path(config['resultados']['ruta']) / 'experimento_una_fase_2026'
    entrenamientos, ruta_test, salida = base / 'entrenamientos', base / 'evaluacion_test', base / 'informe'
    salida.mkdir(parents=True, exist_ok=True)
    estilo()
    resumen_val = leer_validacion(entrenamientos, salida)
    ruta = ruta_test / 'resumen_prueba.csv'
    if not ruta.is_file():
        raise FileNotFoundError(f'Falta evaluación final: {ruta}')
    resumen_test = pd.read_csv(ruta)
    esperadas = {(m, e) for m in MODELOS for e in ESTRATEGIAS}
    if set(zip(resumen_test['modelo'], resumen_test['estrategia'])) != esperadas:
        raise ValueError('El resumen de prueba no contiene las seis combinaciones esperadas')
    resumen_test.to_csv(salida / 'resumen_prueba.csv', index=False)
    imprimir_tabla(resumen_val, 'VALIDACIÓN: F1 macro (347 imágenes)')
    imprimir_tabla(resumen_test, 'PRUEBA: F1 macro (349 imágenes)')
    comparativa(resumen_val, resumen_test, salida)
    curvas(entrenamientos, salida)
    matrices(ruta_test, salida)
    print(f'\nInforme generado en: {salida}')
    print('Figuras: comparacion_f1.png, curvas_<arquitectura>.png y matrices_confusion_semilla_<7|42>.png')
    print('Nota: las curvas representan registros históricos; la evaluación usa el mejor checkpoint guardado.')


if __name__ == '__main__':
    main()
