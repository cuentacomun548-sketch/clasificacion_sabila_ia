"""Evaluación final de 12 clasificadores de sábila, sin reentrenamiento.

Ejecutar desde la raíz del repositorio:
    python evaluate_test.py --config configs/fit.yaml

La evaluación se realiza una sola vez sobre el subconjunto test del manifiesto.
Guarda métricas, matrices de confusión y predicciones en una carpeta independiente.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from dataset import crear_dataloaders
from model import crear_modelo

MODELOS = ("dinov3_vit_s_16", "mobilenetv4_conv_small", "yolo26n_cls")
ESTRATEGIAS = ("desde_cero", "transferencia")
SEMILLAS = (42, 7)


def cargar_yaml(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def evaluar_timm(registro, config, config_modelo, dispositivo):
    # La arquitectura se instancia sin descargar pesos: se recuperan del checkpoint.
    red, imagen = crear_modelo(config_modelo, "desde_cero")
    ckpt = torch.load(registro["checkpoint"], map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError("El checkpoint Lightning no contiene state_dict")
    pesos = {k[len("red."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("red.")}
    red.load_state_dict(pesos, strict=True)
    red.to(dispositivo).eval()

    alto, ancho = imagen["input_size"][-2:]
    if alto != ancho:
        raise ValueError("Se esperaba una entrada cuadrada")
    cargadores = crear_dataloaders(
        config, tamano=alto,
        media=imagen.get("mean", (0.485, 0.456, 0.406)),
        desviacion=imagen.get("std", (0.229, 0.224, 0.225)),
        interpolacion=imagen.get("interpolation", "bicubic"),
        recorte=imagen.get("crop_pct", 1.0),
    )
    reales, predichas = [], []
    with torch.inference_mode():
        for imagenes, etiquetas in cargadores["test"]:
            salida = red(imagenes.to(dispositivo))
            predichas.extend(salida.argmax(dim=1).cpu().tolist())
            reales.extend(etiquetas.tolist())
    del red
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()
    return reales, predichas


def evaluar_yolo(registro, filas_test, nombres_por_etiqueta, dispositivo):
    from ultralytics import YOLO
    modelo = YOLO(registro["checkpoint"])
    # Ultralytics usa índices de clase propios. Se traducen mediante sus nombres.
    nombres_yolo = modelo.names
    if isinstance(nombres_yolo, list):
        nombres_yolo = dict(enumerate(nombres_yolo))
    nombre_a_etiqueta = {str(nombre).strip().casefold(): int(etiqueta)
                        for etiqueta, nombre in nombres_por_etiqueta.items()}
    traduccion = {}
    for indice, nombre in nombres_yolo.items():
        clave = str(nombre).strip().casefold()
        if clave not in nombre_a_etiqueta:
            raise ValueError(f"Clase YOLO inesperada: {nombre!r}. Esperadas: {nombres_por_etiqueta}")
        traduccion[int(indice)] = nombre_a_etiqueta[clave]
    if len(traduccion) != len(nombres_por_etiqueta):
        raise ValueError("No coinciden las clases YOLO y el manifiesto")

    rutas = filas_test["ruta"].astype(str).tolist()
    predichas = []
    # Lotes de imágenes PIL: formato de entrada compatible con Ultralytics.
    for inicio in range(0, len(rutas), 16):
        lote = rutas[inicio:inicio + 16]
        imagenes = []
        try:
            for ruta in lote:
                with Image.open(ruta) as imagen:
                    imagenes.append(imagen.convert("RGB"))
            resultados = modelo.predict(
                source=imagenes, imgsz=224, batch=16,
                device=0 if dispositivo.type == "cuda" else "cpu",
                verbose=False,
            )
            for resultado in resultados:
                if resultado.probs is None:
                    raise ValueError("El checkpoint YOLO no es un clasificador")
                predichas.append(traduccion[int(resultado.probs.top1)])
        finally:
            for imagen in imagenes:
                imagen.close()
    if len(predichas) != len(rutas):
        raise ValueError(f"YOLO predijo {len(predichas)} de {len(rutas)} imágenes")
    return filas_test["etiqueta"].astype(int).tolist(), predichas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fit.yaml")
    parser.add_argument("--salida", default=None)
    args = parser.parse_args()

    config = cargar_yaml(args.config)
    base = Path(config["resultados"]["ruta"])
    entrenamientos = base / "experimento_una_fase_2026" / "entrenamientos"
    salida = Path(args.salida) if args.salida else base / "experimento_una_fase_2026" / "evaluacion_test"
    manifiesto = base / "datos" / "particiones.csv"
    if not manifiesto.is_file():
        raise FileNotFoundError(manifiesto)
    tabla = pd.read_csv(manifiesto)
    columnas = {"ruta", "etiqueta", "clase", "subconjunto"}
    if not columnas.issubset(tabla.columns):
        raise ValueError(f"Faltan columnas: {columnas - set(tabla.columns)}")
    filas_test = tabla.loc[tabla["subconjunto"] == "test"].reset_index(drop=True)
    if len(filas_test) != 349:
        raise ValueError(f"Se esperaban 349 imágenes de prueba; hay {len(filas_test)}")
    if not filas_test["ruta"].map(lambda x: Path(x).is_file()).all():
        raise FileNotFoundError("Faltan imágenes de prueba en las rutas del manifiesto")
    pares = tabla[["etiqueta", "clase"]].drop_duplicates()
    if pares["etiqueta"].duplicated().any() or pares["clase"].duplicated().any():
        raise ValueError("El manifiesto tiene una correspondencia inconsistente de etiquetas y clases")
    nombres = dict(zip(pares["etiqueta"].astype(int), pares["clase"].astype(str)))
    etiquetas = sorted(nombres)
    if etiquetas != list(range(len(etiquetas))):
        raise ValueError(f"Etiquetas inesperadas: {etiquetas}")

    registros = {}
    for ruta in sorted(entrenamientos.rglob("resultado.json")):
        with open(ruta, encoding="utf-8") as archivo:
            registro = json.load(archivo)
        clave = (registro["modelo"], registro["estrategia"], int(registro["semilla"]))
        if clave in registros:
            raise ValueError(f"Resultado duplicado: {clave}")
        registros[clave] = registro
    esperados = {(m, e, s) for m in MODELOS for e in ESTRATEGIAS for s in SEMILLAS}
    if set(registros) != esperados:
        raise ValueError(f"Faltan registros: {esperados - set(registros)}; sobran: {set(registros) - esperados}")
    for clave, registro in registros.items():
        if not Path(registro["checkpoint"]).is_file():
            raise FileNotFoundError(f"Checkpoint no encontrado {clave}: {registro['checkpoint']}")

    salida.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluación de {len(registros)} modelos sobre {len(filas_test)} imágenes | {dispositivo}", flush=True)
    resumen = []
    for clave in sorted(esperados):
        modelo, estrategia, semilla = clave
        registro = registros[clave]
        print(f"Evaluando {modelo} | {estrategia} | semilla {semilla}", flush=True)
        if modelo == "yolo26n_cls":
            reales, predichas = evaluar_yolo(registro, filas_test, nombres, dispositivo)
        else:
            config_modelo = cargar_yaml(Path("configs") / f"{modelo}.yaml")
            reales, predichas = evaluar_timm(registro, config, config_modelo, dispositivo)
        if len(reales) != len(filas_test):
            raise ValueError("Número de predicciones incorrecto")
        matriz = confusion_matrix(reales, predichas, labels=etiquetas)
        prefijo = f"{modelo}_{estrategia}_semilla_{semilla}"
        pd.DataFrame(matriz, index=[f"real_{nombres[i]}" for i in etiquetas],
                     columns=[f"pred_{nombres[i]}" for i in etiquetas]).to_csv(salida / f"matriz_{prefijo}.csv")
        pd.DataFrame({"ruta": filas_test["ruta"], "real": reales,
                      "predicha": predichas}).to_csv(salida / f"predicciones_{prefijo}.csv", index=False)
        fila = {
            "modelo": modelo, "estrategia": estrategia, "semilla": semilla,
            "n_prueba": len(reales),
            "accuracy": accuracy_score(reales, predichas),
            "balanced_accuracy": balanced_accuracy_score(reales, predichas),
            "precision_macro": precision_score(reales, predichas, average="macro", zero_division=0),
            "recall_macro": recall_score(reales, predichas, average="macro", zero_division=0),
            "f1_macro": f1_score(reales, predichas, average="macro", zero_division=0),
        }
        resumen.append(fila)
        pd.DataFrame(resumen).to_csv(salida / "metricas_por_ejecucion.csv", index=False)
        print(f"  F1 macro de prueba: {fila['f1_macro']:.4f}", flush=True)
    detalle = pd.DataFrame(resumen)
    agrupado = detalle.groupby(["modelo", "estrategia"], as_index=False).agg(
        f1_promedio=("f1_macro", "mean"), f1_desviacion=("f1_macro", "std"),
        accuracy_promedio=("accuracy", "mean"),
        balanced_accuracy_promedio=("balanced_accuracy", "mean"),
        precision_promedio=("precision_macro", "mean"),
        recall_promedio=("recall_macro", "mean"),
    )
    agrupado.to_csv(salida / "resumen_prueba.csv", index=False)
    print("\nRESUMEN FINAL\n", agrupado.round(4).to_string(index=False), flush=True)
    print(f"\nResultados guardados en: {salida}", flush=True)


if __name__ == "__main__":
    main()
