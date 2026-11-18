"""Entrenamiento reproducible de clasificadores de Aloe vera."""
import argparse
import json
import shutil
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassF1Score

from dataset import crear_dataloaders
from model import crear_modelo


class Clasificador(L.LightningModule):
    def __init__(self, red, n_clases, lr, weight_decay):
        super().__init__()
        self.red = red
        self.lr = lr
        self.weight_decay = weight_decay
        self.f1_val = MulticlassF1Score(num_classes=n_clases, average="macro")

    def forward(self, x):
        return self.red(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        perdida = F.cross_entropy(self(x), y)
        self.log("train/loss", perdida, on_step=False, on_epoch=True, prog_bar=True)
        return perdida

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predicciones = self(x)
        self.f1_val.update(predicciones, y)
        self.log("val/loss", F.cross_entropy(predicciones, y), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val/f1_macro", self.f1_val.compute(), prog_bar=True)
        self.f1_val.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def leer_yaml(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def preparar_datos(config, imagen):
    alto, ancho = imagen["input_size"][-2:]
    if alto != ancho:
        raise ValueError("El modelo requiere imágenes cuadradas.")
    return crear_dataloaders(
        config,
        tamano=alto,
        media=imagen.get("mean", (0.485, 0.456, 0.406)),
        desviacion=imagen.get("std", (0.229, 0.224, 0.225)),
        interpolacion=imagen.get("interpolation", "bicubic"),
        recorte=imagen.get("crop_pct", 1.0),
        biblioteca="timm",
    )


def guardar_resultado(ruta, nombre, estrategia, semilla, f1, checkpoint):
    ruta.mkdir(parents=True, exist_ok=True)
    archivo = ruta / "resultado.json"
    with open(archivo, "w", encoding="utf-8") as salida:
        json.dump(
            {
                "modelo": nombre,
                "estrategia": estrategia,
                "semilla": semilla,
                "protocolo": "una_fase_15ep_2026",
                "mejor_f1_validacion": float(f1),
                "checkpoint": str(checkpoint),
            },
            salida,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Terminado. Resultado: {archivo}")
    print("El conjunto test permanece sin utilizar.")


def entrenar_timm(config, modelo_config, estrategia, semilla, ruta):
    red, imagen = crear_modelo(modelo_config, estrategia)
    datos = preparar_datos(config, imagen)
    modulo = Clasificador(
        red,
        config["model"]["num_classes"],
        config["model"]["lr_desde_cero" if estrategia == "desde_cero" else "lr_transferencia"],
        config["model"]["weight_decay"],
    )
    parada = config["trainer"]["early_stopping"]
    checkpoint = ModelCheckpoint(
        dirpath=ruta / "checkpoints",
        filename="mejor-{epoch:02d}",
        monitor=parada["monitor"],
        mode=parada["mode"],
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    entrenador = L.Trainer(
        max_epochs=config["trainer"][estrategia]["max_epochs"],
        accelerator="auto",
        devices=1,
        precision="32-true",
        logger=TensorBoardLogger(save_dir=str(ruta), name="tensorboard", version="entrenamiento"),
        callbacks=[
            checkpoint,
            EarlyStopping(monitor=parada["monitor"], mode=parada["mode"], patience=parada["patience"]),
        ],
        default_root_dir=str(ruta),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    entrenador.fit(modulo, train_dataloaders=datos["train"], val_dataloaders=datos["val"])
    if not checkpoint.best_model_path:
        raise RuntimeError("No se guardó ningún checkpoint.")
    guardar_resultado(ruta, modelo_config["nombre"], estrategia, semilla,
                      checkpoint.best_model_score.item(), checkpoint.best_model_path)


def preparar_directorios_yolo(config, ruta):
    """Crea estructura ImageFolder de Ultralytics usando el manifiesto fijo."""
    import pandas as pd

    manifiesto = Path(config["resultados"]["ruta"]) / "datos" / "particiones.csv"
    if not manifiesto.is_file():
        raise FileNotFoundError(f"No existe el manifiesto: {manifiesto}")
    tabla = pd.read_csv(manifiesto)
    requeridas = {"ruta", "etiqueta", "subconjunto"}
    if not requeridas.issubset(tabla.columns):
        raise ValueError(f"El manifiesto debe contener {requeridas}")

    clases = config["data"]["clases"]
    base = ruta / "datos_yolo"
    for subconjunto, destino in (("train", "train"), ("val", "val")):
        filas = tabla[tabla["subconjunto"] == subconjunto]
        if filas.empty:
            raise ValueError(f"No hay imágenes para {subconjunto}")
        for indice, fila in filas.iterrows():
            origen = Path(fila["ruta"])
            if not origen.is_file():
                raise FileNotFoundError(origen)
            etiqueta = int(fila["etiqueta"])
            if not 0 <= etiqueta < len(clases):
                raise ValueError(f"Etiqueta no válida: {etiqueta}")
            carpeta = base / destino / clases[etiqueta]
            carpeta.mkdir(parents=True, exist_ok=True)
            destino_archivo = carpeta / f"{indice:06d}_{origen.name}"
            if not destino_archivo.exists():
                try:
                    destino_archivo.symlink_to(origen)
                except OSError:
                    shutil.copy2(origen, destino_archivo)
    return base


def entrenar_yolo(config, modelo_config, estrategia, semilla, ruta):
    from ultralytics import YOLO

    base = preparar_directorios_yolo(config, ruta)
    modelo_id = modelo_config["modelo"][
        "arquitectura" if estrategia == "desde_cero" else "pesos_preentrenados"
    ]
    modelo = YOLO(modelo_id)
    lr = config["model"]["lr_desde_cero" if estrategia == "desde_cero" else "lr_transferencia"]
    epocas = config["trainer"][estrategia]["max_epochs"]
    modelo.train(
        data=str(base),
        epochs=epocas,
        patience=config["trainer"]["early_stopping"]["patience"],
        imgsz=modelo_config["modelo"]["tamano_imagen"],
        batch=config["data"]["batch_size"],
        seed=semilla,
        optimizer="AdamW",
        lr0=lr,
        weight_decay=config["model"]["weight_decay"],
        project=str(ruta),
        name="yolo_entrenamiento",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=config["data"]["num_workers"],
        plots=True,
    )
    checkpoint = ruta / "yolo_entrenamiento" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("No se guardó el checkpoint de YOLO.")
    # Evaluar el mejor checkpoint sobre VALIDACIÓN, nunca sobre test.
    mejor = YOLO(str(checkpoint))
    metricas = mejor.val(data=str(base), split="val", imgsz=modelo_config["modelo"]["tamano_imagen"],
                        batch=config["data"]["batch_size"], device=0 if torch.cuda.is_available() else "cpu")
    # Ultralytics informa accuracy top-1; F1 macro se calculará desde predicciones por imagen.
    from sklearn.metrics import f1_score
    from PIL import Image
    import pandas as pd
    tabla = pd.read_csv(Path(config["resultados"]["ruta"]) / "datos" / "particiones.csv")
    filas = tabla[tabla["subconjunto"] == "val"]
    reales, predichas = [], []
    for _, fila in filas.iterrows():
        with Image.open(fila["ruta"]) as imagen:
            resultado = mejor.predict(source=imagen.convert("RGB"), imgsz=modelo_config["modelo"]["tamano_imagen"],
                                      device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
        reales.append(int(fila["etiqueta"]))
        predichas.append(int(resultado.probs.top1))
    f1 = f1_score(reales, predichas, labels=list(range(config["model"]["num_classes"])), average="macro", zero_division=0)
    guardar_resultado(ruta, modelo_config["nombre"], estrategia, semilla, f1, checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comando", choices=["fit"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--estrategia", choices=["desde_cero", "transferencia"], required=True)
    parser.add_argument("--seed_everything", type=int, required=True)
    args = parser.parse_args()

    config = leer_yaml(args.config)
    modelo_config = leer_yaml(args.modelo)
    nombre = modelo_config["nombre"]
    if nombre not in config["model"]["backbones"]:
        raise ValueError("Modelo no registrado en fit.yaml")
    if args.estrategia not in config["model"]["estrategias"]:
        raise ValueError("Estrategia no registrada en fit.yaml")
    if args.seed_everything not in config["seed_everything"]:
        raise ValueError("Semilla no registrada en fit.yaml")
    if modelo_config["modelo"]["num_classes"] != config["model"]["num_classes"]:
        raise ValueError("Número de clases inconsistente")

    ruta = (Path(config["resultados"]["ruta"]) / "experimento_una_fase_2026"
            / "entrenamientos" / nombre / args.estrategia / f"semilla_{args.seed_everything}")
    if (ruta / "resultado.json").exists():
        print(f"Entrenamiento ya completado: {ruta / 'resultado.json'}")
        return
    ruta.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed_everything, workers=True)
    if modelo_config["biblioteca"] == "timm":
        entrenar_timm(config, modelo_config, args.estrategia, args.seed_everything, ruta)
    elif modelo_config["biblioteca"] == "ultralytics":
        entrenar_yolo(config, modelo_config, args.estrategia, args.seed_everything, ruta)
    else:
        raise ValueError("Biblioteca no reconocida")


if __name__ == "__main__":
    main()

