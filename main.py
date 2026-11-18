"""Entrenamiento reproducible de clasificadores timm para Aloe vera.

Uso: python main.py fit --config configs/fit.yaml --modelo configs/mobilenetv4_conv_small.yaml --estrategia desde_cero --seed_everything 42
"""
import argparse
import csv
import json
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassF1Score

from dataset import crear_dataloaders
from model import crear_modelo, congelar_base, descongelar_modelo


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
        loss = F.cross_entropy(self(x), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.f1_val.update(logits, y)
        self.log("val/loss", F.cross_entropy(logits, y), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val/f1_macro", self.f1_val.compute(), prog_bar=True)
        self.f1_val.reset()

    def configure_optimizers(self):
        parametros = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(parametros, lr=self.lr, weight_decay=self.weight_decay)


def cargar_yaml(ruta):
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preparar_datos(config, imagen):
    alto, ancho = imagen["input_size"][-2:]
    if alto != ancho:
        raise ValueError("Este proyecto requiere imágenes cuadradas.")
    return crear_dataloaders(
        config,
        tamano=alto,
        media=imagen.get("mean", (0.485, 0.456, 0.406)),
        desviacion=imagen.get("std", (0.229, 0.224, 0.225)),
        interpolacion=imagen.get("interpolation", "bicubic"),
        recorte=imagen.get("crop_pct", 1.0),
        biblioteca="timm",
    )


def entrenar_fase(red, datos, config, ruta, nombre_fase, epocas, lr, paciencia):
    modulo = Clasificador(red, config["model"]["num_classes"], lr, config["model"]["weight_decay"])
    monitor = config["trainer"]["early_stopping"]["monitor"]
    checkpoint = ModelCheckpoint(
        dirpath=ruta / nombre_fase,
        filename="mejor-{epoch:02d}",
        monitor=monitor,
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        max_epochs=epocas,
        accelerator="auto",
        devices=1,
        precision="32-true",
        logger=TensorBoardLogger(save_dir=str(ruta), name="tensorboard", version=nombre_fase),
        callbacks=[checkpoint, EarlyStopping(monitor=monitor, mode="max", patience=paciencia)],
        default_root_dir=str(ruta),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(modulo, train_dataloaders=datos["train"], val_dataloaders=datos["val"])
    if not checkpoint.best_model_path:
        raise RuntimeError("No se guardó ningún checkpoint.")
    pesos = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    modulo.load_state_dict(pesos["state_dict"])
    return modulo.red, str(checkpoint.best_model_path), float(checkpoint.best_model_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comando", choices=["fit"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--estrategia", choices=["desde_cero", "transferencia"], required=True)
    parser.add_argument("--seed_everything", type=int, required=True)
    args = parser.parse_args()
    config = cargar_yaml(args.config)
    modelo_config = cargar_yaml(args.modelo)
    nombre = modelo_config["nombre"]
    if nombre not in config["model"]["backbones"]:
        raise ValueError("Modelo no registrado en fit.yaml")
    if args.seed_everything not in config["seed_everything"]:
        raise ValueError("Semilla no registrada en fit.yaml")
    if modelo_config["biblioteca"] != "timm":
        raise NotImplementedError("YOLO se integrará en la siguiente etapa; no se ha entrenado.")
    if modelo_config["modelo"]["num_classes"] != config["model"]["num_classes"]:
        raise ValueError("Número de clases inconsistente.")
    ruta = Path(config["resultados"]["ruta"]) / "entrenamientos" / nombre / args.estrategia / f"semilla_{args.seed_everything}"
    resultado = ruta / "resultado.json"
    if resultado.exists():
        print(f"Entrenamiento ya completado: {resultado}")
        return
    ruta.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed_everything, workers=True)
    red, imagen = crear_modelo(modelo_config, args.estrategia)
    datos = preparar_datos(config, imagen)
    paciencia = config["trainer"]["early_stopping"]["patience"]
    if args.estrategia == "desde_cero":
        red, checkpoint, f1 = entrenar_fase(red, datos, config, ruta, "desde_cero", config["trainer"]["desde_cero"]["max_epochs"], config["model"]["lr_desde_cero"], paciencia)
    else:
        congelar_base(red)
        red, _, _ = entrenar_fase(red, datos, config, ruta, "fase1", config["trainer"]["transferencia"]["max_epochs_fase1"], config["model"]["lr_fase1"], paciencia)
        descongelar_modelo(red)
        red, checkpoint, f1 = entrenar_fase(red, datos, config, ruta, "fase2", config["trainer"]["transferencia"]["max_epochs_fase2"], config["model"]["lr_fase2"], paciencia)
    with open(resultado, "w", encoding="utf-8") as f:
        json.dump({"modelo": nombre, "estrategia": args.estrategia, "semilla": args.seed_everything, "mejor_f1_validacion": f1, "checkpoint": checkpoint}, f, indent=2, ensure_ascii=False)
    print(f"Terminado. Resultado: {resultado}")
    print("El conjunto test permanece sin utilizar.")


if __name__ == "__main__":
    main()
