
"""Carga y preparación de imágenes de Aloe vera para PyTorch."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Valores de referencia de ImageNet.
# Se pueden sustituir por los específicos de cada arquitectura.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIACION_IMAGENET = (0.229, 0.224, 0.225)


class SabilaDataset(Dataset):
    """Lee las imágenes y etiquetas del manifiesto de particiones."""

    def __init__(self, manifiesto, subconjunto, transformacion=None):
        tabla = pd.read_csv(manifiesto)

        if subconjunto not in {"train", "val", "test"}:
            raise ValueError("Subconjunto no válido.")

        self.datos = tabla[
            tabla["subconjunto"] == subconjunto
        ].reset_index(drop=True)

        if self.datos.empty:
            raise ValueError(f"No hay imágenes en {subconjunto}.")

        self.transformacion = transformacion

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, indice):
        fila = self.datos.iloc[indice]

        with Image.open(Path(fila["ruta"])) as archivo:
            imagen = archivo.convert("RGB")

        if self.transformacion is not None:
            imagen = self.transformacion(imagen)

        return imagen, int(fila["etiqueta"])


def crear_transformaciones(
    tamano=224,
    entrenamiento=False,
    media=MEDIA_IMAGENET,
    desviacion=DESVIACION_IMAGENET,
):
    """Prepara y normaliza imágenes para una arquitectura."""

    if len(media) != 3 or len(desviacion) != 3:
        raise ValueError("La media y desviación deben tener 3 valores RGB.")

    if any(valor <= 0 for valor in desviacion):
        raise ValueError("Las desviaciones deben ser mayores que cero.")

    operaciones = [
        transforms.Resize((tamano, tamano)),
    ]

    if entrenamiento:
        operaciones.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])

    operaciones.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=media,
            std=desviacion,
        ),
    ])

    return transforms.Compose(operaciones)


def crear_dataloaders(
    config,
    tamano=224,
    media=MEDIA_IMAGENET,
    desviacion=DESVIACION_IMAGENET,
    biblioteca="timm",
):
    """Crea los cargadores de entrenamiento, validación y prueba."""

    if biblioteca != "timm":
        raise ValueError(
            "Este cargador es para modelos timm. "
            "YOLO26n-cls utilizará el cargador de Ultralytics."
        )

    ruta = (
        Path(config["resultados"]["ruta"])
        / "datos"
        / "particiones.csv"
    )

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el manifiesto: {ruta}"
        )

    datos = config["data"]
    cargadores = {}

    for subconjunto in ("train", "val", "test"):
        dataset = SabilaDataset(
            manifiesto=ruta,
            subconjunto=subconjunto,
            transformacion=crear_transformaciones(
                tamano=tamano,
                entrenamiento=(subconjunto == "train"),
                media=media,
                desviacion=desviacion,
            ),
        )

        cargadores[subconjunto] = DataLoader(
            dataset,
            batch_size=datos["batch_size"],
            shuffle=(subconjunto == "train"),
            num_workers=datos["num_workers"],
            pin_memory=True,
        )

    return cargadores
