
"""Preparación reproducible del dataset de Aloe vera."""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def identificar_imagen(ruta):
    """Verifica una imagen y calcula su huella digital."""
    try:
        with Image.open(ruta) as imagen:
            imagen.verify()

        sha256 = hashlib.sha256()
        with ruta.open("rb") as archivo:
            for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
                sha256.update(bloque)

        return sha256.hexdigest()

    except (OSError, ValueError, UnidentifiedImageError):
        return None


def preparar(config):
    datos = config["data"]
    origen = Path(datos["ruta"])
    clases = datos["clases"]
    proporcion = datos["particion"]
    semilla = datos["semilla_particion"]

    if not origen.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {origen}")

    if len(clases) != len(set(clases)):
        raise ValueError("Hay nombres de clases repetidos.")

    if abs(sum(proporcion.values()) - 1.0) > 1e-8:
        raise ValueError("Las proporciones deben sumar 1.")

    grupos = defaultdict(list)
    invalidas = []

    for etiqueta, clase in enumerate(clases):
        carpeta = origen / clase

        if not carpeta.is_dir():
            raise FileNotFoundError(f"Falta la clase: {carpeta}")

        for ruta in sorted(carpeta.rglob("*")):
            if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES:
                continue

            huella = identificar_imagen(ruta)

            if huella is None:
                invalidas.append(str(ruta))
            else:
                grupos[huella].append((ruta, clase, etiqueta))

    if not grupos:
        raise ValueError("No se encontraron imágenes válidas.")

    # Una misma imagen exacta no puede tener etiquetas distintas.
    for huella, elementos in grupos.items():
        if len({elemento[2] for elemento in elementos}) > 1:
            raise ValueError(
                "Se detectó una imagen idéntica en distintas clases: "
                + huella
            )

    # Dividir por grupos de imágenes idénticas, no por archivos.
    por_clase = defaultdict(list)

    for huella, elementos in grupos.items():
        por_clase[elementos[0][2]].append(huella)

    asignaciones = {}

    for etiqueta, huellas in sorted(por_clase.items()):
        if len(huellas) < 3:
            raise ValueError(
                f"La clase {clases[etiqueta]} tiene menos de tres grupos."
            )

        entrenamiento, temporal = train_test_split(
            sorted(huellas),
            train_size=proporcion["entrenamiento"],
            random_state=semilla,
        )

        fraccion_validacion = (
            proporcion["validacion"]
            / (proporcion["validacion"] + proporcion["prueba"])
        )

        validacion, prueba = train_test_split(
            temporal,
            train_size=fraccion_validacion,
            random_state=semilla,
        )

        for subconjunto, seleccion in (
            ("train", entrenamiento),
            ("val", validacion),
            ("test", prueba),
        ):
            for huella in seleccion:
                asignaciones[huella] = subconjunto

    salida = Path(config["resultados"]["ruta"]) / "datos"
    salida.mkdir(parents=True, exist_ok=True)

    manifiesto = salida / "particiones.csv"
    conteos = Counter()

    with manifiesto.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            ["ruta", "clase", "etiqueta", "subconjunto", "sha256"]
        )

        for huella, elementos in sorted(grupos.items()):
            subconjunto = asignaciones[huella]

            for ruta, clase, etiqueta in elementos:
                escritor.writerow(
                    [str(ruta), clase, etiqueta, subconjunto, huella]
                )
                conteos[(subconjunto, clase)] += 1

    print("Archivo generado:", manifiesto)
    print("Imágenes válidas:", sum(conteos.values()))
    print("Imágenes no válidas:", len(invalidas))
    print("Grupos de imágenes idénticas:", len(grupos))

    for subconjunto in ("train", "val", "test"):
        total = sum(
            conteos[(subconjunto, clase)] for clase in clases
        )
        print(f"{subconjunto}: {total}")

    if invalidas:
        reporte = salida / "imagenes_invalidas.txt"
        reporte.write_text(
            "\n".join(invalidas), encoding="utf-8"
        )
        print("Reporte de imágenes inválidas:", reporte)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fit.yaml",
        help="Ruta del archivo YAML de configuración",
    )
    argumentos = parser.parse_args()

    with open(argumentos.config, encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)

    preparar(configuracion)
