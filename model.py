
"""Construcción de modelos para clasificar imágenes de sábila."""

import timm
from timm.data import resolve_model_data_config


def crear_modelo(config_modelo, estrategia):
    """
    Construye un modelo timm para tres clases de sábila.

    Estrategias:
        desde_cero: pesos inicializados aleatoriamente.
        transferencia: pesos preentrenados.
    """

    if estrategia not in ("desde_cero", "transferencia"):
        raise ValueError("Estrategia no válida.")

    if config_modelo["biblioteca"] != "timm":
        raise ValueError(
            "Este constructor es para modelos timm. "
            "YOLO26n-cls se manejará con Ultralytics."
        )

    datos = config_modelo["modelo"]

    arquitectura = datos["arquitectura"]
    num_clases = datos["num_classes"]

    usar_preentrenado = estrategia == "transferencia"

    if usar_preentrenado:
        identificador = datos["pesos_preentrenados"]
    else:
        identificador = arquitectura

    modelo = timm.create_model(
        identificador,
        pretrained=usar_preentrenado,
        num_classes=num_clases,
    )

    # Consulta los requisitos de entrada del modelo.
    configuracion_imagen = resolve_model_data_config(modelo)

    return modelo, configuracion_imagen


def congelar_base(modelo):
    """Congela la base y permite entrenar solo el clasificador."""

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    clasificador = modelo.get_classifier()

    if not hasattr(clasificador, "parameters"):
        raise ValueError(
            "No se pudo identificar el clasificador del modelo."
        )

    for parametro in clasificador.parameters():
        parametro.requires_grad = True

    return modelo


def descongelar_modelo(modelo):
    """Permite ajustar todos los parámetros del modelo."""

    for parametro in modelo.parameters():
        parametro.requires_grad = True

    return modelo
