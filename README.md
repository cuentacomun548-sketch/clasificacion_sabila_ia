# Clasificación multiclase de imágenes de sábila mediante Inteligencia Artificial

## 1. Descripción del proyecto

Este proyecto tiene como objetivo desarrollar y evaluar modelos de aprendizaje profundo para la clasificación automática de imágenes de sábila (*Aloe vera*) en tres categorías: Fresh, Rot y Rust.

El trabajo comprende la organización, auditoría, depuración y versionado del conjunto de datos, así como la preparación de un entorno experimental reproducible para el entrenamiento y evaluación de modelos de clasificación de imágenes.

La metodología de entrenamiento y las arquitecturas se definirán durante la etapa de diseño experimental.

## 2. Objetivo

Implementar un sistema de clasificación multiclase de imágenes de sábila mediante técnicas de aprendizaje profundo, utilizando un conjunto de datos auditado y un procedimiento experimental documentado y reproducible.

## 3. Descripción del dataset

El conjunto de datos original contiene 2,337 imágenes distribuidas en tres categorías.

| Clase | Imágenes originales | Imágenes después de excluir duplicados exactos |
|---|---:|---:|
| Fresh Aloe Vera | 630 | 630 |
| Rot Aloe Vera | 838 | 838 |
| Rust Aloe Vera | 869 | 850 |
| **Total** | **2,337** | **2,318** |

Las denominaciones originales de las clases se conservan para mantener la correspondencia con las carpetas del dataset.

### 3.1 Auditoría de calidad

Se realizaron las siguientes verificaciones:

- **Integridad:** 2,337 imágenes verificadas mediante Pillow, sin archivos corruptos detectados.
- **Duplicados exactos:** identificación de 19 copias adicionales mediante SHA-256.
- **Inventario:** registro de los 2,337 archivos, incluyendo nombre, clase, ruta relativa, hash y estado de conservación.
- **Depuración:** generación de una copia de trabajo con 2,318 imágenes.
- **Verificación de copias:** comparación SHA-256 de las 2,318 imágenes seleccionadas, sin discrepancias.
- **Similitud perceptual:** identificación de 9 grupos con dHash idéntico, que involucran 20 imágenes pendientes de revisión visual.

Las coincidencias perceptuales se consideran candidatas a revisión y no se eliminan automáticamente.

## 4. Organización del proyecto

El proyecto utiliza GitHub para controlar las versiones del código y la documentación, Google Drive para almacenar el conjunto de imágenes y Google Colab como entorno de ejecución.

### 4.1 Repositorio GitHub

Estructura actual:

```text
clasificacion_sabila_ia/
├── configs/
│   └── fit.yaml
├── notebooks/
├── .gitignore
└── README.md
```

- `configs/`: archivos de configuración experimental.
- `notebooks/`: cuadernos de trabajo y experimentación.
- `.gitignore`: reglas de exclusión para archivos que no deben incorporarse al repositorio.
- `README.md`: documentación principal del proyecto.

La estructura podrá ampliarse conforme se implementen los módulos y experimentos.

### 4.2 Almacenamiento en Google Drive

```text
Mi unidad/
├── DATASET_ALOE VERA/
│   ├── Fresh Aloe Vera/
│   ├── Rot Aloe Vera/
│   └── Rust Aloe Vera/
│
└── proyecto_sabila/
    ├── auditoria/
    │   ├── duplicados_sha256.csv
    │   └── inventario_dataset.csv
    │
    └── datasets/
        └── v1.0.0/
            ├── Fresh Aloe Vera/
            ├── Rot Aloe Vera/
            └── Rust Aloe Vera/
```

La carpeta `DATASET_ALOE VERA` contiene el conjunto original de 2,337 imágenes.

La carpeta `v1.0.0` contiene una copia de trabajo con 2,318 imágenes, después de excluir las 19 copias exactas identificadas.

## 5. Control de versiones y trazabilidad

Se establece una separación entre el versionado del código y la gestión de los datos.

**Código y documentación**

Git permite registrar cambios en scripts, notebooks, configuraciones y documentos mediante commits descriptivos.

**Dataset**

Google Drive conserva los archivos de imagen. Los inventarios y las huellas SHA-256 permiten identificar los archivos seleccionados y comprobar su integridad.

La carpeta `v1.0.0` corresponde actualmente a una versión de trabajo pendiente de cierre formal. Cualquier depuración posterior deberá documentarse y producir una nueva versión, conservando la anterior.

Los archivos de imágenes y los modelos entrenados no se incorporarán directamente al repositorio GitHub.

## 6. Procedimiento de preparación de datos

El procedimiento aplicado comprende:

1. Identificación de las tres categorías del
