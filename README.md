# Traductor de Lenguaje de Señas (ASL) en Tiempo Real

Este proyecto implementa un sistema interactivo para la detección y traducción del Alfabeto Dactilológico del Lenguaje de Señas Americano (ASL) en tiempo real utilizando la cámara web.

El sistema utiliza MediaPipe Hands para la extracción de puntos clave (landmarks) de la mano en tres dimensiones y modelos de aprendizaje profundo construidos en TensorFlow / Keras para clasificar las señas en letras, espacios o retrocesos (delete).

---

## Referencias y Créditos

* **Dataset:** Los datos de puntos clave (landmarks) de entrenamiento fueron recopilados y generados por el autor del proyecto mediante la herramienta de captura incluida.
* **Alfabeto ASL de Referencia:** El modelado de las señas y el entrenamiento del clasificador toman como base la guía dactilológica oficial de ASL de AI-Media:  
  [AI-Media ASL Alphabet Guide](https://www.ai-media.tv/wp-content/uploads/ASL_Alphabet.jpg)

---

## Estructura del Proyecto

El repositorio está organizado con los siguientes scripts numerados según su flujo de ejecución lógica:

* **`0_collect_data.py`** (Opcional): Interfaz de captura de landmarks. Permite recolectar 100 muestras (coordenadas x, y, z normalizadas de 21 puntos clave) por cada letra del alfabeto y guardarlas como archivos `.npy` dentro del dataset. Este paso es opcional ya que el repositorio incluye un dataset listo para usar.
* **`1_train_basic.py`**: Pipeline para cargar los datos del dataset y entrenar un clasificador básico usando una red neuronal densa (FNN) o una red de memoria a corto-largo plazo (LSTM).
* **`2_train_robust.py`**: Pipeline de entrenamiento avanzado. Aplica técnicas de Data Augmentation (adición de ruido gaussiano, escalado, rotación en 2D y traslación a los puntos clave) y regularización L2 con mayor Dropout para evitar el sobreajuste (overfitting).
* **`3_export_to_tfmobile.py`**: Convierte el modelo Keras entrenado (`.h5`) al formato optimizado para dispositivos móviles TensorFlow Lite (`.tflite`) y exporta las clases a un archivo JSON (`labels.json`).
* **`4_translator_gui.py`**: Aplicación principal con una interfaz gráfica construida en CustomTkinter y control de cámara por OpenCV. Cuenta con:
  * Traducción continua en tiempo real.
  * Borrado inteligente por gestos (por carácter, palabra o limpiar todo).
  * Síntesis de voz integrada mediante pyttsx3 para reproducir las palabras completadas en audio.
  * Panel de estadísticas (palabras por minuto, letras más detectadas, precisión estimada).
* **`asl_dataset.zip`**: Archivo comprimido que contiene la carpeta `asl_dataset/` con todas las muestras `.npy` y las imágenes de referencia. Se incluye comprimido para mantener la velocidad de Git. Debe descomprimirse antes de ejecutar los scripts.
* **`asl_dataset/`**: Directorio que se genera tras descomprimir `asl_dataset.zip`. Contiene las subcarpetas con las muestras y las imágenes de referencia utilizadas durante la recolección de señas. (Ignorado en Git).
* **`asl_models/`**: Directorio donde se guardan los encoders de etiquetas (`label_encoder.pkl`), los reportes de rendimiento y los archivos del modelo entrenado (`.h5`).
* **`asl_models_mobile/`**: Modelos optimizados (`.tflite` y `labels.json`) listos para ser desplegados en aplicaciones móviles.
* **`requirements.txt`**: Archivo de especificación de dependencias Python.

---

## Gestos Especiales Personalizados

El sistema incluye detección de gestos especiales para controlar el flujo de traducción de forma natural:

* **DELETE**: Se activa mostrando **toda la palma abierta** frente a la cámara. Borra caracteres en tiempo real.
* **SPACE**: Se activa colocando la **mano plana en posición completamente horizontal**. Inserta un espacio para separar palabras y activa la lectura en voz alta del texto acumulado.

---

## Requisitos e Instalación

### Requisitos del Sistema
* **Python 3.11** (Requerido para asegurar compatibilidad con versiones específicas de las librerías).
* Cámara web funcional.

---

### Guía de Configuración

Siga los siguientes pasos para configurar el entorno e instalar las dependencias.

#### En Windows (PowerShell / CMD)
1. Abra una terminal y navegue a la carpeta de su proyecto:
   ```powershell
   cd \ruta\a\la\carpeta\del\proyecto
   ```
2. Cree el entorno virtual utilizando Python 3.11:
   ```powershell
   py -3.11 -m venv env_asl
   ```
3. Active el entorno virtual:
   * **En PowerShell:**
     ```powershell
     .\env_asl\Scripts\Activate.ps1
     ```
   * **En CMD:**
     ```cmd
     .\env_asl\Scripts\activate.bat
     ```
4. Actualice `pip` e instale las dependencias:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

#### En Fedora Linux
1. Instale Python 3.11, soporte gráfico de tkinter y dependencias de voz:
   ```bash
   sudo dnf install python3.11 python3.11-tkinter python3.11-devel espeak-ng espeak-ng-devel
   ```
2. Navegue a la carpeta y cree el entorno virtual:
   ```bash
   cd /ruta/a/la/carpeta/del/proyecto
   ```
3. Cree y active el entorno virtual:
   ```bash
   python3.11 -m venv env_asl
   source env_asl/bin/activate
   ```
4. Actualice `pip` e instale las dependencias:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

*(Nota: En `requirements.txt` se han fijado `mediapipe==0.10.14` y `numpy<2` para evitar errores de incompatibilidad con las APIs internas de MediaPipe).*

---

Con el entorno virtual activo, siga este orden para la ejecución:

### Paso Previo: Descomprimir el Dataset
Antes de ejecutar los scripts de entrenamiento o interfaz, descomprima el archivo **`asl_dataset.zip`** en el directorio raíz del proyecto de modo que se genere la carpeta **`asl_dataset/`** conteniendo las carpetas de las señas (A, B, C, etc.).

### Paso 0. Recolección de Datos (Opcional)
Si desea expandir el dataset con nuevas señas personalizadas o registrar un nuevo usuario:
```bash
python 0_collect_data.py
```
* **Controles en pantalla:**
  * **S:** Iniciar o detener la recolección automática de 100 muestras de la letra actual.
  * **C:** Capturar y guardar una foto de referencia de la seña actual.
  * **V:** Mostrar u ocultar la imagen de guía de referencia en la esquina de la pantalla.
  * **N / P:** Avanzar a la siguiente letra o regresar a la anterior.
  * **R:** Reiniciar y limpiar las muestras de la letra actual.
  * **Q:** Guardar el resumen y salir.

### Paso 1. Entrenamiento Básico (Opcional)
Para entrenar una red neuronal densa clásica o un modelo LSTM sin técnicas avanzadas de aumentación:
```bash
python 1_train_basic.py
```

### Paso 2. Entrenamiento Robustecido
Para entrenar el modelo definitivo y evitar el sobreajuste ante variaciones reales de iluminación, posición y escala:
```bash
python 2_train_robust.py
```
Este script dividirá los datos en entrenamiento y validación (70/30), aplicará Data Augmentation y guardará el mejor modelo en la carpeta `asl_models/`.

### Paso 3. Conversión para Dispositivos Móviles
Exporte el modelo Keras a un archivo optimizado para dispositivos móviles:
```bash
python 3_export_to_tfmobile.py
```
Los archivos de salida `model.tflite` y `labels.json` se guardarán en la carpeta `asl_models/`.

### Paso 4. Ejecución del Traductor con Interfaz Gráfica
Ejecute la aplicación en tiempo real:
```bash
python 4_translator_gui.py
```
* **Características principales:**
  * Haga clic en **Iniciar Cámara** para comenzar a traducir.
  * Ajuste la sensibilidad de detección y velocidad desde los controles del panel izquierdo.
  * Realice señas estables para registrar caracteres. Para añadir espacios, realice la seña de **SPACE** y la palabra en búfer se reproducirá por audio automáticamente.
  * Mantenga la seña de **DELETE** para borrar un carácter. Si la sostiene durante 2 segundos borrará la palabra completa, y si la sostiene por 4 segundos limpiará toda la caja de texto.
  * Copie el texto al portapapeles o guárdelo en un archivo de texto con los botones de la interfaz.
