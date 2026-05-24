import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from datetime import datetime

class ASLTrainer:
    def __init__(self, data_dir="asl_dataset", model_dir="asl_models"):
        """
        Inicializa el entrenador ASL
        
        Args:
            data_dir: Directorio con los datos recolectados
            model_dir: Directorio donde se guardarán los modelos
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Crear directorio para modelos si no existe
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Alfabeto ASL
        self.asl_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['SPACE', 'DELETE']
        
        # Encoder para las etiquetas
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.asl_alphabet)
        
        # Datos
        self.X_data = []
        self.y_data = []
        
        # Modelo
        self.model = None
        
    def load_data(self):
        """Carga todos los datos recolectados"""
        print("=== CARGANDO DATOS ===")
        
        for letter in self.asl_alphabet:
            letter_dir = os.path.join(self.data_dir, letter)
            
            if not os.path.exists(letter_dir):
                print(f"⚠️  No se encontraron datos para '{letter}'")
                continue
            
            # Cargar todos los archivos .npy de esta letra
            files = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
            
            for file in files:
                filepath = os.path.join(letter_dir, file)
                try:
                    # Cargar landmarks
                    landmarks = np.load(filepath)
                    self.X_data.append(landmarks)
                    self.y_data.append(letter)
                except Exception as e:
                    print(f"Error cargando {filepath}: {e}")
            
            print(f"✓ Cargadas {len(files)} muestras para '{letter}'")
        
        # Convertir a arrays numpy
        self.X_data = np.array(self.X_data)
        self.y_data = np.array(self.y_data)
        
        print(f"\nTotal de muestras cargadas: {len(self.X_data)}")
        print(f"Forma de los datos: {self.X_data.shape}")
        
        # Mostrar distribución de clases
        unique, counts = np.unique(self.y_data, return_counts=True)
        print("\nDistribución de clases:")
        for letter, count in zip(unique, counts):
            print(f"  {letter}: {count} muestras")
    
    def preprocess_data(self):
        """Preprocesa los datos para el entrenamiento"""
        print("\n=== PREPROCESANDO DATOS ===")
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.transform(self.y_data)
        y_categorical = to_categorical(y_encoded)
        
        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_data, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Datos de entrenamiento: {self.X_train.shape}")
        print(f"Datos de prueba: {self.X_test.shape}")
        
        # Guardar el encoder
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def create_model(self):
        """Crea el modelo de red neuronal"""
        print("\n=== CREANDO MODELO ===")
        
        input_shape = self.X_train.shape[1]
        num_classes = len(self.asl_alphabet)
        
        self.model = Sequential([
            # Primera capa densa
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Segunda capa
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Tercera capa
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Cuarta capa
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Capa de salida
            Dense(num_classes, activation='softmax')
        ])
        
        # Compilar modelo
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
    
    def create_lstm_model(self):
        """Crea un modelo LSTM para capturar secuencias temporales"""
        print("\n=== CREANDO MODELO LSTM ===")
        
        # Reshape para LSTM (samples, timesteps, features)
        # Usamos los landmarks como una secuencia
        timesteps = 21  # 21 puntos de la mano
        features = 3    # x, y, z por punto
        
        # Reshape los datos
        self.X_train = self.X_train.reshape(-1, timesteps, features)
        self.X_test = self.X_test.reshape(-1, timesteps, features)
        
        num_classes = len(self.asl_alphabet)
        
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
            Dropout(0.3),
            
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            
            LSTM(64),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
    
    def train(self, epochs=50, batch_size=32, use_lstm=False):
        """Entrena el modelo"""
        print("\n=== ENTRENANDO MODELO ===")
        
        # Crear el modelo apropiado
        if use_lstm:
            self.create_lstm_model()
        else:
            self.create_model()
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'best_model_{timestamp}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Entrenar
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar el modelo final
        model_path = os.path.join(self.model_dir, f'asl_model_{timestamp}.h5')
        self.model.save(model_path)
        print(f"\n✓ Modelo guardado en: {model_path}")
        
        # Guardar historial
        history_path = os.path.join(self.model_dir, f'history_{timestamp}.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        
        return history
    
    def plot_training_history(self, history):
        """Grafica el historial de entrenamiento"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.show()
    
    def evaluate_model(self):
        """Evalúa el modelo y muestra métricas detalladas"""
        print("\n=== EVALUANDO MODELO ===")
        
        # Predicciones
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Accuracy
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Reporte de clasificación
        class_names = self.label_encoder.classes_
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=class_names)
        print("\nReporte de Clasificación:")
        print(report)
        
        # Guardar reporte
        report_path = os.path.join(self.model_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n\n")
            f.write(report)
        
        # Matriz de confusión
        self.plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Grafica la matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'))
        plt.show()
    
    def save_model_info(self):
        """Guarda información adicional del modelo"""
        info = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_shape': self.X_train.shape[1],
            'num_classes': len(self.asl_alphabet),
            'alphabet': self.asl_alphabet,
            'total_samples': len(self.X_data),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        info_path = os.path.join(self.model_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"\n✓ Información del modelo guardada en: {info_path}")

def main():
    # Crear entrenador
    trainer = ASLTrainer(data_dir="asl_dataset", model_dir="asl_models")
    
    # Pipeline completo
    print("=== SISTEMA DE ENTRENAMIENTO ASL ===\n")
    
    # 1. Cargar datos
    trainer.load_data()
    
    if len(trainer.X_data) == 0:
        print("\n❌ No se encontraron datos para entrenar.")
        print("Por favor, ejecuta primero el recolector de datos.")
        return
    
    # 2. Preprocesar
    trainer.preprocess_data()
    
    # 3. Entrenar (puedes cambiar use_lstm=True para usar LSTM)
    print("\n¿Qué tipo de modelo deseas entrenar?")
    print("1. Red Neuronal Densa (más rápido, buena precisión)")
    print("2. LSTM (más lento, potencialmente mejor para secuencias)")
    
    choice = input("\nSelecciona (1 o 2): ").strip()
    use_lstm = choice == '2'
    
    history = trainer.train(epochs=50, batch_size=32, use_lstm=use_lstm)
    
    # 4. Visualizar resultados
    trainer.plot_training_history(history)
    
    # 5. Evaluar
    trainer.evaluate_model()
    
    # 6. Guardar información
    trainer.save_model_info()
    
    print("\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"Modelos guardados en: {trainer.model_dir}")

if __name__ == "__main__":
    main()