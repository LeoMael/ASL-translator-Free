import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

class ImprovedASLTrainer:
    def __init__(self, data_dir="asl_dataset", model_dir="asl_models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.asl_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['SPACE', 'DELETE']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.asl_alphabet)
        
        self.X_data = []
        self.y_data = []
        self.model = None
        
    def augment_data(self, landmarks):
        """Aplica data augmentation para mejorar la generalización"""
        augmented_samples = []
        
        # Original
        augmented_samples.append(landmarks)
        
        # 1. Añadir ruido gaussiano (simula variaciones naturales)
        for _ in range(3):
            noise = np.random.normal(0, 0.01, landmarks.shape)
            augmented_samples.append(landmarks + noise)
        
        # 2. Escalar (simula diferentes distancias)
        scales = [0.9, 0.95, 1.05, 1.1]
        for scale in scales:
            augmented_samples.append(landmarks * scale)
        
        # 3. Pequeñas rotaciones (simula diferentes ángulos)
        for angle in [-5, 5]:  # grados
            angle_rad = np.radians(angle)
            # Aplicar rotación simple en 2D a las coordenadas x,y
            rotated = landmarks.copy()
            for i in range(0, len(landmarks), 3):
                x, y = landmarks[i], landmarks[i+1]
                rotated[i] = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                rotated[i+1] = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            augmented_samples.append(rotated)
        
        # 4. Traslación (simula diferentes posiciones de la mano)
        translations = [(0.02, 0.02), (-0.02, -0.02), (0.02, -0.02), (-0.02, 0.02)]
        for tx, ty in translations:
            translated = landmarks.copy()
            for i in range(0, len(landmarks), 3):
                translated[i] += tx
                translated[i+1] += ty
            augmented_samples.append(translated)
        
        return augmented_samples
    
    def load_data_with_augmentation(self):
        """Carga datos con data augmentation"""
        print("=== CARGANDO DATOS CON AUGMENTATION ===")
        
        for letter in self.asl_alphabet:
            letter_dir = os.path.join(self.data_dir, letter)
            
            if not os.path.exists(letter_dir):
                print(f"⚠️  No se encontraron datos para '{letter}'")
                continue
            
            files = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
            original_count = len(files)
            augmented_count = 0
            
            for file in files:
                filepath = os.path.join(letter_dir, file)
                try:
                    landmarks = np.load(filepath)
                    
                    # Aplicar augmentation
                    augmented_samples = self.augment_data(landmarks)
                    
                    for sample in augmented_samples:
                        self.X_data.append(sample)
                        self.y_data.append(letter)
                        augmented_count += 1
                        
                except Exception as e:
                    print(f"Error cargando {filepath}: {e}")
            
            print(f"✓ '{letter}': {original_count} originales → {augmented_count} con augmentation")
        
        self.X_data = np.array(self.X_data)
        self.y_data = np.array(self.y_data)
        
        print(f"\nTotal de muestras: {len(self.X_data)}")
        print(f"Factor de aumento: {len(self.X_data) / (len(self.asl_alphabet) * 100):.1f}x")
    
    def create_robust_model(self):
        """Crea un modelo más robusto con regularización"""
        print("\n=== CREANDO MODELO ROBUSTO ===")
        
        input_shape = self.X_train.shape[1]
        num_classes = len(self.asl_alphabet)
        
        self.model = Sequential([
            # Capa de entrada con ruido para mayor robustez
            GaussianNoise(0.01, input_shape=(input_shape,)),
            
            # Primera capa con regularización L2
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),  # Mayor dropout
            
            # Segunda capa
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Tercera capa
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Cuarta capa
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Capa de salida
            Dense(num_classes, activation='softmax')
        ])
        
        # Compilar con learning rate más bajo
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),  # LR más bajo
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
    
    def train_robust(self, epochs=150, batch_size=64):
        """Entrena con configuración más robusta"""
        print("\n=== ENTRENANDO MODELO ROBUSTO ===")
        
        # Preparar datos
        y_encoded = self.label_encoder.transform(self.y_data)
        y_categorical = to_categorical(y_encoded)
        
        # División estratificada con más datos de validación
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_data, y_categorical, 
            test_size=0.3,  # 30% para validación
            random_state=42, 
            stratify=y_encoded
        )
        
        print(f"Datos de entrenamiento: {self.X_train.shape}")
        print(f"Datos de validación: {self.X_test.shape}")
        
        # Crear modelo
        self.create_robust_model()
        
        # Callbacks mejorados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,  # Más paciencia
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'robust_model_{timestamp}.h5'),
                monitor='val_loss',  # Monitorear loss en lugar de accuracy
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.000001,
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
        
        # Guardar
        model_path = os.path.join(self.model_dir, f'asl_robust_{timestamp}.h5')
        self.model.save(model_path)
        
        # Guardar encoder
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n✓ Modelo guardado en: {model_path}")
        
        # Evaluar diferencia entre train y validation
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        print(f"\n📊 Métricas finales:")
        print(f"Training accuracy: {train_acc:.2%}")
        print(f"Validation accuracy: {val_acc:.2%}")
        print(f"Diferencia (overfitting): {(train_acc - val_acc):.2%}")
        
        if train_acc - val_acc > 0.1:
            print("\n⚠️ ADVERTENCIA: Posible overfitting detectado!")
            print("Considera recolectar más datos o aumentar la regularización.")
        
        return history
    
    def plot_detailed_history(self, history):
        """Grafica con más detalle para detectar overfitting"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        plt.title('Accuracy vs Época')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
        plt.title('Loss vs Época')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Diferencia (Overfitting)
        plt.subplot(1, 3, 3)
        train_acc = np.array(history.history['accuracy'])
        val_acc = np.array(history.history['val_accuracy'])
        overfitting = train_acc - val_acc
        
        plt.plot(overfitting, 'r-', linewidth=2)
        plt.axhline(y=0.1, color='orange', linestyle='--', label='Umbral de alerta')
        plt.title('Overfitting (Train - Val Accuracy)')
        plt.xlabel('Época')
        plt.ylabel('Diferencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'detailed_training_history.png'))
        plt.show()

def main():
    trainer = ImprovedASLTrainer()
    
    print("=== ENTRENAMIENTO MEJORADO ASL ===")
    print("Este entrenamiento incluye:")
    print("✓ Data Augmentation")
    print("✓ Regularización L2")
    print("✓ Mayor Dropout")
    print("✓ Detección de Overfitting\n")
    
    # Cargar datos con augmentation
    trainer.load_data_with_augmentation()
    
    if len(trainer.X_data) == 0:
        print("\n❌ No se encontraron datos para entrenar.")
        return
    
    # Entrenar
    history = trainer.train_robust(epochs=150, batch_size=64)
    
    # Visualizar
    trainer.plot_detailed_history(history)
    
    print("\n✅ ENTRENAMIENTO COMPLETADO")
    print("El modelo ahora debería generalizar mejor en condiciones reales.")

if __name__ == "__main__":
    main()