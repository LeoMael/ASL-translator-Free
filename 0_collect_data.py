import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime
import time

class ASLDataCollector:
    def __init__(self, data_dir="asl_dataset"):
        """
        Inicializa el recolector de datos ASL
        
        Args:
            data_dir: Directorio donde se guardarán los datos
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Alfabeto ASL + caracteres especiales (definir antes de crear directorios)
        self.asl_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['SPACE', 'DELETE']
        
        # Directorio para guardar datos
        self.data_dir = data_dir
        self.create_directories()
        
        # Variables de control
        self.current_letter_idx = 0
        self.samples_per_letter = 100
        self.samples_collected = 0
        self.is_collecting = False
        self.collection_delay = 0.1  # Delay entre capturas
        
        # Datos recolectados
        self.current_data = []
        
        # Control para captura de referencia
        self.reference_captured = {letter: False for letter in self.asl_alphabet}
        self.show_reference = False
        self.reference_image = None
        
    def create_directories(self):
        """Crea la estructura de directorios para el dataset"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Crear subdirectorios para cada letra
        for letter in self.asl_alphabet:
            letter_dir = os.path.join(self.data_dir, letter)
            if not os.path.exists(letter_dir):
                os.makedirs(letter_dir)
    
    def extract_hand_landmarks(self, hand_landmarks, image_shape):
        """
        Extrae y normaliza los landmarks de la mano
        
        Returns:
            Array de 63 características (21 puntos x 3 coordenadas)
        """
        landmarks = []
        
        # Obtener coordenadas de referencia (muñeca)
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z
        
        for landmark in hand_landmarks.landmark:
            # Normalizar respecto a la muñeca
            x = landmark.x - wrist_x
            y = landmark.y - wrist_y
            z = landmark.z - wrist_z
            
            landmarks.extend([x, y, z])
        
        return np.array(landmarks)
    
    def save_reference_image(self, image, letter):
        """Guarda una imagen de referencia para la letra"""
        ref_dir = os.path.join(self.data_dir, "referencias")
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        
        filename = f"{letter}_reference.jpg"
        filepath = os.path.join(ref_dir, filename)
        cv2.imwrite(filepath, image)
        self.reference_captured[letter] = True
        print(f"✓ Imagen de referencia guardada para '{letter}'")
    
    def load_reference_image(self, letter):
        """Carga la imagen de referencia de una letra"""
        ref_dir = os.path.join(self.data_dir, "referencias")
        filename = f"{letter}_reference.jpg"
        filepath = os.path.join(ref_dir, filename)
        
        if os.path.exists(filepath):
            return cv2.imread(filepath)
        return None
    
    def save_sample(self, landmarks, letter):
        """Guarda una muestra individual"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{letter}_{timestamp}.npy"
        filepath = os.path.join(self.data_dir, letter, filename)
        
        np.save(filepath, landmarks)
        
    def draw_info(self, image, hand_landmarks):
        """Dibuja información en la imagen"""
        h, w, _ = image.shape
        current_letter = self.asl_alphabet[self.current_letter_idx]
        
        # Dibujar landmarks de la mano
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # Mostrar imagen de referencia si está activado
        if self.show_reference and self.reference_image is not None:
            ref_h, ref_w = self.reference_image.shape[:2]
            # Escalar imagen de referencia
            scale = 0.3
            new_w, new_h = int(ref_w * scale), int(ref_h * scale)
            ref_resized = cv2.resize(self.reference_image, (new_w, new_h))
            
            # Colocar en esquina superior derecha
            y_offset = 90
            x_offset = w - new_w - 10
            image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = ref_resized
            
            # Marco alrededor de la referencia
            cv2.rectangle(image, (x_offset-2, y_offset-2), 
                         (x_offset+new_w+2, y_offset+new_h+2), (255, 255, 0), 2)
            cv2.putText(image, "REFERENCIA", (x_offset, y_offset-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Información en pantalla
        cv2.rectangle(image, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(image, f"Letra: {current_letter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Muestras: {self.samples_collected}/{self.samples_per_letter}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Indicador de referencia capturada
        ref_status = "✓" if self.reference_captured[current_letter] else "✗"
        ref_color = (0, 255, 0) if self.reference_captured[current_letter] else (0, 0, 255)
        cv2.putText(image, f"Ref: {ref_status}", (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ref_color, 2)
        
        # Estado de recolección
        status_color = (0, 255, 0) if self.is_collecting else (0, 0, 255)
        status_text = "RECOLECTANDO" if self.is_collecting else "PRESIONA 'S' PARA INICIAR"
        cv2.putText(image, status_text, (w - 400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Instrucciones
        instructions = [
            "S - Iniciar/Detener recoleccion",
            "C - Capturar imagen de referencia",
            "V - Ver/Ocultar referencia",
            "N - Siguiente letra",
            "P - Letra anterior",
            "R - Reiniciar letra actual",
            "Q - Salir"
        ]
        
        y_offset = h - 180
        for instruction in instructions:
            cv2.putText(image, instruction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Consejos de recolección
        tips = [
            "CONSEJOS:",
            "- Manten la mano a 30-50cm de la camara",
            "- Varia ligeramente posicion y angulo",
            "- Fondo claro y uniforme es mejor"
        ]
        
        y_offset = 120
        for i, tip in enumerate(tips):
            color = (0, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(image, tip, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 20
        
        # Progreso general
        total_progress = (self.current_letter_idx * self.samples_per_letter + self.samples_collected) / (len(self.asl_alphabet) * self.samples_per_letter) * 100
        cv2.putText(image, f"Progreso Total: {total_progress:.1f}%",
                    (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return image
    
    def collect_data(self):
        """Función principal de recolección de datos"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_collection_time = time.time()
        
        print("=== RECOLECTOR DE DATOS ASL ===")
        print(f"Se recolectarán {self.samples_per_letter} muestras por cada letra/símbolo")
        print("Controles disponibles durante la recolección")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Voltear imagen horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Si hay una mano detectada y estamos recolectando
            if results.multi_hand_landmarks and self.is_collecting:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Verificar si ha pasado suficiente tiempo desde la última recolección
                current_time = time.time()
                if current_time - last_collection_time >= self.collection_delay:
                    # Extraer características
                    landmarks = self.extract_hand_landmarks(hand_landmarks, frame.shape)
                    
                    # Guardar muestra
                    current_letter = self.asl_alphabet[self.current_letter_idx]
                    self.save_sample(landmarks, current_letter)
                    
                    self.samples_collected += 1
                    last_collection_time = current_time
                    
                    # Verificar si completamos las muestras para esta letra
                    if self.samples_collected >= self.samples_per_letter:
                        self.is_collecting = False
                        self.samples_collected = 0
                        print(f"\n✓ Completadas {self.samples_per_letter} muestras para '{current_letter}'")
                        
                        # Avanzar automáticamente a la siguiente letra
                        if self.current_letter_idx < len(self.asl_alphabet) - 1:
                            self.current_letter_idx += 1
                            print(f"\nCambiando a letra: {self.asl_alphabet[self.current_letter_idx]}")
            
            # Dibujar información
            if results.multi_hand_landmarks:
                frame = self.draw_info(frame, results.multi_hand_landmarks[0])
            else:
                frame = self.draw_info(frame, None)
            
            # Mostrar frame
            cv2.imshow('Recolector de Datos ASL', frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.is_collecting = not self.is_collecting
                if self.is_collecting:
                    print(f"\nIniciando recolección para '{self.asl_alphabet[self.current_letter_idx]}'")
                else:
                    print("\nRecolección pausada")
            elif key == ord('c') and not self.is_collecting:
                # Capturar imagen de referencia
                if results.multi_hand_landmarks:
                    current_letter = self.asl_alphabet[self.current_letter_idx]
                    self.save_reference_image(frame, current_letter)
            elif key == ord('v'):
                # Mostrar/ocultar referencia
                self.show_reference = not self.show_reference
                if self.show_reference:
                    current_letter = self.asl_alphabet[self.current_letter_idx]
                    self.reference_image = self.load_reference_image(current_letter)
            elif key == ord('n') and not self.is_collecting:
                if self.current_letter_idx < len(self.asl_alphabet) - 1:
                    self.current_letter_idx += 1
                    self.samples_collected = 0
                    print(f"\nCambiando a letra: {self.asl_alphabet[self.current_letter_idx]}")
                    # Cargar referencia si está activado
                    if self.show_reference:
                        self.reference_image = self.load_reference_image(self.asl_alphabet[self.current_letter_idx])
            elif key == ord('p') and not self.is_collecting:
                if self.current_letter_idx > 0:
                    self.current_letter_idx -= 1
                    self.samples_collected = 0
                    print(f"\nCambiando a letra: {self.asl_alphabet[self.current_letter_idx]}")
                    # Cargar referencia si está activado
                    if self.show_reference:
                        self.reference_image = self.load_reference_image(self.asl_alphabet[self.current_letter_idx])
            elif key == ord('r') and not self.is_collecting:
                self.samples_collected = 0
                # Limpiar archivos de la letra actual
                current_letter = self.asl_alphabet[self.current_letter_idx]
                letter_dir = os.path.join(self.data_dir, current_letter)
                for file in os.listdir(letter_dir):
                    os.remove(os.path.join(letter_dir, file))
                print(f"\nReiniciando recolección para '{current_letter}'")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Generar resumen
        self.generate_summary()
    
    def generate_summary(self):
        """Genera un resumen del dataset recolectado"""
        summary = {
            "fecha_recoleccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "letras": {},
            "total_muestras": 0
        }
        
        for letter in self.asl_alphabet:
            letter_dir = os.path.join(self.data_dir, letter)
            if os.path.exists(letter_dir):
                num_samples = len([f for f in os.listdir(letter_dir) if f.endswith('.npy')])
                summary["letras"][letter] = num_samples
                summary["total_muestras"] += num_samples
        
        # Guardar resumen
        with open(os.path.join(self.data_dir, "dataset_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print("\n=== RESUMEN DE RECOLECCIÓN ===")
        print(f"Total de muestras: {summary['total_muestras']}")
        print("\nMuestras por letra:")
        for letter, count in summary["letras"].items():
            status = "✓" if count >= self.samples_per_letter else "✗"
            print(f"{status} {letter}: {count}/{self.samples_per_letter}")

if __name__ == "__main__":
    # Crear recolector
    collector = ASLDataCollector(data_dir="asl_dataset")
    
    # Configurar número de muestras por letra (ajustable)
    collector.samples_per_letter = 100
    
    # Iniciar recolección
    collector.collect_data()