import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import json
from collections import deque
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk
import re
import pyttsx3

# Configurar apariencia de CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ProfessionalASLPredictor:
    def __init__(self, model_dir="asl_models"):
        self.model_dir = model_dir
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Cargar modelo
        self.model = None
        self.label_encoder = None
        self.load_model()
        
        # Variables de predicción
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.75
        self.last_prediction = ""
        self.last_prediction_time = 0
        self.cooldown_time = 0.4
        
        # Variables de borrado inteligente
        self.delete_start_time = 0
        self.delete_duration = 0
        self.is_deleting = False
        self.delete_mode = "char"  # char, word, all
        self.last_delete_action_time = 0
        self.delete_cooldown = 0.5
        
        # Variables de texto
        self.current_text = ""
        self.word_buffer = ""
        
        # GUI
        self.root = None
        self.text_queue = queue.Queue()
        self.is_running = False
        self.cap = None
        
        # Estadísticas
        self.prediction_count = {}
        self.start_time = time.time()
        self.words_per_minute = 0
        self.session_words = 0
        
        # Animaciones
        self.animation_queue = queue.Queue()
        
        # Hilo de voz persistente (para evitar problemas de garbage collection y ReferenceError en espeak)
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()
        
    def speech_worker(self):
        """Hilo de fondo persistente para reproducir voz de manera segura"""
        try:
            engine = pyttsx3.init()
        except Exception:
            engine = None
            
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            if engine and text.strip():
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            self.speech_queue.task_done()

    def load_model(self):
        """Carga el modelo entrenado"""
        try:
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.endswith('.h5') and ('robust' in f or 'asl_model' in f)]
            if not model_files:
                raise FileNotFoundError("No se encontró ningún modelo entrenado")
            
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(self.model_dir, latest_model)
            
            print(f"Cargando modelo: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✓ Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            raise
    
    def extract_landmarks(self, hand_landmarks):
        """Extrae y normaliza los landmarks"""
        landmarks = []
        
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z
        
        for landmark in hand_landmarks.landmark:
            x = landmark.x - wrist_x
            y = landmark.y - wrist_y
            z = landmark.z - wrist_z
            landmarks.extend([x, y, z])
        
        return np.array(landmarks)
    
    def predict_gesture(self, landmarks):
        """Predice el gesto"""
        landmarks = landmarks.reshape(1, -1)
        
        if len(self.model.input_shape) == 3:
            landmarks = landmarks.reshape(1, 21, 3)
        
        # Optimización crítica: Evitar el overhead de model.predict en tiempo real usando llamada directa
        landmarks_tensor = tf.convert_to_tensor(landmarks, dtype=tf.float32)
        predictions = self.model(landmarks_tensor, training=False).numpy()
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_letter = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_letter, confidence
    
    def create_modern_gui(self):
        """Crea una interfaz moderna y profesional"""
        self.root = ctk.CTk()
        self.root.title("ASL Translator Pro")
        self.root.geometry("1400x800")
        
        # Frame principal con gradiente
        self.main_container = ctk.CTkFrame(self.root, corner_radius=0)
        self.main_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Header elegante
        self.create_header()
        
        # Contenedor principal con 3 columnas
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Columna izquierda - Panel de control
        self.create_control_panel()
        
        # Columna central - Área de texto
        self.create_text_area()
        
        # Columna derecha - Estadísticas y visualización
        self.create_stats_panel()
        
        # Footer con información
        self.create_footer()
        
        # Iniciar actualizaciones
        self.update_gui()
        self.update_animations()
        
    def create_header(self):
        """Crea un header moderno"""
        header_frame = ctk.CTkFrame(self.main_container, height=80, corner_radius=0)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Logo y título
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left", padx=30, pady=15)
        
        # Emoji como logo
        logo_label = ctk.CTkLabel(title_frame, text="🤟", font=("Arial", 36))
        logo_label.pack(side="left", padx=(0, 15))
        
        # Título con estilo
        title_label = ctk.CTkLabel(title_frame, 
                                  text="ASL Translator",
                                  font=("SF Pro Display", 32, "bold"))
        title_label.pack(side="left")
        
        subtitle_label = ctk.CTkLabel(title_frame,
                                     text="Professional Edition",
                                     font=("SF Pro Display", 14),
                                     text_color="gray")
        subtitle_label.pack(side="left", padx=(10, 0))
        
        # Estado de conexión
        self.connection_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        self.connection_frame.pack(side="right", padx=30, pady=20)
        
        self.connection_dot = ctk.CTkLabel(self.connection_frame, 
                                          text="●", 
                                          font=("Arial", 16),
                                          text_color="red")
        self.connection_dot.pack(side="left", padx=(0, 5))
        
        self.connection_label = ctk.CTkLabel(self.connection_frame,
                                           text="Desconectado",
                                           font=("SF Pro Display", 14))
        self.connection_label.pack(side="left")
    
    def create_control_panel(self):
        """Panel de control izquierdo"""
        control_frame = ctk.CTkFrame(self.content_frame, width=300)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        control_frame.grid_propagate(False)
        
        # Título del panel
        panel_title = ctk.CTkLabel(control_frame, 
                                  text="Control Panel",
                                  font=("SF Pro Display", 20, "bold"))
        panel_title.pack(pady=(20, 30))
        
        # Botón principal de cámara
        self.camera_btn = ctk.CTkButton(control_frame,
                                       text="🎥  Iniciar Cámara",
                                       command=self.toggle_camera,
                                       height=50,
                                       font=("SF Pro Display", 16, "bold"),
                                       corner_radius=25)
        self.camera_btn.pack(pady=10, padx=20, fill="x")
        
        # Separador
        separator1 = ctk.CTkFrame(control_frame, height=2, fg_color="gray30")
        separator1.pack(fill="x", padx=20, pady=20)
        
        # Controles de edición
        edit_label = ctk.CTkLabel(control_frame, 
                                 text="Edición Rápida",
                                 font=("SF Pro Display", 16))
        edit_label.pack(pady=(0, 10))
        
        # Botones de edición en grid
        edit_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        edit_frame.pack(pady=10, padx=20, fill="x")
        
        self.space_btn = ctk.CTkButton(edit_frame,
                                      text="⎵ Espacio",
                                      command=self.add_space,
                                      width=120,
                                      height=40)
        self.space_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.delete_btn = ctk.CTkButton(edit_frame,
                                       text="⌫ Borrar",
                                       command=self.delete_last_word,
                                       width=120,
                                       height=40)
        self.delete_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.clear_btn = ctk.CTkButton(edit_frame,
                                      text="🗑️ Limpiar",
                                      command=self.clear_text,
                                      width=120,
                                      height=40,
                                      fg_color="red",
                                      hover_color="darkred")
        self.clear_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.undo_btn = ctk.CTkButton(edit_frame,
                                     text="↶ Deshacer",
                                     command=self.undo_last,
                                     width=120,
                                     height=40)
        self.undo_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Separador
        separator2 = ctk.CTkFrame(control_frame, height=2, fg_color="gray30")
        separator2.pack(fill="x", padx=20, pady=20)
        
        # Configuración
        config_label = ctk.CTkLabel(control_frame, 
                                   text="Configuración",
                                   font=("SF Pro Display", 16))
        config_label.pack(pady=(0, 10))
        
        # Slider de sensibilidad
        sens_label = ctk.CTkLabel(control_frame, 
                                 text="Sensibilidad",
                                 font=("SF Pro Display", 12))
        sens_label.pack()
        
        self.sensitivity_slider = ctk.CTkSlider(control_frame,
                                              from_=0.5,
                                              to=0.95,
                                              number_of_steps=9,
                                              command=self.update_sensitivity)
        self.sensitivity_slider.set(self.confidence_threshold)
        self.sensitivity_slider.pack(padx=20, pady=5, fill="x")
        
        self.sens_value_label = ctk.CTkLabel(control_frame,
                                           text=f"{self.confidence_threshold:.0%}",
                                           font=("SF Pro Display", 12))
        self.sens_value_label.pack()
        
        # Velocidad de detección
        speed_label = ctk.CTkLabel(control_frame, 
                                  text="Velocidad de Detección",
                                  font=("SF Pro Display", 12))
        speed_label.pack(pady=(10, 0))
        
        self.speed_slider = ctk.CTkSlider(control_frame,
                                         from_=0.2,
                                         to=1.0,
                                         number_of_steps=8,
                                         command=self.update_speed)
        self.speed_slider.set(self.cooldown_time)
        self.speed_slider.pack(padx=20, pady=5, fill="x")
        
        self.speed_value_label = ctk.CTkLabel(control_frame,
                                            text=f"{self.cooldown_time:.1f}s",
                                            font=("SF Pro Display", 12))
        self.speed_value_label.pack()
    
    def create_text_area(self):
        """Área central de texto"""
        text_frame = ctk.CTkFrame(self.content_frame)
        text_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        
        # Título del área
        text_title = ctk.CTkLabel(text_frame,
                                 text="Traducción en Tiempo Real",
                                 font=("SF Pro Display", 20, "bold"))
        text_title.pack(pady=(20, 10))
        
        # Área de texto principal con estilo moderno
        self.text_widget = ctk.CTkTextbox(text_frame,
                                         font=("SF Pro Text", 18),
                                         height=400,
                                         corner_radius=15)
        self.text_widget.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Frame de acciones
        action_frame = ctk.CTkFrame(text_frame, fg_color="transparent")
        action_frame.pack(pady=10, fill="x", padx=20)
        
        self.copy_btn = ctk.CTkButton(action_frame,
                                     text="📋 Copiar",
                                     command=self.copy_text,
                                     width=150,
                                     height=40)
        self.copy_btn.pack(side="left", padx=5)
        
        self.save_btn = ctk.CTkButton(action_frame,
                                     text="💾 Guardar",
                                     command=self.save_text,
                                     width=150,
                                     height=40)
        self.save_btn.pack(side="left", padx=5)
        
        self.export_btn = ctk.CTkButton(action_frame,
                                       text="📤 Exportar",
                                       command=self.export_text,
                                       width=150,
                                       height=40)
        self.export_btn.pack(side="left", padx=5)
        
        # Indicador de modo de borrado
        self.delete_indicator = ctk.CTkLabel(text_frame,
                                           text="",
                                           font=("SF Pro Display", 14, "bold"),
                                           text_color="red")
        self.delete_indicator.pack(pady=5)
    
    def create_stats_panel(self):
        """Panel de estadísticas derecho"""
        stats_frame = ctk.CTkFrame(self.content_frame, width=350)
        stats_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        stats_frame.grid_propagate(False)
        
        # Título
        stats_title = ctk.CTkLabel(stats_frame,
                                  text="Estadísticas en Vivo",
                                  font=("SF Pro Display", 20, "bold"))
        stats_title.pack(pady=(20, 20))
        
        # Frame de detección actual
        detection_frame = ctk.CTkFrame(stats_frame)
        detection_frame.pack(pady=10, padx=20, fill="x")
        
        self.current_letter_display = ctk.CTkLabel(detection_frame,
                                                 text="-",
                                                 font=("SF Pro Display", 72, "bold"))
        self.current_letter_display.pack(pady=20)
        
        self.confidence_bar = ctk.CTkProgressBar(detection_frame)
        self.confidence_bar.pack(padx=20, pady=10, fill="x")
        self.confidence_bar.set(0)
        
        self.confidence_label = ctk.CTkLabel(detection_frame,
                                           text="Confianza: 0%",
                                           font=("SF Pro Display", 14))
        self.confidence_label.pack(pady=(0, 10))
        
        # Separador
        separator = ctk.CTkFrame(stats_frame, height=2, fg_color="gray30")
        separator.pack(fill="x", padx=20, pady=10)
        
        # Métricas de sesión
        metrics_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        metrics_frame.pack(pady=10, padx=20, fill="x")
        
        # WPM
        self.wpm_frame = self.create_metric_card(metrics_frame, "⚡", "PPM", "0")
        self.wpm_frame.grid(row=0, column=0, padx=5, pady=5)
        
        # Palabras totales
        self.words_frame = self.create_metric_card(metrics_frame, "📝", "Palabras", "0")
        self.words_frame.grid(row=0, column=1, padx=5, pady=5)
        
        # Tiempo de sesión
        self.time_frame = self.create_metric_card(metrics_frame, "⏱️", "Tiempo", "0:00")
        self.time_frame.grid(row=1, column=0, padx=5, pady=5)
        
        # Precisión
        self.accuracy_frame = self.create_metric_card(metrics_frame, "🎯", "Precisión", "-%")
        self.accuracy_frame.grid(row=1, column=1, padx=5, pady=5)
        
        # Gráfico de letras más usadas
        graph_label = ctk.CTkLabel(stats_frame,
                                  text="Letras Más Detectadas",
                                  font=("SF Pro Display", 16))
        graph_label.pack(pady=(20, 10))
        
        self.letter_graph_frame = ctk.CTkFrame(stats_frame)
        self.letter_graph_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
    def create_metric_card(self, parent, icon, title, value):
        """Crea una tarjeta de métrica"""
        card = ctk.CTkFrame(parent, width=150, height=80)
        card.grid_propagate(False)
        
        icon_label = ctk.CTkLabel(card, text=icon, font=("Arial", 24))
        icon_label.pack(pady=(10, 0))
        
        value_label = ctk.CTkLabel(card, text=value, font=("SF Pro Display", 20, "bold"))
        value_label.pack()
        card.value_label = value_label  # Guardar referencia
        
        title_label = ctk.CTkLabel(card, text=title, font=("SF Pro Display", 12), 
                                  text_color="gray")
        title_label.pack()
        
        return card
    
    def create_footer(self):
        """Crea el footer"""
        footer_frame = ctk.CTkFrame(self.main_container, height=40, corner_radius=0)
        footer_frame.pack(side="bottom", fill="x")
        footer_frame.pack_propagate(False)
        
        # Tips que rotan
        self.tip_label = ctk.CTkLabel(footer_frame,
                                     text="💡 Tip: Mantén la mano estable para mejor detección",
                                     font=("SF Pro Display", 12))
        self.tip_label.pack(side="left", padx=20, pady=10)
        
        # Versión
        version_label = ctk.CTkLabel(footer_frame,
                                    text="v2.0 Professional",
                                    font=("SF Pro Display", 10),
                                    text_color="gray")
        version_label.pack(side="right", padx=20, pady=10)
    
    def toggle_camera(self):
        """Inicia o detiene la cámara"""
        if not self.is_running:
            self.is_running = True
            self.camera_btn.configure(text="⏹️  Detener Cámara")
            self.connection_dot.configure(text_color="green")
            self.connection_label.configure(text="Conectado")
            self.animation_queue.put(('camera_start', None))
            
            camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            camera_thread.start()
        else:
            self.is_running = False
            self.camera_btn.configure(text="🎥  Iniciar Cámara")
            self.connection_dot.configure(text_color="red")
            self.connection_label.configure(text="Desconectado")
            self.animation_queue.put(('camera_stop', None))
    
    def camera_loop(self):
        """Loop principal de la cámara con detección de borrado inteligente"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                landmarks = self.extract_landmarks(hand_landmarks)
                predicted_letter, confidence = self.predict_gesture(landmarks)
                
                # Actualizar UI con información
                self.text_queue.put(('info', {
                    'letter': predicted_letter,
                    'confidence': confidence
                }))
                
                if confidence > self.confidence_threshold:
                    self.prediction_buffer.append(predicted_letter)
                    
                    if len(self.prediction_buffer) >= 8:
                        most_common = max(set(self.prediction_buffer), 
                                        key=self.prediction_buffer.count)
                        consistency = self.prediction_buffer.count(most_common) / len(self.prediction_buffer)
                        
                        if consistency > 0.7:
                            current_time = time.time()
                            
                            # Manejo especial para DELETE
                            if most_common == 'DELETE':
                                self.handle_delete_gesture(current_time)
                            else:
                                # Resetear estado de borrado si no es DELETE
                                if self.is_deleting:
                                    self.is_deleting = False
                                    self.delete_start_time = 0
                                    self.text_queue.put(('delete_mode', 'none'))
                                
                                # Procesar otras letras normalmente
                                if (most_common != self.last_prediction and 
                                    current_time - self.last_prediction_time > self.cooldown_time):
                                    
                                    if most_common == 'SPACE':
                                        self.text_queue.put(('add', ' '))
                                        print(self.word_buffer)
                                        # Hilo de voz persistente (evita bloqueos y ReferenceError)
                                        self.speech_queue.put(self.word_buffer)
                                        self.word_buffer = ""

                                    else:
                                        self.text_queue.put(('add', most_common))
                                        self.word_buffer += most_common
                                    
                                    self.last_prediction = most_common
                                    self.last_prediction_time = current_time
                                    self.update_stats(most_common)
                
                # Mostrar información en el frame
                self.draw_camera_overlay(frame, predicted_letter, confidence)
            
            cv2.imshow('ASL Detector Pro', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def handle_delete_gesture(self, current_time):
        """Maneja el gesto de borrado con diferentes modos"""
        if not self.is_deleting:
            # Iniciar modo de borrado
            self.is_deleting = True
            self.delete_start_time = current_time
            self.delete_duration = 0
            self.text_queue.put(('delete_mode', 'char'))
            
            # Borrar un carácter inmediatamente
            if current_time - self.last_delete_action_time > self.delete_cooldown:
                self.text_queue.put(('delete', 'char'))
                self.last_delete_action_time = current_time
        else:
            # Continuar en modo de borrado
            self.delete_duration = current_time - self.delete_start_time
            
            if self.delete_duration < 2:
                # Modo: borrar caracteres (0-2 segundos)
                if self.delete_mode != "char":
                    self.delete_mode = "char"
                    self.text_queue.put(('delete_mode', 'char'))
                
                # Borrar caracteres continuamente
                if current_time - self.last_delete_action_time > 0.1:  # Borrar rápido
                    self.text_queue.put(('delete', 'char'))
                    self.last_delete_action_time = current_time
                    
            elif self.delete_duration < 4:
                # Modo: borrar palabras (2-4 segundos)
                if self.delete_mode != "word":
                    self.delete_mode = "word"
                    self.text_queue.put(('delete_mode', 'word'))
                    # Borrar primera palabra
                    self.text_queue.put(('delete', 'word'))
                    self.last_delete_action_time = current_time
                elif current_time - self.last_delete_action_time > 0.5:
                    # Borrar palabras cada 0.5 segundos
                    self.text_queue.put(('delete', 'word'))
                    self.last_delete_action_time = current_time
                    
            else:
                # Modo: borrar todo (4+ segundos)
                if self.delete_mode != "all":
                    self.delete_mode = "all"
                    self.text_queue.put(('delete_mode', 'all'))
                    self.text_queue.put(('delete', 'all'))
                    self.is_deleting = False  # Resetear después de borrar todo
    
    def draw_camera_overlay(self, frame, letter, confidence):
        """Dibuja información elegante en el frame de la cámara"""
        h, w = frame.shape[:2]
        
        # Panel superior con transparencia
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Letra detectada
        cv2.putText(frame, f"{letter if letter else '-'}", 
                   (20, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        
        # Barra de confianza
        bar_width = int((w - 200) * confidence)
        cv2.rectangle(frame, (180, 40), (w - 20, 60), (50, 50, 50), -1)
        cv2.rectangle(frame, (180, 40), (180 + bar_width, 60), (0, 255, 0), -1)
        cv2.putText(frame, f"{confidence:.0%}", 
                   (180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Indicador de modo de borrado
        if self.is_deleting:
            delete_text = f"BORRANDO: {self.delete_mode.upper()}"
            color = (0, 0, 255) if self.delete_mode == "all" else (0, 165, 255)
            cv2.putText(frame, delete_text,
                       (w//2 - 100, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2)
            
            # Barra de progreso para modo de borrado
            progress = min(self.delete_duration / 4, 1.0)
            bar_length = int(200 * progress)
            cv2.rectangle(frame, (w//2 - 100, h - 20), (w//2 + 100, h - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (w//2 - 100, h - 20), (w//2 - 100 + bar_length, h - 10), color, -1)
    
    def update_stats(self, letter):
        """Actualiza las estadísticas"""
        if letter not in self.prediction_count:
            self.prediction_count[letter] = 0
        self.prediction_count[letter] += 1
        
        # Calcular palabras por minuto
        if letter == 'SPACE':
            self.session_words += 1
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes > 0:
            self.words_per_minute = self.session_words / elapsed_minutes
    
    def update_gui(self):
        """Actualiza la GUI con los datos de la cola"""
        try:
            while True:
                action, data = self.text_queue.get_nowait()
                
                if action == 'add':
                    self.text_widget.insert("end", data)
                    self.text_widget.see("end")
                    
                elif action == 'delete':
                    content = self.text_widget.get("1.0", "end-1c")
                    if data == 'char' and len(content) > 0:
                        self.text_widget.delete("end-2c")
                    elif data == 'word' and len(content) > 0:
                        # Borrar última palabra
                        words = content.rstrip().split()
                        if words:
                            words.pop()
                            new_text = ' '.join(words)
                            if new_text:
                                new_text += ' '
                            self.text_widget.delete("1.0", "end")
                            self.text_widget.insert("1.0", new_text)
                    elif data == 'all':
                        self.text_widget.delete("1.0", "end")
                        self.animation_queue.put(('clear_animation', None))
                
                elif action == 'delete_mode':
                    if data == 'none':
                        self.delete_indicator.configure(text="")
                    elif data == 'char':
                        self.delete_indicator.configure(text="🔴 Borrando caracteres...")
                    elif data == 'word':
                        self.delete_indicator.configure(text="🟠 Borrando palabras...")
                    elif data == 'all':
                        self.delete_indicator.configure(text="🔴 BORRANDO TODO...")
                
                elif action == 'info':
                    # Actualizar letra actual
                    self.current_letter_display.configure(text=data['letter'])
                    
                    # Actualizar barra de confianza
                    self.confidence_bar.set(data['confidence'])
                    self.confidence_label.configure(text=f"Confianza: {data['confidence']:.0%}")
                    
                    # Animar la detección
                    if data['confidence'] > self.confidence_threshold:
                        self.animation_queue.put(('detection', data['letter']))
                
        except queue.Empty:
            pass
        
        # Actualizar estadísticas
        self.update_statistics_display()
        
        self.root.after(50, self.update_gui)
    
    def update_statistics_display(self):
        """Actualiza las estadísticas mostradas"""
        # Tiempo transcurrido
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        self.time_frame.value_label.configure(text=f"{minutes}:{seconds:02d}")
        
        # Palabras por minuto
        self.wpm_frame.value_label.configure(text=f"{int(self.words_per_minute)}")
        
        # Total de palabras
        self.words_frame.value_label.configure(text=str(self.session_words))
        
        # Precisión (simulada basada en consistencia)
        if len(self.prediction_buffer) > 0:
            consistency = len(set(self.prediction_buffer)) / len(self.prediction_buffer)
            accuracy = int((1 - consistency) * 100)
            self.accuracy_frame.value_label.configure(text=f"{accuracy}%")
        
        # Actualizar gráfico de letras
        self.update_letter_graph()
    
    def update_letter_graph(self):
        """Actualiza el gráfico de letras más usadas"""
        # Limpiar frame anterior
        for widget in self.letter_graph_frame.winfo_children():
            widget.destroy()
        
        if not self.prediction_count:
            return
        
        # Top 5 letras
        sorted_letters = sorted(self.prediction_count.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        
        if sorted_letters:
            max_count = sorted_letters[0][1]
            
            for i, (letter, count) in enumerate(sorted_letters):
                # Frame para cada barra
                bar_frame = ctk.CTkFrame(self.letter_graph_frame, fg_color="transparent")
                bar_frame.pack(fill="x", pady=2, padx=10)
                
                # Letra
                letter_label = ctk.CTkLabel(bar_frame, text=letter, width=40,
                                          font=("SF Pro Display", 14, "bold"))
                letter_label.pack(side="left")
                
                # Barra de progreso
                progress = ctk.CTkProgressBar(bar_frame, height=20)
                progress.pack(side="left", fill="x", expand=True, padx=10)
                progress.set(count / max_count if max_count > 0 else 0)
                
                # Contador
                count_label = ctk.CTkLabel(bar_frame, text=str(count), width=40,
                                         font=("SF Pro Display", 12))
                count_label.pack(side="left")
    
    def update_animations(self):
        """Maneja las animaciones de la UI"""
        try:
            while True:
                animation, data = self.animation_queue.get_nowait()
                
                if animation == 'detection':
                    # Animar la detección de letra
                    self.current_letter_display.configure(text_color="green")
                    self.root.after(200, lambda: self.current_letter_display.configure(
                        text_color="white"))
                
                elif animation == 'camera_start':
                    # Animación de inicio de cámara
                    self.camera_btn.configure(fg_color="green")
                    self.root.after(500, lambda: self.camera_btn.configure(
                        fg_color=("#3B8ED0", "#1F6AA5")))
                
                elif animation == 'clear_animation':
                    # Animación al limpiar todo
                    self.text_widget.configure(fg_color="red")
                    self.root.after(200, lambda: self.text_widget.configure(
                        fg_color=("#212121", "#212121")))
                
        except queue.Empty:
            pass
        
        # Rotar tips
        self.rotate_tips()
        
        self.root.after(100, self.update_animations)
    
    def rotate_tips(self):
        """Rota los tips del footer"""
        tips = [
            "💡 Tip: Mantén la mano estable para mejor detección",
            "💡 Tip: Usa buena iluminación para resultados óptimos",
            "💡 Tip: Mantén presionado DELETE para borrar palabras",
            "💡 Tip: La práctica mejora la velocidad de traducción",
            "💡 Tip: Ajusta la sensibilidad si hay muchos errores"
        ]
        
        current_tip = self.tip_label.cget("text")
        current_index = next((i for i, tip in enumerate(tips) if tip == current_tip), -1)
        next_index = (current_index + 1) % len(tips)
        
        if hasattr(self, '_tip_counter'):
            self._tip_counter += 1
            if self._tip_counter >= 50:  # Cambiar cada 5 segundos
                self.tip_label.configure(text=tips[next_index])
                self._tip_counter = 0
        else:
            self._tip_counter = 0
    
    def update_sensitivity(self, value):
        """Actualiza la sensibilidad de detección"""
        self.confidence_threshold = float(value)
        self.sens_value_label.configure(text=f"{self.confidence_threshold:.0%}")
    
    def update_speed(self, value):
        """Actualiza la velocidad de detección"""
        self.cooldown_time = float(value)
        self.speed_value_label.configure(text=f"{self.cooldown_time:.1f}s")
    
    def add_space(self):
        """Añade un espacio"""
        self.text_widget.insert("end", " ")
        self.word_buffer = ""
        self.session_words += 1
    
    def delete_last_word(self):
        """Borra la última palabra"""
        content = self.text_widget.get("1.0", "end-1c")
        words = content.rstrip().split()
        if words:
            words.pop()
            new_text = ' '.join(words)
            if new_text:
                new_text += ' '
            self.text_widget.delete("1.0", "end")
            self.text_widget.insert("1.0", new_text)
    
    def clear_text(self):
        """Limpia todo el texto con confirmación"""
        if self.text_widget.get("1.0", "end-1c").strip():
            # Crear diálogo personalizado
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Confirmar")
            dialog.geometry("400x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Centrar el diálogo
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Contenido
            ctk.CTkLabel(dialog, text="¿Estás seguro de borrar todo el texto?",
                        font=("SF Pro Display", 16)).pack(pady=30)
            
            button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            button_frame.pack(pady=20)
            
            ctk.CTkButton(button_frame, text="Sí, borrar", 
                         command=lambda: [self.text_widget.delete("1.0", "end"),
                                        self.animation_queue.put(('clear_animation', None)),
                                        dialog.destroy()],
                         fg_color="red", width=120).pack(side="left", padx=10)
            
            ctk.CTkButton(button_frame, text="Cancelar",
                         command=dialog.destroy, width=120).pack(side="left", padx=10)
    
    def undo_last(self):
        """Deshace la última acción (simulado)"""
        # Por simplicidad, borra el último carácter
        content = self.text_widget.get("1.0", "end-1c")
        if content:
            self.text_widget.delete("end-2c")
    
    def copy_text(self):
        """Copia el texto al portapapeles"""
        text = self.text_widget.get("1.0", "end-1c").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            
            # Mostrar notificación
            self.show_notification("✓ Texto copiado al portapapeles")
    
    def save_text(self):
        """Guarda el texto en un archivo"""
        text = self.text_widget.get("1.0", "end-1c").strip()
        if not text:
            self.show_notification("⚠️ No hay texto para guardar", "warning")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
            initialfile=f"asl_traduccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.show_notification(f"✓ Guardado exitosamente")
            except Exception as e:
                self.show_notification(f"❌ Error al guardar: {str(e)}", "error")
    
    def export_text(self):
        """Exporta el texto con formato especial"""
        text = self.text_widget.get("1.0", "end-1c").strip()
        if not text:
            self.show_notification("⚠️ No hay texto para exportar", "warning")
            return
        
        # Crear documento con metadatos
        export_content = f"""ASL TRANSLATOR PRO - DOCUMENTO EXPORTADO
{'='*50}

Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Duración de sesión: {int((time.time() - self.start_time) / 60)} minutos
Palabras totales: {self.session_words}
Palabras por minuto: {int(self.words_per_minute)}

TEXTO TRADUCIDO:
{'='*50}

{text}

{'='*50}
Generado por ASL Translator Professional Edition
"""
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Documento de texto", "*.txt"), 
                      ("Documento Word", "*.doc"),
                      ("PDF", "*.pdf")],
            initialfile=f"asl_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(export_content)
                self.show_notification("✓ Exportado exitosamente")
            except Exception as e:
                self.show_notification(f"❌ Error: {str(e)}", "error")
    
    def show_notification(self, message, type="success"):
        """Muestra una notificación temporal"""
        notification = ctk.CTkToplevel(self.root)
        notification.title("")
        notification.geometry("300x80")
        notification.overrideredirect(True)
        
        # Posicionar en la parte superior derecha
        notification.update_idletasks()
        x = self.root.winfo_x() + self.root.winfo_width() - 320
        y = self.root.winfo_y() + 100
        notification.geometry(f"+{x}+{y}")
        
        # Color según tipo
        colors = {
            "success": "#2ECC71",
            "warning": "#F39C12",
            "error": "#E74C3C"
        }
        
        frame = ctk.CTkFrame(notification, fg_color=colors.get(type, "#2ECC71"))
        frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(frame, text=message, font=("SF Pro Display", 14),
                    text_color="white").pack(expand=True)
        
        # Auto-cerrar después de 3 segundos
        notification.after(3000, notification.destroy)
    
    def run(self):
        """Inicia la aplicación"""
        try:
            self.create_modern_gui()
            self.root.mainloop()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    # Verificar dependencias
    try:
        import customtkinter
    except ImportError:
        print("❌ CustomTkinter no está instalado.")
        print("Instálalo con: pip install customtkinter")
        return
    
    # Verificar modelo
    if not os.path.exists("asl_models"):
        print("❌ No se encontró la carpeta de modelos.")
        print("Por favor, ejecuta primero el entrenamiento.")
        return
    
    # Crear y ejecutar
    predictor = ProfessionalASLPredictor()
    predictor.run()

if __name__ == "__main__":
    main()