import os
import tensorflow as tf
import pickle
import json

def main():
    model_dir = "asl_models"
    
    # 1. Buscar el modelo .h5 más reciente
    try:
        if not os.path.exists(model_dir):
            print(f"❌ No se encontró el directorio {model_dir}")
            return
            
        model_files = [f for f in os.listdir(model_dir) 
                      if f.endswith('.h5') and ('robust' in f or 'asl_model' in f)]
        if not model_files:
            print(f"❌ No se encontró ningún modelo .h5 en {model_dir}/")
            return
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)
        
        print(f"Cargando modelo de Keras: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # 2. Convertir a TFLite
        print("Convertiendo modelo a formato TensorFlow Lite (TFLite)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(model_dir, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ Modelo TFLite guardado en: {tflite_path}")
        
        # 3. Convertir label_encoder.pkl a labels.json
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            print(f"Cargando codificador de etiquetas: {encoder_path}")
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            
            classes_list = list(label_encoder.classes_)
            json_path = os.path.join(model_dir, 'labels.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(classes_list, f, ensure_ascii=False, indent=4)
            print(f"✓ Etiquetas exportadas en JSON a: {json_path}")
            print(f"  Clases: {classes_list}")
        else:
            print("⚠️ No se encontró label_encoder.pkl para exportar las etiquetas.")
            
        print("\n🎉 ¡Exportación completada con éxito! Ya puedes usar estos dos archivos en tu aplicación móvil.")
        
    except Exception as e:
        print(f"❌ Error durante la exportación: {e}")

if __name__ == "__main__":
    main()
