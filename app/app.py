from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS  # Importar a extensão
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas as rotas

# Carregar modelo
MODEL_PATH = os.getenv("MODEL_PATH", "/default/path/to/model/flower_model.h5")
print(f"O caminho do modelo é: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Classes (modifique de acordo com o dataset)
CLASS_NAMES = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Pong!"})

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Processar a imagem
    img = load_img(file, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fazer previsão
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    response = jsonify({"class": predicted_class, "confidence": float(confidence)})
    
    # Adicionar cabeçalhos personalizados
    response.headers.add("Access-Control-Allow-Origin", "*")  # Habilitar qualquer origem
    response.headers.add("X-Powered-By", "Flask with TensorFlow")
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=500)
