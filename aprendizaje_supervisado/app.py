from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

@app.route("/")
def home():
    return "API Titanic funcionando"

@app.route("/predecir", methods=["POST"])
def predecir():

    data = request.json
    features = np.array(data["input"]).reshape(1, -1)

    pred = modelo.predict(features)

    return jsonify({
        "prediccion": int(pred[0])
    })

if __name__ == "__main__":
    app.run(debug=True)