from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from utils.preprocessing import preprocess_data
import config

app = Flask(__name__)
model = joblib.load(config.MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def frontend_predict():
    if request.method == 'POST':
        try:
            form_data = {
                "amount": float(request.form['amount']),
                "location": int(request.form['location']),
                "time": int(request.form['time']),
                "merchant_score": float(request.form['merchant_score']),
            }
            df = pd.DataFrame([form_data])
            processed = preprocess_data(df)
            prediction = model.predict(processed)[0]
            return render_template('result.html', prediction=prediction)
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        input_data = pd.DataFrame(request.json)
        processed = preprocess_data(input_data)
        predictions = model.predict(processed)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
# config.py
MODEL_PATH = "model/pipeline.pkl"
DATA_PATH = "data/transactions.csv"
