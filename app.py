from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("HeartPredict.pkl", "rb"))
scaler = pickle.load(open("HeartScaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    probability = None
    risk_level = None
    risk_class = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"])
            ]

            # Scale input
            scaled = scaler.transform([features])

            # Predict probability
            proba = model.predict_proba(scaled)[0][1]
            probability = round(proba * 100, 2)

            if probability < 60:
                risk_level = "Low Risk ✅ You are likely safe."
                risk_class = "success"
            elif probability <= 75:
                risk_level = "Medium Risk ⚠️ Please consult a doctor."
                risk_class = "warning"
            else:
                risk_level = "High Risk 🚨 Immediate attention recommended!"
                risk_class = "danger"

        except Exception as e:
            risk_level = f"Error: {str(e)}"
            risk_class = "danger"

    return render_template("index.html",risk_class=risk_class, probability=probability, risk_level=risk_level)

if __name__ == "__main__":
    app.run(debug=True)
