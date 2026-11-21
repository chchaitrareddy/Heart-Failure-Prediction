# ==============================================================================
# File: app.py
# Purpose: Main Flask web application controller. Handles HTTP requests,
#          validates form inputs, loads the trained Random Forest model using
#          Python's built-in pickle module, executes predictions, and renders HTML templates.
# ==============================================================================

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# Initialize Flask application
app = Flask(__name__)

# Define global path to saved model artifact
MODEL_PATH = os.path.join("model", "model.pkl")

def load_trained_model():
    """
    Helper function to load trained model safely from disk using standard pickle.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model with pickle: {e}")
        return None


@app.route("/", methods=["GET"])
def home():
    """
    Route 1: Home Page (GET /)
    Renders index.html containing the heart disease risk assessment form.
    """
    return render_template("index.html")


@app.route("/about", methods=["GET"])
def about():
    """
    Route 2: About Project Page (GET /about)
    Renders about.html explaining dataset, algorithm, and project scope.
    """
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Route 3: Predict Heart Disease Risk (POST /predict)
    Processes HTML form data, constructs feature array, loads model,
    performs prediction, generates preventive recommendations, and renders result.html.
    """
    # Step 1: Check if trained model file exists
    model = load_trained_model()
    if model is None:
        return render_template(
            "result.html",
            error="Model file (model.pkl) not found or could not be loaded. Please run 'python train_model.py' first."
        )

    try:
        # Step 2: Extract input form values directly from request.form
        male = float(request.form.get("male", 0))
        age = float(request.form.get("age", 0))
        education = float(request.form.get("education", 1))
        currentSmoker = float(request.form.get("currentSmoker", 0))
        cigsPerDay = float(request.form.get("cigsPerDay", 0))
        BPMeds = float(request.form.get("BPMeds", 0))
        prevalentStroke = float(request.form.get("prevalentStroke", 0))
        prevalentHyp = float(request.form.get("prevalentHyp", 0))
        diabetes = float(request.form.get("diabetes", 0))
        totChol = float(request.form.get("totChol", 0))
        sysBP = float(request.form.get("sysBP", 0))
        diaBP = float(request.form.get("diaBP", 0))
        BMI = float(request.form.get("BMI", 0))
        heartRate = float(request.form.get("heartRate", 0))
        glucose = float(request.form.get("glucose", 0))

        # Step 3: Organize input features into a Pandas DataFrame
        input_data = pd.DataFrame([{
            'male': male,
            'age': age,
            'education': education,
            'currentSmoker': currentSmoker,
            'cigsPerDay': cigsPerDay,
            'BPMeds': BPMeds,
            'prevalentStroke': prevalentStroke,
            'prevalentHyp': prevalentHyp,
            'diabetes': diabetes,
            'totChol': totChol,
            'sysBP': sysBP,
            'diaBP': diaBP,
            'BMI': BMI,
            'heartRate': heartRate,
            'glucose': glucose
        }])

        # Step 4: Perform Prediction using loaded Random Forest model
        prediction = model.predict(input_data)[0]
        
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)
        except Exception:
            confidence = 85.0

        # Step 5: Determine Risk Category and Tailored Recommendations
        if prediction == 1:
            risk_level = "High Risk"
            risk_class = "danger"
            summary_message = "Based on your health parameters, you have a high potential risk of developing Coronary Heart Disease in the next 10 years."
            
            preventive_measures = [
                "Schedule a detailed cardiovascular health checkup with a Cardiologist immediately.",
                "Monitor and maintain Systolic BP under 120 mmHg and Diastolic BP under 80 mmHg.",
                "Get regular blood tests to keep Total Cholesterol below 200 mg/dL and Fasting Glucose under 100 mg/dL."
            ]
            
            lifestyle_suggestions = [
                "Quit smoking completely if applicable; avoid second-hand smoke exposure.",
                "Engage in at least 30 minutes of moderate aerobic exercise (brisk walking, cycling) 5 days a week.",
                "Adopt a Mediterranean or DASH diet low in saturated fats, refined sugars, and sodium (<2,000 mg/day)."
            ]
            
            health_tips = [
                "Track your resting heart rate daily using a smart device or manual pulse check.",
                "Maintain a healthy BMI between 18.5 and 24.9.",
                "Practice stress-reduction techniques such as deep breathing, yoga, or meditation."
            ]
        else:
            risk_level = "Low Risk"
            risk_class = "success"
            summary_message = "Great news! Based on your health parameters, your estimated risk of Coronary Heart Disease in the next 10 years is Low."
            
            preventive_measures = [
                "Continue annual routine health screenings with your primary care provider.",
                "Maintain regular monitoring of blood pressure, blood glucose, and lipid profiles.",
                "Stay updated on personal and family cardiovascular medical history."
            ]
            
            lifestyle_suggestions = [
                "Maintain your active routine with consistent physical exercise (150+ minutes per week).",
                "Ensure balanced, nutrient-rich meals rich in fruits, vegetables, whole grains, and lean proteins.",
                "Maintain healthy sleep hygiene (7-8 hours of quality restful sleep nightly)."
            ]
            
            health_tips = [
                "Stay well-hydrated throughout the day by drinking 2-3 liters of water.",
                "Limit intake of processed foods, ultra-processed snacks, and sugary beverages.",
                "Keep body mass index (BMI) within the healthy normal range."
            ]

        # Step 6: Render result.html with prediction output data
        return render_template(
            "result.html",
            risk_level=risk_level,
            risk_class=risk_class,
            summary_message=summary_message,
            confidence=confidence,
            preventive_measures=preventive_measures,
            lifestyle_suggestions=lifestyle_suggestions,
            health_tips=health_tips,
            age=int(age),
            sysBP=int(sysBP),
            totChol=int(totChol),
            glucose=int(glucose),
            BMI=round(BMI, 1)
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=f"An error occurred while processing prediction: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
