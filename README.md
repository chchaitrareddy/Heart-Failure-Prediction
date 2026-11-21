# Heart Failure Prediction Using Classification Analysis Algorithms

A beginner-friendly Machine Learning web application built using **Python, Flask, Pandas, Scikit-learn, HTML, and CSS**. This application predicts 10-year risk of Coronary Heart Disease (CHD) based on clinical parameters from the Framingham Heart Study dataset.

---

## 📁 Project Structure

```text
HeartFailurePrediction/
├── app.py                  # Main Flask web application server & routing
├── train_model.py          # Offline ML model training script
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── model/
│   └── model.pkl           # Trained Random Forest Classifier pickle binary file
├── dataset/
│   └── heart.csv           # Framingham Heart Study dataset
├── templates/
│   ├── index.html          # Patient health assessment input form
│   ├── result.html         # Heart disease risk & preventive tips display page
│   └── about.html          # Project background & technology stack explanation
└── static/
    └── style.css           # Vanilla CSS stylesheet for web design
```

---

## ⚙️ Requirements & Dependencies

- Python 3.8+ (includes built-in `pickle` module)
- Flask (`3.0.0`)
- Pandas (`2.2.0`)
- NumPy (`1.26.3`)
- Scikit-Learn (`1.4.0`)

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Open terminal inside project directory and run:
```bash
pip install -r requirements.txt
```

### 2. Train Machine Learning Model
Train the Random Forest Classifier model on the Framingham dataset:
```bash
python train_model.py
```
This saves the trained model using Python's built-in `pickle` to `model/model.pkl`.

### 3. Launch Flask Web Application
Start local development server:
```bash
python app.py
```

### 4. Access Web Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 📊 ML Pipeline Overview

1. **Dataset Loading**: Reads `dataset/heart.csv`.
2. **Preprocessing**: Missing values are imputed using column medians.
3. **Train-Test Split**: Data divided into 80% Training and 20% Testing sets.
4. **Model Training**: `RandomForestClassifier(n_estimators=100, random_state=42)`.
5. **Serialization**: Model serialized using Python's built-in `pickle` and saved to `model/model.pkl`.

---

## 👥 Authors & Academic Usage
Designed for College Project Viva Examinations & Machine Learning Technical Placement Interviews.
