from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/prediction", methods=["POST"])
def prediction():

    age = float(request.form["age"])
    sex = int(request.form["sex"])
    cp = int(request.form["cp"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = int(request.form["fbs"])
    restecg = int(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    ca = int(request.form["ca"])
    thal = int(request.form["thal"])

    data = pd.DataFrame([[

        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal

    ]],

    columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][prediction] * 100

    if prediction == 1:
        result = "High Risk of Heart Disease"
        color = "red"

        recommendation = [
            "Consult a Cardiologist",
            "Maintain Healthy Diet",
            "Exercise Regularly",
            "Reduce Cholesterol Intake"
        ]

    else:

        result = "Low Risk of Heart Disease"
        color = "green"

        recommendation = [
            "Maintain Healthy Lifestyle",
            "Regular Health Checkup",
            "Balanced Diet",
            "Regular Exercise"
        ]

    return render_template(

        "result.html",

        prediction=result,
        probability=round(probability,2),
        color=color,
        recommendation=recommendation

    )


@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")


@app.route("/about")
def about():

    return render_template("about.html")


if __name__ == "__main__":

    app.run(debug=True)
