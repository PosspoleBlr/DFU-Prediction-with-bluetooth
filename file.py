from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- Generate synthetic dataset and train model (software part complete) ---
np.random.seed(42)
n_samples = 1000

duration_diabetes = np.random.randint(0, 31, size=n_samples)
peripheral_neuropathy = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
pad = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
poor_glycemic_control = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
foot_deformities = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
previous_ulcers = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
pressure = np.random.normal(loc=50, scale=15, size=n_samples)

risk_prob = (
    0.02 * duration_diabetes +
    0.3 * peripheral_neuropathy +
    0.25 * pad +
    0.2 * poor_glycemic_control +
    0.15 * foot_deformities +
    0.4 * previous_ulcers +
    0.01 * (pressure - 40)
)
risk_prob = np.clip(risk_prob, 0, 1)
ulcer_risk = np.random.binomial(1, risk_prob)

data = pd.DataFrame({
    'duration_diabetes': duration_diabetes,
    'peripheral_neuropathy': peripheral_neuropathy,
    'pad': pad,
    'poor_glycemic_control': poor_glycemic_control,
    'foot_deformities': foot_deformities,
    'previous_ulcers': previous_ulcers,
    'pressure': pressure,
    'ulcer_risk': ulcer_risk
})

data.to_csv('synthetic_diabetic_foot_ulcer_data_new.csv', index=False)

X = data.drop(columns=['ulcer_risk'])
y = data['ulcer_risk']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- Placeholder for Bluetooth sensor integration (to be done by hardware team) ---
def get_pressure_from_bluetooth():
    # TODO: Replace this placeholder with actual Bluetooth sensor code
    # For now, returns a random float between 0 and 150
    return float(np.random.uniform(0, 150))

# --- HTML Template (UI/UX complete) ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Diabetic Foot Ulcer Risk Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    <style>
        body, html { min-height: 100vh; background: #fff !important; }
        .main-row { min-height: 100vh; display: flex; align-items: flex-start; justify-content: center; }
        .left-col { display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-start; padding-right: 2rem; margin-top: 0; width: 500px; min-width: 350px; }
        .logo-wrapper { position: fixed; top: 5px; left: 5px; z-index: 1000; }
        .logo-img { max-width: 100%; max-height: 90px; width: 220px; height: auto; margin-bottom: 1.2rem; margin-top: 2.2rem; display: block; object-fit: contain; }
        .main-heading { font-size: 2.1rem; font-weight: 700; color: #222; margin-bottom: 0.1rem; margin-top: 9.0rem; text-align: left; letter-spacing: 0.01em; }
        .description-text { font-size: 0.9rem; color: #555; margin-top: 2.0rem; margin-bottom: 1.5rem; max-width: 480px; line-height: 1.3; }
        .form-container { background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 0 15px rgba(0,0,0,0.07); max-width: 480px; min-width: 340px; margin-top: 6.5rem; }
        .outlined-input { border: 2px solid #0d6efd !important; border-radius: 0.375rem; }
        .outlined-radio { width: 1.5em; height: 1.5em; border: 2px solid #0d6efd; }
        .outlined-range { border: 2px solid #0d6efd; border-radius: 0.375rem; }
        .outlined-button { border-width: 2px !important; font-weight: 600; }
        output { display: inline-block; width: 3rem; text-align: center; font-weight: 600; margin-left: 0.5rem; }
        .container-fluid { padding-left: 210px; }
        @media (max-width: 900px) {
            .logo-wrapper { position: static; width: 100%; text-align: center; margin-bottom: 1rem; }
            .container-fluid { padding-left: 0; }
            .main-row { flex-direction: column; justify-content: flex-start; align-items: stretch; }
            .left-col { align-items: center; padding-right: 0; margin-bottom: 2rem; width: 100%; }
            .main-heading { text-align: center; }
            .description-text { text-align: center; }
            .form-container { margin-top: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="logo-wrapper">
        <img src="https://lp.posspole.com/wp-content/uploads/2025/02/Posspole-Logo-1320x262.png" alt="POSSPPOLE Logo" class="logo-img">
    </div>
    <div class="container-fluid">
        <div class="row main-row">
            <div class="col-lg-5 col-md-6 left-col">
                <h2 class="main-heading">Diabetic Foot Ulcer Risk Prediction</h2>
                <p class="description-text">
                    This app gives the risk of developing a diabetic foot ulcer for diabetic patients based on factors such as the duration of diabetes, peripheral neuropathy, peripheral arterial disease, poor glycemic control, foot deformities, previous ulcer history and foot pressure sensor value.
                </p>
            </div>
            <div class="col-lg-5 col-md-7 d-flex justify-content-center">
                <div class="form-container">
                    <form method="POST" action="/">
                        <div class="mb-3">
                            <label for="duration_diabetes" class="form-label">Duration of Diabetes (years)</label>
                            <input type="range" class="form-range outlined-range" id="duration_diabetes" name="duration_diabetes" min="0" max="100" value="{{ request.form.duration_diabetes or 5 }}" oninput="this.nextElementSibling.value = this.value" />
                            <output>{{ request.form.duration_diabetes or 5 }}</output>
                        </div>
                        {% for field, label in [
                            ('peripheral_neuropathy', 'Peripheral Neuropathy'),
                            ('pad', 'Peripheral Arterial Disease (PAD)'),
                            ('poor_glycemic_control', 'Poor Glycemic Control'),
                            ('foot_deformities', 'Foot Deformities'),
                            ('previous_ulcers', 'Previous History of Ulcers')
                        ] %}
                        <div class="mb-3">
                            <label class="form-label">{{ label }}</label><br />
                            <div class="form-check form-check-inline">
                                <input class="form-check-input outlined-radio" type="radio" name="{{ field }}" id="{{ field }}_no" value="0" {% if (request.form.get(field) or '0') == '0' %}checked{% endif %} />
                                <label class="form-check-label" for="{{ field }}_no">No</label>
                            </div>
                            <div class="form-check form-check-inline">
                                <input class="form-check-input outlined-radio" type="radio" name="{{ field }}" id="{{ field }}_yes" value="1" {% if request.form.get(field) == '1' %}checked{% endif %} />
                                <label class="form-check-label" for="{{ field }}_yes">Yes</label>
                            </div>
                        </div>
                        {% endfor %}
                        <div class="mb-3">
                            <label class="form-label">Foot Pressure Sensor Value (from Bluetooth)</label>
                            <input type="number" class="form-control outlined-input" id="pressure" name="pressure" value="{{ pressure_value }}" readonly />
                        </div>
                        <button type="submit" class="btn btn-outline-primary w-100 outlined-button">Predict Ulcer Risk</button>
                    </form>
                    {% if prediction %}
                    <div class="alert alert-info mt-4" role="alert">
                        <h4 class="alert-heading">Prediction Result</h4>
                        <p><strong>Risk Level:</strong> {{ prediction }}</p>
                        <p><strong>Risk Probability:</strong> {{ probability }}</p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    # Get pressure value from Bluetooth (placeholder for now)
    pressure_value = get_pressure_from_bluetooth()

    if request.method == 'POST':
        try:
            duration_diabetes = int(request.form['duration_diabetes'])
            peripheral_neuropathy = int(request.form['peripheral_neuropathy'])
            pad = int(request.form['pad'])
            poor_glycemic_control = int(request.form['poor_glycemic_control'])
            foot_deformities = int(request.form['foot_deformities'])
            previous_ulcers = int(request.form['previous_ulcers'])
            pressure = float(pressure_value)  # Use Bluetooth value

            input_df = pd.DataFrame({
                'duration_diabetes': [duration_diabetes],
                'peripheral_neuropathy': [peripheral_neuropathy],
                'pad': [pad],
                'poor_glycemic_control': [poor_glycemic_control],
                'foot_deformities': [foot_deformities],
                'previous_ulcers': [previous_ulcers],
                'pressure': [pressure]
            })

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0, 1]

            prediction = "High Risk" if pred == 1 else "Low Risk"
            probability = f"{proba:.2f}"
        except Exception:
            prediction = None
            probability = None

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        probability=probability,
        request=request,
        pressure_value=pressure_value
    )

if __name__ == '__main__':
    app.run(debug=True)
