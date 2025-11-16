import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, render_template, request

# 1. Initialize the Flask app
app = Flask(__name__)

# 2. Load all the trained pipeline models
print("Loading models...")
try:
    diabetes_model = joblib.load('model/diabetes_pipeline.pkl')
    heart_model = joblib.load('model/heart_disease_pipeline.pkl')
    insurance_model = joblib.load('model/insurance_pipeline.pkl')
    car_model = joblib.load('model/car_pipeline.pkl')
    house_model = joblib.load('model/house_price_pipeline.pkl')
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    print("Please make sure all .pkl files are in the main directory.")
    # In a real app, you might exit or handle this more gracefully
except Exception as e:
    print(f"An error occurred: {e}")

# 3. Define Webpage Routes

# --- Homepage ---
@app.route('/')
def home():
    # Renders the index.html file from the 'templates' folder
    return render_template('index.html')

# --- Diabetes Prediction ---
@app.route('/diabetes')
def diabetes_page():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        # 1. Get data from the form
        preg = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bp = int(request.form['BloodPressure'])
        skin = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        # 2. Create the feature DataFrame
        # Column names MUST match those used during training
        data = {
            'Pregnancies': [preg],
            'Glucose': [glucose],
            'BloodPressure': [bp],
            'SkinThickness': [skin],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        }
        features_df = pd.DataFrame(data)

        # 3. Make prediction
        prediction = diabetes_model.predict(features_df)
        result_text = "This person is DIABETIC" if prediction[0] == 1 else "This person is NOT DIABETIC"

        # 4. Return the result to the same page
        return render_template('diabetes.html', prediction_text=result_text)

# --- Heart Disease Prediction ---
@app.route('/heart')
def heart_page():
    return render_template('heart.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        # 1. Get data from form (ensure all are converted to numbers)
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # 2. Create DataFrame
        # Column names MUST match
        data = {
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
            'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
            'slope': [slope], 'ca': [ca], 'thal': [thal]
        }
        features_df = pd.DataFrame(data)
        
        # 3. Make prediction
        prediction = heart_model.predict(features_df)
        result_text = "This person HAS Heart Disease" if prediction[0] == 1 else "This person does NOT have Heart Disease"
        
        # 4. Return result
        return render_template('heart.html', prediction_text=result_text)

# --- Insurance Cost Prediction ---
@app.route('/insurance')
def insurance_page():
    return render_template('insurance.html')

@app.route('/predict_insurance', methods=['POST'])
def predict_insurance():
    if request.method == 'POST':
        # 1. Get data from form
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # 2. Create DataFrame
        data = {
            'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
            'smoker': [smoker], 'region': [region]
        }
        features_df = pd.DataFrame(data)

        # 3. Make prediction
        prediction = insurance_model.predict(features_df)
        result_text = f"Predicted Insurance Cost: ${prediction[0]:,.2f}"
        
        # 4. Return result
        return render_template('insurance.html', prediction_text=result_text)

# --- Car Price Prediction ---
@app.route('/car')
def car_page():
    return render_template('car.html')

@app.route('/predict_car', methods=['POST'])
def predict_car():
    if request.method == 'POST':
        # 1. Get data from form
        present_price = float(request.form['Present_Price'])
        kms_driven = int(request.form['Kms_Driven'])
        owner = int(request.form['Owner'])
        year = int(request.form['Year'])
        fuel_type = request.form['Fuel_Type']
        seller_type = request.form['Seller_Type']
        transmission = request.form['Transmission']
        
        # 2. Feature Engineering: Create 'Car_Age'
        current_year = datetime.now().year
        car_age = current_year - year

        # 3. Create DataFrame
        data = {
            'Present_Price': [present_price], 'Kms_Driven': [kms_driven],
            'Owner': [owner], 'Fuel_Type': [fuel_type],
            'Seller_Type': [seller_type], 'Transmission': [transmission],
            'Car_Age': [car_age]
        }
        features_df = pd.DataFrame(data)
        
        # 4. Make prediction
        prediction = car_model.predict(features_df)
        result_text = f"Predicted Resale Value: ₹{prediction[0]:,.2f} lakhs" # Assuming price is in lakhs
        
        # 5. Return result
        return render_template('car.html', prediction_text=result_text)

@app.route('/predict_house', methods=['POST'])
def predict_house():
    if request.method == 'POST':
        # 1. Get data from form
        med_inc = float(request.form['MedInc'])
        house_age = float(request.form['HouseAge'])
        avg_rooms = float(request.form['AveRooms'])
        avg_bedrms = float(request.form['AveBedrms'])
        population = float(request.form['Population'])
        avg_occup = float(request.form['AveOccup'])
        latitude = float(request.form['Latitude'])
        longitude = float(request.form['Longitude'])

        # 2. Create DataFrame (Column names from fetch_california_housing)
        data = {
            'MedInc': [med_inc], 'HouseAge': [house_age], 'AveRooms': [avg_rooms],
            'AveBedrms': [avg_bedrms], 'Population': [population],
            'AveOccup': [avg_occup], 'Latitude': [latitude], 'Longitude': [longitude]
        }
        features_df = pd.DataFrame(data)
        
        # 3. Make prediction
        prediction = house_model.predict(features_df)
        # Model predicts in $100,000s
        predicted_price = prediction[0] * 100000 
        result_text = f"Predicted House Value: ${predicted_price:,.2f}"
        
        # 4. Return result
        return render_template('house.html', prediction_text=result_text)


# 4. Run the app
if __name__ == '__main__':
    # debug=True allows you to see errors and auto-reloads the server on changes
    app.run(debug=True)