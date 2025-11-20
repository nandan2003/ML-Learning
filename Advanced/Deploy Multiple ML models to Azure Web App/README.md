# Multi-Model Machine Learning Web Application

This project is a web application designed to serve multiple machine learning models through a single, user-friendly Flask interface. It allows users to input data and receive real-time predictions for several different problems, including both regression and classification tasks.

The application is built with a Python and Flask backend, with pre-trained Scikit-learn models, and is deployed on Microsoft Azure.

> **⚠️ Note on Performance:**
> As this application is hosted on a **free-tier** Azure instance, the server spins down when inactive.
> * **Cold Start:** Please allow **up to 5 minutes** for the website to load initially while the server wakes up.
> * **Subsequent Usage:** Once loaded, the application responds quickly.

## Live Demo

You can access the live, deployed application here:
[https://multi-ml-models-nandanv76.azurewebsites.net/](https://multi-ml-models-nandanv76.azurewebsites.net/)

## Features

The application provides prediction services for four distinct machine learning models:

  * **Car Price Prediction:** (Regression) Predicts the selling price of a used car based on its features.
  * **Diabetes Prediction:** (Classification) Predicts the likelihood of an individual having diabetes based on diagnostic measures.
  * **Heart Disease Prediction:** (Classification) Predicts the likelihood of an individual having heart disease based on health metrics.
  * **Medical Insurance Cost Prediction:** (Regression) Predicts the estimated medical insurance cost for an individual based on their demographic and health data.

## Technology Stack

  * **Backend:** Python, Flask
  * **Machine Learning:** Scikit-learn
  * **Model Serialization:** Joblib
  * **Frontend:** HTML, CSS
  * **Deployment:** Microsoft Azure App Service

## Project Structure

The project is organized in a standard Flask application structure:

```
/
|-- model/
|   |-- car_pipeline.pkl
|   |-- diabetes_pipeline.pkl
|   |-- heart_disease_pipeline.pkl
|   |-- insurance_pipeline.pkl
|
|-- templates/
|   |-- car.html
|   |-- diabetes.html
|   |-- heart.html
|   |-- house.html
|   |-- index.html
|   |-- insurance.html
|
|-- static/
|   |-- css/
|       |-- style.css
|
|-- app.py 
|-- requirements.txt 
```

## Model Details

All models are built using Scikit-learn pipelines, which bundle preprocessing steps and the final estimator. This ensures that the same transformations applied during training are applied to user input during prediction.

### 1\. Car Price Prediction

  * **Task:** Regression
  * **Model:** `RandomForestRegressor`
  * **Preprocessing Pipeline:**
      * A `ColumnTransformer` is used to apply different preprocessing to numeric and categorical columns.
      * **Numeric Features** (`Present_Price`, `Kms_Driven`, `Owner`, `Car_Age`): Processed with `StandardScaler`. The `Car_Age` feature is engineered from the original `Year` column.
      * **Categorical Features** (`Fuel_Type`, `Seller_Type`, `Transmission`): Processed with `OneHotEncoder`.

### 2\. Diabetes Prediction

  * **Task:** Classification
  * **Model:** `RandomForestClassifier`
  * **Preprocessing Pipeline:**
      * **Imputation:** Uses `SimpleImputer` with a `median` strategy to fill missing values (which are represented as '0' in the source dataset for columns like `Glucose`, `BloodPressure`, `BMI`, etc.).
      * **Scaling:** All features are scaled using `StandardScaler`.

### 3\. Heart Disease Prediction

  * **Task:** Classification
  * **Model:** `LogisticRegression`
  * **Preprocessing Pipeline:**
      * A `ColumnTransformer` is used.
      * **Numeric Features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`): Processed with `StandardScaler`.
      * **Categorical Features** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`): Processed with `OneHotEncoder`.

### 4\. Medical Insurance Cost Prediction

  * **Task:** Regression
  * **Model:** `RandomForestRegressor`
  * **Preprocessing Pipeline:**
      * A `ColumnTransformer` is used.
      * **Numeric Features** (`age`, `bmi`, `children`): Processed with `StandardScaler`.
      * **Categorical Features** (`sex`, `smoker`, `region`): Processed with `OneHotEncoder`.

## How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository (or download the source code):**

    ```bash
    git clone https://github.com/nandan2003/ML-Learning.git
    cd ML-Learning/Advanced/Deploy Multiple ML models to Azure Web App
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    
4. **Add this code at bottom of app.py:**
    ```bash
    if __name__ == '__main__':
    
        app.run(debug=True)
    ```

4.  **Run the Flask application:**

    ```bash
    python app.py
    ```

5.  Open your web browser and navigate to `http://127.0.0.1:5000` to view the application.
