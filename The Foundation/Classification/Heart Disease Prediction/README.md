# Heart Disease Predictor ❤️

Ever looked at a bunch of medical numbers and just thought... "what does any of this mean?" This project builds a machine learning model that cuts through the noise to make a clear prediction: does a person have heart disease or not?

This isn't just a single script; it's a complete model-comparison-battle-royale. We take 13 clinical features (like age, cholesterol, and chest pain type) and use them to train **five** different ML models to find the most accurate predictor.

## What's Inside?

  * **Model Showdown:** We pit five classic models against each other:
      * Logistic Regression
      * Random Forest
      * K-Nearest Neighbors (KNN)
      * Decision Tree
      * Support Vector Machine (SVM)
  * **Smart Evaluation:** We don't just look at accuracy. The code generates a full **Confusion Matrix** and **Classification Report** for every model, so we can see *exactly* how they perform, especially for a sensitive medical topic.
  * **Ready-to-Use:** The notebook identifies the "best" model, retrains it on all the data, and provides a clean `predict_heart_disease()` function. Just plug in the numbers and get a plain-English prediction.

## How it Works 🤖

The process is straightforward but powerful:

1.  **Load & Prep:** The `heart_disease_data.csv` is loaded. We check for any missing values (phew, none\!) and then split the data into our features (X) and the one thing we want to predict (Y, the `target`).
2.  **Feature Scaling:** To make sure our models treat all features fairly (so `age` doesn't get more "weight" than `sex` just because the numbers are bigger), we use `StandardScaler`. This is a critical step for models like SVM and KNN.
3.  **The Gauntlet:** We split the data into training and testing sets. Then, we loop through all five models, training each one and immediately evaluating it on the test data.
4.  **Crowning the Victor:** The script prints the performance for all models and automatically finds the one with the best accuracy score. (In this notebook, **K-Nearest Neighbors** was the champion\!).
5.  **Final Polish:** The champion model is then re-trained on the *entire* dataset to make it as smart and robust as possible.
6.  **The Oracle:** This final, fully-trained model is used in the `predict_heart_disease()` function, ready for new predictions.

## How to Use This

Want to run the prediction-battle yourself?

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/nandan2003/ML-Learning.git
    cd ML-Learning/"The Foundation"/Classification/"Heart Disease Prediction"
    ```
2.  **Install Dependencies:**
    You'll need a few key libraries.
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Check Your Data:**
    Make sure your `heart_disease_data.csv` file is in a `data/` folder, or update the path in the notebook's second code cell:
    `heart_data = pd.read_csv('data/heart_disease_data.csv')`
4.  **Run the Notebook:**
    Open and run the `heart_disease_prediction.ipynb` notebook. You can see the entire process, from data loading to the final model comparison.
5.  **Make a Prediction:**
    At the very end of the notebook, you can use the custom function to test new data:
    ```python
    # Pass in the 13 required features
    my_prediction = predict_heart_disease(
        age=62, 
        sex=0, 
        cp=0, 
        trestbps=140, 
        chol=268, 
        fbs=0, 
        restecg=0, 
        thalach=160, 
        exang=0, 
        oldpeak=3.6, 
        slope=0, 
        ca=2, 
        thal=2
    )

    print(my_prediction)
    # Output: Prediction: [1] - The Person has Heart Disease.
    ```