# Titanic Survival Prediction: The "Unsinkable" Pipeline 🚢

This project tackles the legendary Titanic survival challenge. But instead of just a simple script, this repository contains a robust, professional machine learning workflow designed for accuracy, reusability, and to prevent common errors like data leakage.

This isn't your average Titanic kernel. It's built to be better.

-----

## ✨ Key Features

What makes this project "more than the regular projects out there"?

  * **Professional `sklearn` Pipeline:** No data leakage here\! All preprocessing (imputation, scaling, encoding) is neatly bundled into a `ColumnTransformer` and `Pipeline`. The model only learns from the training data, just like it would in a production environment.
  * **Smart Feature Engineering:** We go beyond the basic columns. This model learns from specially engineered features like:
      * `Title`: Extracted from passenger names (Mr., Mrs., Miss, Master).
      * `FamilySize` & `IsAlone`: Combined from the `SibSp` and `Parch` columns.
      * `HasCabin`: A binary feature based on whether the `Cabin` data was missing or not.
      * `Fare`: Log-transformed to handle its heavy skew.
  * **Model Selection:** The notebook doesn't just pick one model. It trains three:
    1.  Logistic Regression
    2.  Random Forest
    3.  Gradient Boosting

## 🔧 Dependencies

All you need are the classics. You can run the `.ipynb` file in any environment (like VS Code, Jupyter Lab, or Google Colab) that has the following libraries installed:

  * `pandas`
  * `numpy`
  * `seaborn`
  * `matplotlib`
  * `scikit-learn`

## Workflow Overview

The notebook follows a clear, step-by-step process:

1.  **Import & Load:** All libraries and the `titanic.csv` data are loaded.
2.  **Feature Engineering:** This is where the magic happens. We create all the new, predictive features described above.
3.  **Define Pipelines:** We create separate, clean pipelines for our `numeric_features` (which get imputed and scaled) and `categorical_features` (which get imputed and one-hot encoded).
4.  **Bundle:** These two pipelines are combined into a single `ColumnTransformer` that handles all preprocessing in one step.
5.  **Train & Select:** The notebook trains all three models and uses their test F1-scores to automatically assign the top performer to the `best_model_pipeline` variable.
6.  **Predict:** The final, fitted pipeline is used to make a prediction on new, completely raw sample data.

## 🚀 Making Your Own Prediction

The best part? The final `best_model_pipeline` object handles *everything*. You can feed it completely raw data (with `NaN`s, strings, and all), and it will automatically apply all the imputation, scaling, and encoding steps it learned from the training data.

Here's how easy it is to predict a new passenger's fate:

```python
# --- Make a Prediction on a New Passenger ---

# 1. Create new passenger data as a DataFrame
# Notice the data is RAW (strings, normal fare, etc.)
new_passenger_data = pd.DataFrame({
    'Pclass': [3],               # 3rd Class
    'Sex': ['male'],             # String 'male'
    'Age': [25.0],               # A float
    'Fare': [7.25],              # The original fare
    'Embarked': ['S'],           # String 'S'
    'Title': ['Mr'],             # String 'Mr'
    'FamilySize': [1],           # 1 (FamilySize = 0 SibSp + 0 Parch + 1)
    'IsAlone': [1],              # 1 (because FamilySize is 1)
    'HasCabin': [0]              # 0 (no cabin)
})

# 2. Use the .predict() method of the BEST pipeline
prediction = best_model_pipeline.predict(new_passenger_data)

# 3. Print the result
if prediction[0] == 1:
    print("Prediction: The new passenger is LIKELY TO SURVIVE.")
else:
    print("Prediction: The new passenger is LIKELY TO NOT SURVIVE.")

# Output:
# Prediction: The new passenger is LIKELY TO NOT SURVIVE.
```