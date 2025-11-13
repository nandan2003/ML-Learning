# Diabetes Prediction: The "Pro" Version

Tired of seeing the same old `diabetes.csv` project? You know, the one where they just load the data, run `train_test_split`, and call `.fit()`?

Me too.

This project takes that classic dataset and gives it the professional treatment. We're not just building a model; we're building a **robust, leakage-free, and persistent predictive system**. This notebook is a perfect portfolio piece to show you don't just *use* machine learning, you *understand* it.

## What Makes This Project Different?

This notebook goes beyond the basics to tackle the real-world challenges hidden in this "simple" dataset.

  * **Data Detective Work:** We find the landmines in the data. Biologically impossible values (like a `BMI` or `Glucose` of 0) are identified as missing data, replaced with `NaN`, and then correctly imputed using the *median* of the training set.
  * **The "No Spoilers" Rule (No Data Leakage):** We strictly prevent data leakage. The `StandardScaler` and median imputer are fit *only* on the training data, then used to transform both the train and test sets. This ensures our model is evaluated on data it has truly never seen before.
  * **The Model Bake-Off:** Why settle for a basic `SVC(kernel='linear')`? We use `GridSearchCV` to find the *actual* best-performing SVM model, testing multiple kernels and parameters to squeeze out the best performance.
  * **Beyond "Accuracy":** In a medical context, "accuracy" is a dangerous metric, especially with an imbalanced dataset. We use a full **Classification Report** (Precision, Recall, F1) and a **Confusion Matrix** to understand *what* our model gets right and, more importantly, *what it gets wrong*.
  * **Save It For Later (Model Persistence):** The final, tuned model, the data scaler, *and* the imputation values are all saved to disk using `joblib`. This creates a complete, reusable pipeline.

## How to Run This Project

1.  **Clone or Download:** Get this project onto your local machine.
2.  **Get the Data:** This project uses the **PIMA Indians Diabetes Database**. You can download it from Kaggle.
      * Place the `diabetes.csv` file inside a `/data` folder in the project's root directory (i.e., `.../data/diabetes.csv`).
3.  **Install Dependencies:** You'll need the following Python libraries. You can install them via pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
    ```
4.  **Run the Notebook:**
      * Launch Jupyter Notebook: `jupyter notebook`
      * Open `diabetes_prediction.ipynb` and run the cells from top to bottom.

## The Workflow: A Quick Tour

The notebook is broken down into clear, documented steps:

1.  **Import & Explore (EDA):** We load the data and immediately visualize it. Histograms reveal the "0" value problem, and a correlation heatmap shows us which features are most promising.
2.  **Clean & Prepare:** This is the most critical part. We split the data *first*. Then, we apply our 0-to-`NaN` fix, median imputation, and standard scaling—all without leaking data from the test set.
3.  **Train & Tune:** We define a parameter grid for our `SVC` model and let `GridSearchCV` run a 5-fold cross-validation to find the best combination of parameters based on the `f1-score`.
4.  **Evaluate:** We unleash our best model on the unseen test set and generate a clean Classification Report and Confusion Matrix to assess its real-world performance.
5.  **Save & Predict:** The final step serializes (saves) the entire pipeline. We then build a `predict_diabetes` function that loads these saved files to make a prediction on new, raw data, just as you would in a real application.

## The Final Product: A Real Predictive System

The best part is the final section. It includes a `predict_diabetes()` function that simulates a real-world prediction.

This function takes raw patient data (as a tuple) and automatically:

1.  Loads the saved `scaler`, `impute_medians`, and `best_model`.
2.  Converts the raw data to a DataFrame.
3.  Replaces any '0' values with `NaN`.
4.  Fills the `NaN` values using the *saved medians* from the training data.
5.  Scales the data using the *saved scaler*.
6.  Makes the final prediction.

**Example Usage from the notebook:**

```python
# Data for a person who is diabetic
input_data_1 = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
prediction_1 = predict_diabetes(input_data_1)
# Output: Prediction 1: The person IS diabetic (Outcome: 1)

# Data for a likely non-diabetic person (with '0' for Insulin)
input_data_2 = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
prediction_2 = predict_diabetes(input_data_2)
# Output: Prediction 2: The person is NOT diabetic (Outcome: 0)
```