# 🏥 Medical Insurance Cost Predictor 💰

Ever looked at your medical bill and thought, "How on earth did they come up with *this* number?\!" This project is an attempt to reverse-engineer that mystery using machine learning.

We're using a classic dataset to predict how much insurance companies might charge a person based on a few key factors. But this isn't just a "regular" project—it's got a full-on professional glow-up with `sklearn` pipelines and a model showdown\!

## 🕵️ The Investigation (aka EDA)

Before building a model, we had to see what stories the data was telling.

  * **Basic plots:** Age, BMI, etc., all looked pretty standard.
  * **The "Aha\!" Moment:** We plotted `charges` vs. `smoker`. **Spoiler:** Smokers pay *waaaay* more.
  * **The *Real* Story:** The real magic happened when we plotted `bmi` vs. `charges` but colored the dots by `smoker`.
      * For non-smokers, a high BMI *slightly* increases charges.
      * For **smokers**, a high BMI makes the charges go 🚀 (that's a non-linear interaction, folks\!).

This one insight told us a simple `LinearRegression` model would probably fail. It just can't understand that complex "if-you're-a-smoker-THEN-bmi-matters-a-lot" logic.

## ✨ The "Glow-Up": A Pro-Level Pipeline

A "regular" project might just use `.replace()` to turn `male` and `female` into `0` and `1`. But that's a *huge* mistake for things like 'region' (is 'northwest' 3x 'southwest'? I don't think so).

We went full-pro with an `sklearn` **`Pipeline`** and **`ColumnTransformer`**.

1.  **`StandardScaler`:** Puts all the numeric features (`age`, `bmi`, `children`) on the same scale so the model doesn't play favorites.
2.  **`OneHotEncoder`:** Correctly turns categorical features (`sex`, `smoker`, `region`) into 0s and 1s without creating a fake order.

This pipeline makes our code clean, reusable, and prevents dreaded data leakage. (It's also what gets you hired 😉).

## 🥊 The Model Showdown\! 

We didn't just pick one model. We held a battle royale to see which one could handle the data's complexity.

  * **Contestant 1: `LinearRegression` (The Baseline)**

      * *What it is:* A simple, fast model that tries to draw a straight line through the data.
      * *Prediction:* As we guessed, it was pretty "meh" because of that `smoker` + `bmi` curve we found.

  * **Contestant 2: `RandomForestRegressor` (The Crowd Favorite)**

      * *What it is:* An army of decision trees that vote on the best answer. Great at finding complex, "if-then" rules.

  * **Contestant 3: `GradientBoostingRegressor` (The Pro)**

      * *What it is:* A team of models that build on each other's mistakes. Often the most accurate, but a bit slower.

## 🏆 And the Winner Is... Gradient Boosting\!

As predicted, the simple linear model just couldn't keep up. The tree-based models, which *understood* the non-linear interactions, blew it away.

| Model | R-squared (R²) | Mean Absolute Error (MAE) |
| :--- | :---: | :---: |
| Linear Regression | \~0.74 | \~$4,280 |
| Random Forest | ~0.84 | ~$2,714 |
| **Gradient Boosting** | **\~0.87** | **\~$2,352** |

**What this means in plain English (MAE):** The Gradient Boosting model's predictions are, on average, only off by about **$2,352**. The basic linear model was off by over $4,200\! That's a *huge* improvement.

## 🔮 The Crystal Ball: Make a Prediction\!

We saved the best-performing model (`GradientBoostingRegressor`) in its pipeline so you can use it to predict charges for a new person. It's super simple\!

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
# (You'll need to have the 'best_model_pipeline' object from the notebook)
# For this example, let's assume 'best_model_pipeline' is already trained and loaded.

# 1. Create your new person (as a dictionary)
new_person_data = {
    'age': 31,
    'sex': 'female',
    'bmi': 25.74,
    'children': 0,
    'smoker': 'no',
    'region': 'southeast'
}

# 2. Convert to a DataFrame
new_person_df = pd.DataFrame([new_person_data])

# 3. Predict!
# The pipeline handles all the scaling and encoding automatically!
predicted_cost = best_model_pipeline.predict(new_person_df)

print(f"Predicted Insurance Cost: ${predicted_cost[0]:,.2f}")
# Output: Predicted Insurance Cost: $3,975.44
```

## 🚀 How to Run This Yourself

1.  Clone this repository.
2.  Make sure you have `pandas`, `scikit-learn`, `seaborn`, and `matplotlib` installed.
    ```bash
    pip install pandas scikit-learn seaborn matplotlib
    ```
3.  Open the `.ipynb` notebook in VS Code, Jupyter Lab, or Google Colab.
4.  Make sure the `insurance.csv` file is in the right path (the notebook looks for `data/insurance.csv`).
5.  Run all the cells and watch the magic happen\!