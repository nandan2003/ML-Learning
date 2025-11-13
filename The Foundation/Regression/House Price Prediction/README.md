# 🌴 California Dreamin' (of House Prices\!) 🏠

## 🚀 Welcome to the ML "Hello, World\!"

You know how every coder starts with "Hello, World\!"? Well, in Machine Learning, you start by predicting house prices. It's basically a rite of passage.

This project is my journey through that classic challenge\! I'm diving into the **California Housing dataset** to see if I can build a model that predicts how much a house is worth. 💰

Spoiler: It's more than just `model.fit()`\!

## 🛠️ What's Under the Hood?

I didn't just use a basic model. I went for the "deluxe package" to build this thing properly:

  * **🧹 Clean Data, Happy Model:** Full data cleaning and exploratory data analysis (EDA), complete with a `seaborn` correlation heatmap.
  * **🤖 The "Fancy" Pipeline:** Using `sklearn.pipeline.Pipeline` to bundle my `StandardScaler` (so all features are treated fairly) and my model all in one. This makes it clean and prevents data leakage.
  * **🏆 Model Hunger Games:** I made three models fight for my approval:
      * Linear Regression (The old reliable 🤷‍♂️)
      * Random Forest (The crowd-pleaser 🌳)
      * XGBoost (The powerhouse 🔥)
  * **✨ Hyperparameter Tuning:** I used `GridSearchCV` to find the *absolute best* settings for my XGBoost model. No default settings allowed\!
  * **📊 Why, though?:** Plotted **Feature Importance** to see *why* the model thinks a house is expensive. (Hint: It's all about the `MedInc`\!)
  * **🔮 The Crystal Ball:** A final script that lets you input 8 features for a single house and get a live price prediction\!

## 🏆 And the Winner Is...

After the battle, one model stood tall. Based on the R2 score (how much of the price variance the model can explain), the **Tuned XGBoost** was the clear champion.

| Model | R2 Score |
| :--- | :--- |
| **🥇 XGBoost (Tuned)** | **\~0.85** |
| XGBoost (Default) | \~0.83 |
| Random Forest | \~0.81 |
| Linear Regression | \~0.60 |

That's right\! The tuned model could explain about **85%** of the price variation. The 15% it *can't* explain is probably due to things like "vibes" or "how close it is to a good coffee shop." ☕

## 📚 Tech Stack

  * Python 3
  * Pandas
  * NumPy
  * Matplotlib & Seaborn (for the pretty graphs 📈)
  * Scikit-learn (for Pipelines, GridSearchCV, and model metrics)
  * XGBoost

## 🏃‍♂️ How to Run This

Want to try it yourself?

1.  Clone this repository.
2.  Install the required libraries (you'll definitely need `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and `xgboost`).
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4.  Open `house_price_prediction.ipynb` and run all the cells\!
5.  Go to the very last cell to try your own predictions\!