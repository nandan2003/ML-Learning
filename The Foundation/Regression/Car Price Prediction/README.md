# Car Price Prediction Pro 🚗

Ever tried selling a used car and felt like you were just guessing the price? 😅 Or maybe you're in the market for one and want to know if you're getting ripped off? This project is here to help\!

This isn't just *another* basic regression project. This is a complete, portfolio-ready machine learning project that uses a robust **Scikit-Learn Pipeline**, **Feature Engineering**, and **Random Forest** to predict car prices with surprising accuracy.

## 📊 Project Overview

This project takes a raw dataset of used car sales (`car_data.csv`) and builds a high-performance regression model from start to finish. We don't just dump the data into a model; we clean it, analyze it, engineer new features, and build a *bulletproof preprocessing pipeline* before tuning and training an advanced model.

-----

## ✨ Key Features

This model is "more than regular" because it focuses on the *right* way to build an ML model:

  * 🕵️ **Deep-Dive EDA:** We go beyond just `df.info()`. The notebook includes a full exploratory data analysis with heatmaps, distribution plots, and boxplots to *truly understand* the relationships between features (like, who knew "Seller\_Type" mattered so much?\!).

  * 🧙‍♂️ **Magical Feature Engineering:** We didn't just drop the `Year` column. We transformed it into a much more powerful feature: `Car_Age`. We also used some string-splitting wizardry to extract the car's `Brand` from the `Car_Name` column.

  * 🤖 **Bulletproof ML Pipeline:** This is the project's secret sauce\! Instead of manually scaling and encoding, we use a `ColumnTransformer` and `Pipeline` to handle all preprocessing. This makes the code cleaner, prevents data leakage, and makes it *incredibly* easy to run new predictions.

  * 🌲 **Advanced Modeling:** We left simple linear regression in the dust and brought in the big guns: a **Random Forest Regressor**. This allows the model to capture complex, non-linear relationships between a car's features and its price.

  * 🛠️ **Hyperparameter Tuning:** We didn't just guess the model's settings. We used `GridSearchCV` to automatically test dozens of combinations and find the *optimal* parameters for the best performance.

-----

## 🚀 The Results

**TL;DR:** The model is pretty darn good\!

  * **R-squared (R2):** Our final model explains **\~96.38%** of the variance in car prices. (Translation: It's very accurate\!)
  * **Mean Absolute Error (MAE):** On average, the model's prediction is only off by about **₹61,933** (0.62 lakhs). For the world of used cars, that's a solid prediction\!

You can see the full breakdown in the notebook, including this plot of our model's predictions vs. the actual prices:

-----

## 🛠️ Tech Stack

  * **Core:** Python 3
  * **Data Analysis:** Pandas & NumPy
  * **Data Visualization:** Matplotlib & Seaborn
  * **Machine Learning:** Scikit-learn (Pipelines, ColumnTransformer, StandardScaler, OneHotEncoder, RandomForestRegressor, GridSearchCV)
  * **Notebook:** Jupyter

-----

## 🏃‍♀️ How to Run This Project

Ready to see the magic for yourself? Here's how to get it running on your local machine.

### 1\. Clone the Repository

```bash
git clone https://github.com/nandan2003/ML-Learning.git
cd ML-Learning/The Foundation/Regression/Car Price Prediction
```

### 2\. Create a Virtual Environment (Recommended)

```bash
# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3\. Install the Goodies

First, create a file named `requirements.txt` in the project folder and paste the following lines into it:

```txt
pandas
matplotlib
seaborn
scikit-learn
xgboost
numpy
jupyter
```

Now, install everything with one command:

```bash
pip install -r requirements.txt
```

### 4\. Run the Notebook\!

Launch Jupyter and open the notebook file:

```bash
jupyter notebook car_price_prediction.ipynb
```

Run all the cells from top to bottom, and watch the model train\!

-----

## 🔮 Predict Your Own Car's Price\!

Want to try it yourself? After you've run the whole notebook, go to the very last cell (`[123]` in your file).

You can change the details in the `new_car_data` dictionary to match any car you want.

```python
# 1. Create a dictionary with your new car's details
new_car_data = {
    'Present_Price': [8.5],  # Price in lakhs
    'Kms_Driven': [45000],
    'Fuel_Type': ['Petrol'],
    'Seller_Type': ['Individual'],
    'Transmission': ['Manual'],
    'Owner': [0],
    'Car_Age': [6],     # Remember to provide Car_Age, not Year!
    'Brand': ['honda']  # The brand we extracted
}

# 2. Convert to a DataFrame (the code does this for you)
new_car_df = pd.DataFrame(new_car_data)

# 3. Predict! 🚀
predicted_price = best_rf_model.predict(new_car_df)

print(f"Predicted Selling Price: {predicted_price[0]:.2f} lakhs")
```

Just run that cell, and you'll get an instant price prediction\!

-----

## 📈 Future Improvements

  * **Deploy It\!** This model is begging to be turned into a simple web app using Streamlit or Flask.
  * **More Models:** The notebook is already set up to try `XGBoost`. Why not see if it can beat the Random Forest?
  * **More Data:** Scrape more recent data to keep the model sharp and relevant.

-----

*This project was a fun exercise in building a professional-grade ML model. Thanks for checking it out\!*