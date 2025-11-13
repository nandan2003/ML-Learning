# The (Not-So-Spammy) Spam Mail Classifier 🤖

Ever get a text about winning a "FREE" prize you never signed up for? This project is our answer to that. It's a machine learning model that reads a message and decides if it's **Spam** (the annoying junk) or **Ham** (the good stuff, like a message from a friend).

This notebook builds a complete, end-to-end classifier using `scikit-learn`. It not only builds a model but also finds the *best possible* version of that model using a professional-grade workflow.

## ✨ Features (What makes this project stand out)

This isn't just another "hello world" of spam filtering. This project is built using professional data science practices to be efficient, accurate, and robust.

  * **Smart Text Vectorization:** Uses `TfidfVectorizer` to turn raw text into meaningful numbers, focusing on words that are important for distinguishing spam from ham.
  * **Efficient `Pipeline`:** The entire workflow (vectorizing + modeling) is bundled into a single `sklearn.pipeline.Pipeline` object. This makes the code cleaner, less prone to errors (like data leakage), and easy to use for predictions.
  * **Automatic Model Tuning:** We don't just guess the best settings. We use `GridSearchCV` to automatically test 12 different combinations of parameters to find the absolute best-performing model.
  * **Imbalance-Aware Splitting:** The dataset is highly imbalanced (way more ham than spam). We use `stratify=Y` during our train/test split to ensure our model is trained and tested on a realistic distribution of data.
  * **In-Depth Evaluation:** We go way beyond simple accuracy. The model is evaluated with a full **classification report** (Precision, Recall, F1-Score) and a **confusion matrix** to understand *exactly* what kind of mistakes it makes (and doesn't make\!).

## Tech Stack

  * **Data Manipulation:** `pandas` & `numpy`
  * **Data Visualization:** `matplotlib` & `seaborn`
  * **Machine Learning:** `scikit-learn`
      * `TfidfVectorizer` for text processing
      * `LogisticRegression` for classification
      * `Pipeline` for workflow management
      * `GridSearchCV` for hyperparameter tuning
      * `train_test_split`, `classification_report`, `confusion_matrix` for evaluation

## How to Use

1.  **Clone this repository.**
2.  **Install dependencies.** Make sure you have the libraries from the `Tech Stack` section.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Get the data.** Make sure your `mail_data.csv` file is in the same directory as the notebook.
4.  **Run the notebook.** You can use Jupyter Lab, Jupyter Notebook, or VS Code. Run the cells one by one from top to bottom.

## Project Workflow 🌪️

Here's how the magic happens, step-by-step:

1.  **Data Loading & Cleaning:** We load the `mail_data.csv` and do some initial pre-processing. This includes filling in any missing values and mapping our text labels ('spam', 'ham') to numbers (0, 1) that the model can understand.
2.  **Exploratory Data Analysis (EDA):** We put on our detective hats and investigate the data. We create a plot to visualize the class distribution and discover that our dataset is **imbalanced**—a crucial finding that influences our next steps.
3.  **Train/Test Split:** We separate our data into features (`X`, the messages) and a target (`Y`, the 0/1 labels). We then split them into training and testing sets, using `stratify=Y` to ensure both sets have the same percentage of spam and ham.
4.  **The Magic Pipeline:** We create our `Pipeline` object. It has two steps:
      * `('tfidf', TfidfVectorizer(...))`: First, turn the text into TF-IDF features.
      * `('model', LogisticRegression(...))`: Second, feed those features into a logistic regression model.
5.  **Hyperparameter Tuning:** We define a `param_grid` of settings to try (e.g., test different n-gram ranges, different model strengths). We then pass our pipeline and this grid to `GridSearchCV` and tell it to find the best combination based on the `f1-score`.
6.  **Training:** We `fit` the `GridSearchCV` object on our training data. It automatically runs 3-fold cross-validation for all 12 parameter combinations (36 fits in total\!) and selects the champion model.
7.  **Evaluation:** The moment of truth\! We use the `best_model` from our grid search to make predictions on the unseen `X_test` data. We find it achieves **98.48% accuracy** and generate a confusion matrix, which shows it *rarely* misclassifies a real message (ham) as spam.
8.  **Predictive System:** The notebook finishes by showing how to use the fully-trained model to predict any new, custom message, even printing the model's confidence in its decision.

## Results

Our finely-tuned model is a spam-crushing machine.

  * **Test Accuracy:** **98.48%**
  * **Best Parameters Found:** `{'model__C': 10, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 2)}`
  * **Performance Deep-Dive:**
      * **Ham (Real Messages):** It has near-perfect recall (0.99), meaning it correctly identifies 99% of all real messages.
      * **Spam (Junk):** It has excellent precision (0.99) and good recall (0.90), meaning that when it says something is spam, it's almost always right, and it successfully catches 90% of all incoming spam.