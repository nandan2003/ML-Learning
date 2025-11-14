# 🍿 IMDB Sentiment Sorter: An LSTM Adventure\!

Welcome\! This project is a deep dive into the world of movie reviews, using a smart AI 🤖 to figure out if a review is glowing (positive) or scathing (negative).

We're not just guessing; we're using a **Long Short-Term Memory (LSTM)** neural network, a special kind of AI that's great at understanding sequences, like text. This notebook walks through the *entire* process, from downloading the data to building a predictive model.

### Key Features

  * **Model:** A Sequential Keras model powered by an `Embedding` layer and an `LSTM` layer.
  * **Dataset:** The classic "IMDB Dataset of 50k Movie Reviews" (a perfectly balanced 25,000 positive and 25,000 negative reviews).
  * **Performance:** Achieves a solid **\~85.4% accuracy** on the 10,000-review test set.
  * **End-to-End:** From data collection to a ready-to-use `predict_sentiment` function, this notebook covers the whole pipeline.

-----

## How It Works: The Step-by-Step Breakdown

This project isn't just a black box. Here’s exactly what's happening under the hood:

1.  **Data Collection:**

      * We start by securely loading a `kaggle.json` API key.
      * The Kaggle API is used to download the `imdb-dataset-of-50k-movie-reviews` dataset directly.
      * The downloaded `.zip` file is extracted to get our `IMDB Dataset.csv`.

2.  **Data Loading & Prep:**

      * The 50,000 reviews are loaded into a `pandas` DataFrame.
      * We check the data and see it's perfectly balanced.
      * To make it machine-readable, we convert the "positive" and "negative" labels into numbers: `1` for positive and `0` for negative.

3.  **Preprocessing (The Text-to-Number Magic):**

      * The data is split into a training set (40,000 reviews) and a testing set (10,000 reviews).
      * **Tokenization:** We use the Keras `Tokenizer` to build a "vocabulary" of the top 5,000 most common words from the training data.
      * **Sequencing:** Each review is converted from a string of words into a sequence of numbers (e.g., "The movie was good" might become `[2, 14, 23, 10]`).
      * **Padding:** LSTMs need all inputs to be the same length. We use `pad_sequences` to make every review exactly `200` words long. Shorter reviews get padded with zeros, and longer ones are truncated.

4.  **Building the LSTM Model:**
    Our "brain" is a `Sequential` model with three main layers:

      * **`Embedding` Layer:** This is a clever layer that learns a 128-dimension vector for each of our 5,000 words. It helps the model find relationships between words (like "good" and "great").
      * **`LSTM` Layer:** The powerhouse. With 128 units, this layer reads the word vectors *in order*, remembering the context of the sentence to understand meaning.
      * **`Dense` Layer:** A single output neuron with a `sigmoid` activation. It squashes the final output into a single number between 0 and 1—the probability that the review is positive.

5.  **Training & Evaluation:**

      * The model is compiled with the `adam` optimizer and `binary_crossentropy` loss (the standard for binary classification).
      * It's trained for 5 epochs on the training data.
      * Finally, we unleash it on the 10,000 unseen test reviews to see how it *really* performs.

-----

## ✅ Final Results

After training, the model was evaluated on the test set:

  * **Test Loss:** 0.339
  * **Test Accuracy:** **85.42%**

This means our model can correctly predict the sentiment of a brand new, unseen movie review about 85% of the time\!

## 🚀 How to Use This Project

Ready to try it yourself?

1.  **Get Your Key:** This notebook needs a `kaggle.json` API key file in the same directory to download the dataset. You can get this from your Kaggle account page.

2.  **Install Dependencies:** You'll need the libraries listed in the first code cell.

    ```bash
    pip install tensorflow pandas scikit-learn
    ```

3.  **Run the Notebook:** Just run the cells in `IMDB_reviews_sentiment_analysis_LSTM.ipynb` from top to bottom.

4.  **Test Your Own Reviews\!**
    The very last cell has a `predict_sentiment` function. Just change the `new_review` variable to write your own review and see what the model thinks\!

    ```python
    # example usage
    new_review = "This was the best film I've seen all year. Truly amazing."
    sentiment = predict_sentiment(new_review)
    print(f"The sentiment of the review is: {sentiment}")
    ```