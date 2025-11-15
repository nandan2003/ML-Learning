# Handwritten Digit Recognizer ✍️

Ever wondered how a computer learns to read handwriting? This project is a fun, end-to-end example of exactly that!

We use the famous MNIST dataset to teach a neural network how to identify handwritten digits (0-9). This notebook walks you through every single step, from loading the data to building a predictive system that can read a *new* image.

This model achieves **~97.1% accuracy** on the test data!

## 🧠 What's Inside This Notebook?

This isn't just a model; it's a complete workflow. Here's the play-by-play:

1.  **Data Loading:** We import the full MNIST dataset (60,000 training images and 10,000 test images) directly from `keras.datasets`.
2.  **Exploration:** We use `matplotlib` to peek at the data, visualizing a few digits to make sure they look right.
3.  **Preprocessing:** We "normalize" the images by scaling their pixel values from 0-255 down to 0-1. This is a critical step that helps the network learn faster and more reliably.
4.  **Building the Neural Network:** We create a simple but powerful `Sequential` model using Keras.
    * `Flatten` layer: Converts the 28x28 image into a single line of 784 pixels.
    * `Dense` layers: Two hidden layers (with 50 neurons each) act as the "brain" that finds patterns.
    * `Dense` output layer: A final 10-neuron layer that gives the probability for each digit.
5.  **Training:** We train the model for 10 epochs and watch its accuracy climb.
6.  **Evaluation:** We check our work! The model is evaluated on the 10,000 "unseen" test images to get its final accuracy score.
7.  **Analysis:** We build a **Confusion Matrix** with `seaborn` to see exactly *which* digits our model gets confused about. (Spoiler: it sometimes mixes up 9s and 4s).
8.  **Predictive System:** The best part! The final cells build a system to load a *brand new image* from your computer, process it just like the training data, and have the model tell you which digit it sees.

## 🤔 A Pro Challenge to Make It Even Better

This model is great, but here are two small tweaks you can try to make it *even better* and more technically robust:

* **The Output Layer:** The model uses a `sigmoid` activation in its final layer. This works, but for a multi-class problem like this, `softmax` is the industry standard. Try swapping it out and see what happens!
* **The Inversion Problem:** The test image (`MNIST_digit.png`) is a *black* digit on a *white* background, but the MNIST data is the exact opposite (white digits on black). Our model is smart enough to figure it out, but for a truly robust system, try adding a `cv2.bitwise_not()` step to *invert* the new image before predicting.

## How to Use It

1.  Open the `.ipynb` file in Google Colab or Jupyter Notebook.
2.  Run all the cells from top to bottom.
3.  When you get to the "Predictive System" at the end, upload the included `MNIST_digit.png` (or your own 28x28 image!) to see the model make a prediction.