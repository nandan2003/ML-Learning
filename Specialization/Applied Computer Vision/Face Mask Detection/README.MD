# 😷 Face Mask Detector Pro

Is that person wearing a mask? Or just holding their hand to their chin? This project builds a smart CNN that knows the difference!

This isn't just a basic classifier. It's a full, real-world pipeline that:
* Uses **Transfer Learning** with the powerful MobileNetV2 model for high accuracy.
* Leverages **OpenCV** to automatically *detect* faces in any image first.
* Classifies the detected face as `with_mask` (green box) or `without_mask` (red box).
* Uses efficient `tf.data` pipelines, so it doesn't crash your RAM.

## 🚀 How to Run This Bad Boy

1.  **Get the Data:**
    * This notebook downloads its dataset from Kaggle. Instructions to get the dataset id in `data` folder.
    * This notebook extracts the `.zip` data.

2.  **Train the Model:**
    * Run the cells in order from top to bottom.
    * The notebook will build the model (using MobileNetV2 + Data Augmentation).
    * It will train the model until it stops improving (thanks, `EarlyStopping`!) and get to ~98-99% accuracy.
    * It saves the final, trained model as `face_mask_mobilenet.keras`.

3.  **Test It!** ✅
    * The final cells download an OpenCV face detector (`.xml` file).
    * The very last cell will ask you for an image path: `input('Path of the image to be predicted: ')`.
    * Upload a test image of your own to the Colab environment (e.g., `my_test_image.jpg`), type that path when prompted, and hit Enter.
    * Watch as it finds the faces and draws the boxes!

## Tech Stack

* **TensorFlow & Keras:** For building and training the model.
* **MobileNetV2:** As the pre-trained base for transfer learning.
* **OpenCV:** For its powerful Haar Cascade face detector.
* **Matplotlib:** To plot our awesome accuracy graphs.
* **NumPy:** For all the number-crunching.