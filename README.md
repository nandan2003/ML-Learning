# Machine Learning & AI Portfolio

Welcome to my Machine Learning repository\! This collection documents my journey from foundational algorithms to advanced Deep Learning systems and Cloud Deployments. It contains end-to-end projects spanning Natural Language Processing (NLP), Computer Vision, and MLOps using tools like Azure, Docker, and TensorFlow.

## 📂 Repository Structure

The projects are organized by complexity and domain:

  * **Advanced:** Full-stack ML applications deployed to the cloud (Azure, Docker).
  * **Specialization:** Deep Learning projects focusing on NLP (LSTMs, Transformers) and Computer Vision (CNNs, Transfer Learning).
  * **The Foundation:** Core classical machine learning implementations (Regression, Classification, Clustering).

-----

## Advanced: Deployments & MLOps

These projects demonstrate the ability to take models out of notebooks and into production environments.

### 1\. Multi-Model Azure Web App

**Stack:** Flask, Scikit-Learn, Azure App Service

  * **Description:** A unified web interface serving four distinct ML models simultaneously. It handles both regression and classification tasks via a single Flask backend.
  * **Models Included:** Car Price Prediction, Diabetes Diagnosis, Heart Disease Prediction, and Medical Insurance Cost estimation.
  * **Key Tech:** Uses `Joblib` for serialization and `ColumnTransformer` pipelines to ensure preprocessing consistency between training and inference.
  * **Live Demo:** [https://multi-ml-models-nandanv76.azurewebsites.net/](https://multi-ml-models-nandanv76.azurewebsites.net/)

### 2\. Multiple Disease Prediction System (Docker + Streamlit)

**Stack:** Streamlit, Docker, SVM, Logistic Regression

  * **Description:** A containerized web application predicting Diabetes, Heart Disease, and Parkinson's Disease.
  * **Deployment:** Fully containerized using Docker for cross-platform compatibility and pushed to Docker Hub.
  * **Performance:** The Parkinson's model achieves **87.1% accuracy** using vocal measurement features like `jitter` and `shimmer`.

-----

## Specialization: Deep Learning

### Natural Language Processing (NLP)

**1. End-to-End News Sentiment Pipeline (Microsoft Fabric)**

  * **Stack:** Azure Synapse, Bing API, Delta Lake, Power BI, Microsoft Teams.
  * **Workflow:** A sophisticated pipeline that ingests news data via the Bing API (Bronze layer), processes it into structured tables (Silver layer), and performs sentiment analysis (Gold layer).
  * **Automation:** Uses Data Activator to trigger real-time alerts to Microsoft Teams when positive news is detected.

**2. IMDB Sentiment Analysis with LSTM**

  * **Stack:** TensorFlow/Keras, LSTM, Embedding Layers.
  * **Architecture:** Uses Long Short-Term Memory (LSTM) networks to understand the sequence of words in movie reviews, rather than just isolated keywords.
  * **Results:** Achieves **\~85.4% accuracy** on the test set using a vocabulary of the top 5,000 most common words.

### 👁️ Applied Computer Vision

**1. Dog vs. Cat Classification (Transfer Learning)**

  * **Stack:** TensorFlow, MobileNetV2.
  * **Method:** Leverages **Transfer Learning** by freezing the layers of a pre-trained MobileNetV2 model and adding a custom dense output layer.
  * **Performance:** Achieves **\~97.75% accuracy** with only 5 epochs of training on a subset of images.

**2. Face Mask Detection**

  * **Stack:** OpenCV, MobileNetV2, TensorFlow.
  * **Features:** A real-time pipeline that first detects faces using OpenCV's Haar Cascades and then classifies them as "With Mask" or "Without Mask".

**3. MNIST Digit Classification**

  * **Stack:** Keras Sequential API.
  * **Overview:** A foundational Computer Vision project that achieves **\~97.1% accuracy** recognizing handwritten digits.

-----

## 🏗️ The Foundation: Classical ML

**1. Spam Mail Classifier**

  * **Technique:** TF-IDF Vectorization + Logistic Regression.
  * **Highlights:** Uses `GridSearchCV` to tune hyperparameters across 12 combinations, achieving **98.48% accuracy**. The workflow includes a rigorous pipeline to prevent data leakage.

**2. Mall Customer Segmentation**

  * **Technique:** K-Means Clustering (Unsupervised Learning).
  * **Highlights:** Goes beyond 2D plotting by utilizing a 4-dimensional approach (Age, Gender, Income, Spending Score) and visualizes the result using 3D interactive plots with Plotly.
  * **Validation:** Uses the **Silhouette Score** and Elbow Method to scientifically determine the optimal number of clusters.

-----

## Tech Stack

  * **Languages:** Python
  * **Deep Learning:** TensorFlow, Keras
  * **Machine Learning:** Scikit-learn, NumPy, Pandas
  * **Cloud & DevOps:** Microsoft Azure (Web App, Synapse, Data Factory), Docker
  * **Web Frameworks:** Flask, Streamlit
  * **Visualization:** Power BI, Matplotlib, Seaborn, Plotly

## Getting Started

To run any of these projects locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nandan2003/ML-Learning.git
    ```
2.  **Navigate to the specific project folder.**
3.  **Install dependencies** (found in the project's `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application** (e.g., for the Azure Web App):
    ```bash
    python app.py
    ```
    *(Refer to individual project READMEs for specific instructions.)*