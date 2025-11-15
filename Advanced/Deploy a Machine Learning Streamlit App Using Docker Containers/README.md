# Multiple Disease Prediction System 💡

Welcome! This project is an awesome Streamlit web application that uses machine learning to predict multiple diseases. It bundles three separate models into one easy-to-use interface, all powered by Python and containerized with Docker.

This app can predict:

  * **Diabetes:** Based on factors like Glucose, BMI, Age, and more.
  * **Heart Disease:** Based on factors like Age, Sex, Chest Pain (cp), and Cholesterol.
  * **Parkinson's Disease:** Based on a wide range of vocal measurements like `MDVP:Fo(Hz)`, `jitter`, and `shimmer`.

-----

## 🛠️ The Tech Inside

This project combines data science with web deployment. Here’s a quick look at the stack:

  * **Frontend:** **Streamlit** (with `streamlit-option-menu`) provides the interactive web UI.
  * **Machine Learning:** **Scikit-learn** is used for all model training and prediction.
  * **Data Handling:** **Pandas** and **Numpy** are used for data manipulation and processing.
  * **Deployment:** **Docker** is used to containerize the application for easy, cross-platform deployment.

### The Models

The models were trained separately in the Jupyter notebooks found in the `colab_files` directory:

1.  **Diabetes Prediction:**

      * **Model:** Support Vector Machine (`svm.SVC(kernel='linear')`)
      * **Dataset:** `dataset/diabetes.csv`
      * **Accuracy:** 77.2% on test data

2.  **Heart Disease Prediction:**

      * **Model:** Logistic Regression (`LogisticRegression()`)
      * **Dataset:** `dataset/heart.csv`
      * **Accuracy:** 81.9% on test data

3.  **Parkinson's Prediction:**

      * **Model:** Support Vector Machine (`svm.SVC(kernel='linear')`)
      * **Dataset:** `dataset/parkinsons.csv`
      * **Accuracy:** 87.1% on test data

-----

## 📂 Repository Deep Dive

Here's a map of the project to help you find your way around:

  * `app.py`: The main Python script that runs the Streamlit application.
  * `dataset/`: Contains the raw `.csv` files (diabetes, heart, parkinsons) used for training the models.
  * `colab_files/`: Contains the Jupyter Notebooks (`.ipynb`) that show the step-by-step process of data analysis, preprocessing, and model training.
  * `saved_models/`: Holds the final, trained models saved as `.sav` files using `pickle`. These are loaded by `app.py` to make live predictions.
  * `Dockerfile`: The "recipe" file that tells Docker how to build an image of this app, including all dependencies and configurations.
  * `requirements.txt`: A list of all the Python packages required to run the project.
  * `config.toml` & `credentials.toml`: Configuration files for Streamlit, used to set the server port (to 80) and other settings inside the Docker container.

-----

## 🚀 Get it Running!

Here are the original, step-by-step instructions for testing and deploying the application.

## Preferred IDE: Pycharm

### Test the streamlit app on local:

1.  Install required dependencies on local:

    ```commandline
    pip install -r requirements.txt
    ```

2.  Test the streamlit app on local:

    ```
    streamlit run app.py
    ```

### Building the docker image

(Note: Run as administrator on Windows and remove "sudo" in commands)

3.  Important - Make sure you have installed Docker on your PC:

      * Linux: Docker
      * Windows/Mac: Docker Desktop

4.  Start Docker:

      * Linux (Home Directory):
        ```
        sudo systemctl start docker
        ```
      * Windows: You can start Docker engine from Docker Desktop.

5.  Build Docker image from the project directory:

    ```commandline
    sudo docker build -t Image_name:tag .
    ```

### (Note: Rerun the Docker build command if you want to make any changes to the code files and redeploy.)

### Running the container & removing it

6.  Switch to Home Directory:

    ```
    cd ~
    ```

    List the built Docker images

    ```
    $ sudo docker images
    ```

7.  Start a container:

    ```commandline
    sudo docker run -p 80:80 Image_ID
    ```

8.  This will display the URL to access the Streamlit app ([http://0.0.0.0:80](http://0.0.0.0:80)). Note that this URL may not work on Windows. For Windows, go to http://localhost/.

9.  In a different terminal window, you can check the running containers with:

    ```
    sudo docker ps
    ```

10. Stop the container:

      * Use `ctrl + c` or stop it from Docker Desktop.

11. Check all containers:

    ```
    sudo docker ps -a
    ```

12. Delete the container if you are not going to run this again:

    ```
    sudo docker container prune
    ```

### Pushing the docker image to Docker Hub

13. Sign up on Docker Hub.

14. Create a repository on Docker Hub.

15. Log in to Docker Hub from the terminal. You can log in with your password or access token.

    ```
    sudo docker login
    ```

16. Tag your local Docker image to the Docker Hub repository:

    ```
    sudo docker tag Image_ID username/repo-name:tag
    ```

17. Push the local Docker image to the Docker Hub repository:

    ```
    sudo docker push username/repo-name:tag
    ```

(If you want to delete the image, you can delete the repository in Docker Hub and force delete it locally.)

18. Command to force delete an image (but don't do this yet):
    ```
    $ sudo docker rmi -f IMAGE_ID
    ```
