# Features of Machine Learning

This repository contains a project that builds an Optical Character Recognition (OCR) model with TensorFlow to detect characters, predict characters and words in a bank's mutation transaction, and classify class for Date, Nominal, and Type in mutation. The main problem We want to solve is the inconvenience and inconsistency of tracking finances manually. By using OCR to automatically pull data from bank transactions, our solution makes it easy for users to keep track of their finances without needing special knowledge or spending time on constant manual updates.

## Machine Learning Team Members
| Team Member                | Cohort ID    |
| -------------------------- | ------------ |
| Annisa Risty               | M002B4KX0583 |
| Ashanti Fairuza            | M002B4KX0691 |
| Muhammad Fathan Assadad    | M002B4KY2835 |

## Feature Description and Its Training

This project is designed to help users track their financial activity overtime and also set and achieve financial goals.

To achieve the goals of this project, we developed an OCR system that is divided into two models, an Object Detection Model and a Character Recognition Model:

### 1. **Object Detection Model: **
   An object detection model is a type of machine learning model designed to detect and locate objects within an image or video. 
   Unlike image classification models, which only predict a label for the entire image, 
   object detection models identify the presence and position of multiple objects within the image.
   
   The dataset used to train this Object Detection Model consists of BCA bank mutation. Initially, we collected 288 mutation account and 
   manually labeled them into three classes: blue square for Date, red square for Nominal, and green square for Type, using Roboflow.

   #### **Before Labeling**
   ![Before Labeling](https://github.com/Capstone-C242-PS367/finku-ml/blob/main/BeforeLabel.png)
   #### **After Labeling**
   ![After Labeling](https://github.com/Capstone-C242-PS367/finku-ml/blob/main/AfterLabel.png)

   
   After we labeled all the mutation account, we train them in google colab using **YOLOV5**

   The evaluation of the training looks like this
   ![Training Result](https://raw.githubusercontent.com/Capstone-C242-PS367/finku-ml/main/Object%20Detection%20Training/Training_Result.png)

   #### How the Object Detection Model Works: 
   1. Input: A Mutation Account image/pdf is input into the model.
   2. Feature Extraction: The image is passed through the backbone.
   3. Predictions: The model predicts bounding boxes, objectness scores, and class probabilities for each grid cell. 
   4. Refinement: The predictions are refined using anchor boxes and multi-scale feature maps.
   5. Non-Maximum Suppression (NMS): After predictions, NMS is applied to eliminate overlapping boxes and keep the best one for each detected object.
   6. Output: The final bounding boxes with their corresponding class labels and confidence scores are output.

### 2. **Character Recognition Model: **
   A character recognition model is a type of machine learning model designed to recognize and 
   classify individual characters (numbers from 0-9, letters from a-z in uppercase and lowercase) in an image.

   Our TensorFlow Character Recognition Model look like this: 
### Program Flows
1. Nutrition Label Scanning and Grading
- Input: URL of an image of a nutrition label.
- Process:
  - Read the image using `scikit-image` to parse it into a `numpy` array, and convert it into 3 channel only.
  - If the image is rotated or skewed, correct the image's orientation using `pytesseract` OSD feature.
  - Pass the already corrected image to the nutrition table detection model, the model will then output a prediction of the nutrition table's location.
  - Crop the original image based on the predicted nutrition table location, leaving only the nutrition table to be processed.
  - Pass the cropped image to the preprocessing function, which does many processing process to the image before passed on to `pytesseract`'s OCR.
  - The preprocessed image will then be read by `pytesseract` to extract words contained in the image, we used the sparse text with OSD method.
  - The read text is then processed once more to get the relevant content/nutritional value based on the file `nutrients.txt` located in `ocr/core/data`.
  - Every nutritional value read will then be converted to miligram units of calorie except for energy, which stays on kilocalorie.
3. Weight Category Prediction and Calorie Estimation
- Input: Users provide personal information including gender, age, height, weight, and activity level.
- Process:
  - Calculate Basal Metabolic Rate (BMR) using the provided user data and activity level to estimate Total Daily Energy Expenditure (TDEE).
  - Prepare the input data for the weight prediction model.
  - Normalize the input data using a pre-trained scaler.
  - Use a machine learning model to predict the user's weight category (e.g., normal weight, overweight) based on the normalized input.
  - Estimate the daily calorie needs of the user based on the predicted weight category and activity level.
    
## Features

### 1. Optical Character Recognition (OCR) (`ocr/`)

This feature provides an OCR tool to extract text from images. The file `app.py` on the root `ocr/` directory is used to run the Flask application for deployment. A docker container is needed to properly run the application on the cloud as it needs to install **Tesseract OCR Engine** for it to work.

- **Overview**: 
  - Extracts text from images using OCR techniques.
  - Used to extract nutritional values contained in nutrition label.
  
- **Usage**:
  1. Navigate to the `ocr/` directory:
     ```bash
     cd ocr/
     ```
  2. Make a virtual environment
     ```bash
     python3 -m venv .venv
     ```
  3. Activate the environment
     ```bash
     # with Mac OS/Linux
     source .venv/bin/activate
     # with Windows
     .venv\Scripts\activate
     ```
  4. Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
  5. Build and run the `Dockerimage`:
     ```bash
     # Make sure you have Docker daemon installed on your machine.
     docker compose up --build
     ```
  6. Hit your local endpoint (`http://127.0.0.1:5000/ocr`) with a `POST` request with request body
     ```python
     { "url": "url-of-the-image.png" }
     ```
     and you'll get the result!

- **Files**:
  - `ocr/core/main.py`: Main script for OCR processing.
  - `ocr/requirements.txt`: List of dependencies for this feature.
  - `ocr/core/data/example.png`: Sample image for testing the OCR.
  - `ocr/core/data/tessdata`: Local tessdata stored to use with `pytesseract`.
  - `ocr/core/models/detect-nutrition-label.pt`: Model used to detect nutrition label.
  - `ocr/core/utils.py`: File containing definitions of helper functions used in processing the OCR.

### 2. Grade Prediction (`feat/grade_prediction`)

This feature provides a model to predict student grades based on various inputs.

- **Overview**: 
  - Predicts student grades using historical data and other relevant features.
  - Implements machine learning algorithms for prediction.
  
- **Usage**:
  1. Navigate to the `feat/grade_prediction` directory:
     ```bash
     cd feat/grade_prediction
     ```
  2. Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the prediction script:
     ```bash
     python grade_prediction.py
     ```

- **Files**:
  - `grade_prediction.py`: Main script for grade prediction.
  - `requirements.txt`: List of dependencies for this feature.
  - `model.pkl`: Pre-trained model for grade prediction.
    
### 3. Weight Classification (`feat/weight_class`)

This feature provides a model to classify weight categories based on input data.

- **Overview**: 
  - Classifies individuals into different weight categories.
  - Uses machine learning for classification.
  
- **Usage**:
  1. Navigate to the `feat/weight_class` directory:
     ```bash
     cd feat/weight_class
     ```
  2. Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the classification script:
     ```bash
     python app.py
     ```

- **Files**:
  - `weight_class.py`: Main script for weight classification.
  - `requirements.txt`: List of dependencies for this feature.
  - `weight_model.pkl`: Pre-trained model for weight classification.

## Installation

To install all the dependencies for the entire project, you can run the following command in the root directory:

```bash
pip install -r requirements.txt
```

## Directory Structure
```bash
.
├── machine-learning-kaisar
│   ├── grade_prediction
│   │   ├── grade_prediction.py
│   │   ├── requirements.txt
│   │   └── model.pkl
│   ├── ocr
│   │   ├── core
│   │   │   ├── data
│   │   │   │   ├── tessdata
│   │   │   │   ├── nutrients.txt
│   │   │   │   └── example.jpg
│   │   │   ├── models
│   │   │   │   └── detect-nutrition-table.pt
│   │   │   ├── main.py
│   │   │   └── utils.py
│   │   │   
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   ├── compose.yaml
│   │   └── cloudbuild.yaml
│   └── weight_class
│       ├── app.py
│       ├── requirements.txt
│       ├── model_new.h5
│       └── weight_model.pkl
├── requirements.txt
└── README.md
```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or new features.
