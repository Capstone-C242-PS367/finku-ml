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

### 1. **Object Detection Model:**
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

### 2. **Character Recognition Model:**
   A character recognition model is a type of machine learning model designed to recognize and 
   classify individual characters (numbers from 0-9, letters from a-z in uppercase and lowercase) in an image.

   An example of our character recognition dataset used for training: 
   
   ![OCR Datasetl](https://raw.githubusercontent.com/Capstone-C242-PS367/finku-ml/main/OCR%20Training/Dataset/Example_Dataset.png)
   
   Our TensorFlow Character Recognition Model look like this: 
   ![ModelSummary](https://github.com/Capstone-C242-PS367/finku-ml/blob/main/ModelSummary.jpg)

   After we train it for 20 epochs, we obtain
   
   **Accuracy:** 0.9408

   **Loss:** 0.1494

   **Validation Accuracy:** 0.8906

   **Validation Loss:** 0.3106

   ![OCR Eval](https://raw.githubusercontent.com/Capstone-C242-PS367/finku-ml/main/OCR%20Training/OCR%20Training%20Evaluation.png)
