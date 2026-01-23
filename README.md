# 🏠 House Price Prediction System

A full-stack **Machine Learning + Web Application** designed to predict real estate prices based on location, property characteristics, and market-related features. This project demonstrates the complete ML lifecycle — from data preprocessing and feature engineering to model training, deployment, and API consumption via a web interface.

> 📌 This project was developed as an academic and practical capstone project, focusing on applying machine learning techniques to real-world housing data.

---

## 📖 Table of Contents

* [Project Overview](#project-overview)
* [Key Objectives](#key-objectives)
* [Features](#features)
* [System Architecture](#system-architecture)
* [Dataset Description](#dataset-description)
* [Machine Learning Pipeline](#machine-learning-pipeline)
* [Modeling Approach](#modeling-approach)
* [API Design](#api-design)
* [Web Application](#web-application)
* [Technology Stack](#technology-stack)
* [Installation & Setup](#installation--setup)
* [Running the Project](#running-the-project)
* [Model Evaluation](#model-evaluation)
* [Results & Insights](#results--insights)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Project Structure](#project-structure)
* [Screenshots](#screenshots)
* [Conclusion](#conclusion)

---

## 📌 Project Overview

The **House Price Prediction System** is a machine learning–powered web application that estimates housing prices based on multiple input factors such as:

* Geographic location (City → District → Ward)
* Property size (area in square meters)
* Property attributes (number of rooms, floors, frontage, etc.)

The goal is to provide an **accurate, scalable, and user-friendly solution** for real estate price estimation by combining data science and web development practices.

---

## 🎯 Key Objectives

* Apply supervised machine learning techniques to real estate pricing problems
* Build a clean and reusable data preprocessing pipeline
* Develop a RESTful API for model inference
* Integrate the ML model into a web-based user interface
* Ensure reproducibility and deployability of the trained model

---

## ✨ Features

### 🔍 Machine Learning

* Regression-based house price prediction
* Feature scaling using `StandardScaler`
* Model persistence using serialized artifacts
* Consistent preprocessing for training and inference

### 🌐 Web Application

* User-friendly form for inputting house features
* Location-based selection (City → District → Ward)
* Real-time price prediction results
* Clean and responsive UI

### ⚙ Backend API

* Flask-based REST API
* JSON-based request/response
* Separation of ML logic and web logic

---

## 🧱 System Architecture

```text
User Interface (HTML/CSS)
        ↓
   Flask Web Server
        ↓
ML Prediction Pipeline
        ↓
 Trained Regression Model
```

The system follows a **layered architecture**, ensuring modularity and maintainability.

---

## 📊 Dataset Description

The dataset consists of real estate listings collected from public sources and structured into tabular format.

### Common Features:

* Location (City, District, Ward)
* House area (m²)
* Number of bedrooms
* Number of bathrooms
* Frontage width
* Number of floors
* Legal status

> ⚠️ The dataset was cleaned and normalized before training to handle missing values and outliers.

---

## 🧪 Machine Learning Pipeline

1. **Data Cleaning**

   * Remove duplicates
   * Handle missing values
   * Normalize inconsistent fields

2. **Feature Engineering**

   * Encode categorical location features
   * Scale numerical features

3. **Model Training**

   * Train multiple regression models
   * Compare performance metrics

4. **Model Serialization**

   * Save trained model and scaler

5. **Inference Pipeline**

   * Load saved model
   * Apply same preprocessing steps
   * Return predicted price

---

## 🤖 Modeling Approach

The problem is formulated as a **regression task**.

### Algorithms Used:

* Linear Regression
* (Optional) Ridge / Lasso Regression

### Why Regression?

* Continuous target variable (price)
* Interpretability of model coefficients
* Fast training and inference

---

## 🔌 API Design

### Endpoint: Predict House Price

**POST** `/predict`

#### Request Body (JSON):

```json
{
  "city": "Hanoi",
  "district": "Cau Giay",
  "ward": "Dich Vong",
  "area": 80,
  "bedrooms": 3,
  "bathrooms": 2,
  "floors": 3
}
```

#### Response:

```json
{
  "predicted_price": 3200000000
}
```

---

## 🖥 Web Application

The frontend allows users to:

* Select location hierarchy
* Enter property details
* Submit data for prediction
* View predicted house price instantly

The UI is intentionally simple to keep the focus on the ML model and usability.

---

## 🧰 Technology Stack

### Machine Learning

* Python
* scikit-learn
* pandas
* numpy

### Backend

* Flask
* Pickle / Joblib

### Frontend

* HTML5
* CSS3
* JavaScript

---

## ⚙ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/lethanhdo1704/dudoangianhacuoicung.git
cd dudoangianhacuoicung
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

---

## 📈 Model Evaluation

Evaluation metrics used:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

The trained model achieved **reasonable accuracy** for academic and demonstration purposes.

---

## 📊 Results & Insights

* Location plays a dominant role in price estimation
* Area and number of floors significantly affect predictions
* Feature scaling improves model stability

---

## ⚠️ Limitations

* Dataset size is limited
* Model performance depends heavily on data quality
* Prices may not reflect real-time market changes

---

## 🚀 Future Improvements

* Use larger and more diverse datasets
* Apply advanced models (XGBoost, Random Forest)
* Integrate real-time data scraping
* Deploy model using Docker
* Improve UI/UX

---

## 📁 Project Structure

```text
.
├── data/
├── model/
├── app.py
├── requirements.txt
├── templates/
├── static/
└── README.md
```

---

## 🏁 Conclusion

This project demonstrates the practical application of machine learning in real estate pricing while showcasing end-to-end system design — from data preprocessing to deployment.

It serves as a strong foundation for further experimentation and production-level enhancements.

---

## 👤 Author

**Lê Thành Đô**
Fullstack Developer | Machine Learning Enthusiast
GitHub: [https://github.com/lethanhdo1704](https://github.com/lethanhdo1704)

---

⭐ If you find this project helpful, feel free to give it a star!
