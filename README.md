# Diabetes_Prediction_System
# 🩺 Diabetes Prediction System with AI Health Assistant

An interactive, AI-powered web application that predicts the likelihood of diabetes using a stacked machine learning model and provides personalized health suggestions. Built using **Streamlit**, this system also includes a **GPT-powered chatbot** for real-time health advice.

---

## 📌 Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Results](#results)
- [License](#license)

---

## 🚀 Demo

🧪 Coming Soon! *(You can deploy on [Streamlit Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces))*

---

## ✅ Features

- Predicts diabetes risk based on user input
- Uses **StackingClassifier** (XGBoost + RandomForest + Logistic Regression)
- Data preprocessing with **StandardScaler**, **OneHotEncoder**, and **SMOTE**
- Personalized health suggestions based on prediction
- Integrated **GPT-powered AI assistant** for Q&A
- Clean and responsive **Streamlit UI**

---

## 🛠 Tech Stack

- **Frontend/UI:** Streamlit, HTML/CSS  
- **Backend:** Python, Pandas, Scikit-learn, XGBoost, SMOTE, Joblib  
- **AI Integration:** OpenAI GPT API (GPT-4o-mini)  
- **Model:** StackingClassifier with hyperparameter tuning  
- **Data Storage:** CSV, joblib models  
- **Others:** BMI Categorization, Feature Engineering, Data Cleaning

---

## 📂 Dataset

- **Source:** [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:** Glucose, Blood Pressure, BMI, Age, Insulin, etc.
- **Modifications:** Removed invalid zeroes, added BMI category, handled class imbalance using SMOTE

---

## 🔍 How It Works

1. Users input health data into a Streamlit form.
2. Preprocessing and transformation is applied (scaling, encoding).
3. Model predicts diabetes risk using a trained StackingClassifier.
4. Based on the result, health suggestions are shown.
5. A built-in **AI health assistant** answers user queries using GPT.

---

## 🏗 Project Structure

diabetes_prediction_system/
│
├── app.py # Streamlit UI + prediction + chatbot
├── train_model.py # Model training and tuning script
├── data_preprocess.py # Data cleaning and feature engineering
├── diabetes_model_tuned.pkl # Trained ML model (stacked)
├── preprocessor.pkl # Saved preprocessing pipeline
├── cleaned_diabetes_modified.csv # Final dataset after cleaning
├── scaler.pkl # (If saved separately)
└── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/diabetes-prediction-system.git
cd diabetes-prediction-system
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
(Create requirements.txt with libraries like streamlit, scikit-learn, xgboost, openai, pandas, etc.)

3. Run the App
bash
Copy
Edit
streamlit run app.py
Open your browser and go to: http://localhost:8501

🖼 Screenshots
(Insert screenshots here of the UI form, prediction output, and AI assistant response)

📊 Results
Final model: StackingClassifier

Evaluated on ROC-AUC and Accuracy

Handles imbalanced data with SMOTE

Trained using 5-fold stratified cross-validation

📄 License
This project is licensed under the MIT License.

🙋‍♂️ Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

💡 Acknowledgements
Kaggle - PIMA Diabetes Dataset

Streamlit

OpenAI

yaml
Copy
Edit

---

Would you like me to:
- Create the `requirements.txt` for you?
- Help add screenshots or badges?
- Suggest a domain name or deployment plan?

Let me know!
