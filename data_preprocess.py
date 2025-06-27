import pandas as pd
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

    df = pd.read_csv(raw_url, header=None, names=column_names)

    cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero_invalid:
        df[col] = df[col].replace(0, pd.NA)

    df.fillna(df.median(), inplace=True)

    # Drop Pregnancies
    df = df.drop(columns=["Pregnancies"])

    # Add BMI Category (underweight, normal, overweight, obese)
    def bmi_category(bmi):
        if bmi < 18.5:
            return 0  # Underweight
        elif bmi < 25:
            return 1  # Normal
        elif bmi < 30:
            return 2  # Overweight
        else:
            return 3  # Obese

    df['BMI_Category'] = df['BMI'].apply(bmi_category)

    # Save processed data
    cleaned_csv_path = os.path.join(current_dir, "cleaned_diabetes_modified.csv")
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Modified dataset saved to {cleaned_csv_path}")

if __name__ == "__main__":
    main()
