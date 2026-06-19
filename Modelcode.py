import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import os

RANDOM_STATE = 42
FILE_NAME = 'Liver Patient Dataset (LPD)_train.csv'
TARGET_COL = 'Result'
warnings.filterwarnings('ignore')

print("\n--- Liver Patient Analysis Pipeline ---")

if not os.path.exists(FILE_NAME):
    print(f"\nCRITICAL ERROR: File '{FILE_NAME}' not found.")
    print("Please upload the file to your session!")
else:
    print(f"\nFound file: '{FILE_NAME}'. Starting process...")
    try:

        data = pd.read_csv(FILE_NAME, encoding='latin1')
        data.columns = data.columns.str.strip()

        print("\n--- EDA START ---")

        sns.set_style("whitegrid")

        plt.figure(figsize=(7,5))
        sns.countplot(x=TARGET_COL, data=data)
        plt.title("Target Variable Distribution")
        plt.show()

        plt.figure(figsize=(7,5))
        sns.countplot(x="Gender of the patient", data=data)
        plt.title("Gender Distribution")
        plt.show()

        plt.figure(figsize=(18,5))

        plt.subplot(1,3,1)
        sns.histplot(data['Age of the patient'], kde=True)
        plt.title("Age")

        plt.subplot(1,3,2)
        sns.histplot(data['Total Bilirubin'], kde=True)
        plt.title("Total Bilirubin")

        plt.subplot(1,3,3)
        sns.histplot(data['Alkphos Alkaline Phosphotase'], kde=True)
        plt.title("Alkaline Phosphotase")

        plt.tight_layout()
        plt.show()

        df2 = data.copy()
        df2['Gender of the patient'] = df2['Gender of the patient'].map({'Male':1,'Female':0})
        plt.figure(figsize=(12,9))
        sns.heatmap(df2.corr(numeric_only=True), annot=True)
        plt.title("Correlation Heatmap")
        plt.show()

        print("--- EDA DONE ---\n")


        categorical = ['Gender of the patient']
        numerical = [c for c in data.columns if c not in categorical + [TARGET_COL]]

        X = data.drop(TARGET_COL, axis=1)
        y = data[TARGET_COL].map({1:1, 2:0})

        X_train_main, X_test, y_train_main, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )

        num_tf = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_tf = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_tf, numerical),
            ('cat', cat_tf, categorical)
        ])

        models = {

            'SVM': SVC(random_state=RANDOM_STATE),
            'GBM': GradientBoostingClassifier(random_state=RANDOM_STATE),

            'XGBoost': XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss',

                n_estimators=120,
                max_depth=4,
                learning_rate=0.12,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.5
            )
        }

        print("\n--- Model Training and Evaluation ---")

        model_performance = {}

        for model_name, model in models.items():

            print(f"\n--- Training {model_name} ---")

            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', model)
            ])

            pipeline.fit(X_train_main, y_train_main)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            model_performance[model_name] = acc

            print(f"\nAccuracy: {acc:.4f}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("------------------------------------------------------------")

        print("\n--- Final Model Comparison ---")
        performance_df = pd.DataFrame(model_performance.items(), columns=['Model', 'Accuracy'])
        print(performance_df)

        plt.figure(figsize=(8,5))
        sns.barplot(data=performance_df, x='Model', y='Accuracy')
        plt.title('Accuracy Comparison')
        plt.show()

    except Exception as e:

        print(f"\nPipeline Error: {e}")
