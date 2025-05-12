# Importing Data Manipulation Libraries
import numpy as np 
import pandas as pd  

# Importing data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# importing Warnings
import warnings 
warnings.filterwarnings('ignore')

# Importing logging Libraries
import logging
logging.basicConfig(filename='model.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


# importing machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Loading data
URL = 'https://raw.githubusercontent.com/anirudhajohare19/Loan_Approval_Prediction_Model/refs/heads/main/research/train.csv'

df = pd.read_csv(URL)

# Dropping id Column
df.drop('id', axis=1, inplace=True)

# Splitting Data into Numerical and Categorical Data
Numerical_Data=df.select_dtypes(exclude='object')

Categorical_Data=df.select_dtypes(include='object')


# Capping Outliers 
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
    return df

outlier_cols = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
for col in outlier_cols:
    df = cap_outliers(df, col)

# Converting categorical Columns to numerical using Lable Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['person_home_ownership'] = le.fit_transform(df['person_home_ownership'])
df['loan_intent'] = le.fit_transform(df['loan_intent'])
df['loan_grade'] = le.fit_transform(df['loan_grade'])
df['cb_person_default_on_file'] = le.fit_transform(df['cb_person_default_on_file'])

# spltting Data in X and y
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Scaling The Data Using Robust scaler
from sklearn.preprocessing import RobustScaler, StandardScaler,MinMaxScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Using Smote to handle imbalance data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model Building 
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
# Predicting on test data
y_pred = RF.predict(X_test)
# Evaluating the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

