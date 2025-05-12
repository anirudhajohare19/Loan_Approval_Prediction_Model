import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE

def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
    return df

def preprocess_data(df):
    # Outlier capping
    outlier_cols = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    for col in outlier_cols:
        df = cap_outliers(df, col)

    # Label Encoding
    le = LabelEncoder()
    for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE for imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test