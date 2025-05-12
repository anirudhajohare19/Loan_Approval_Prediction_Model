import pandas as pd 

def load_data():
    URL = 'https://raw.githubusercontent.com/anirudhajohare19/Loan_Approval_Prediction_Model/refs/heads/main/research/train.csv'
    df = pd.read_csv(URL)
    df.drop('id', axis=1, inplace=True)
    return df