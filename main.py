from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
import logging

# Set logging
logging.basicConfig(filename='model.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Load and preprocess data
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)

import joblib
joblib.dump(model, 'loan_model.pkl')