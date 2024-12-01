import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from DataProcessingUtils import process_dataset

file_path = "Data/Telco_customer_churn.xlsx"
churn_all_data = pd.read_excel(file_path)

print(churn_all_data.head())

# Split the datset into training and testing
train, test = train_test_split(churn_all_data, test_size=0.2)

# Save the training and testing data
train.to_csv("Data/train.csv", index=False)
test.to_csv("Data/test.csv", index=False)

# Process the dataset and save the processed data
processed_train = process_dataset(train)
proccessed_test = process_dataset(test)

# Save the processed data
processed_train.to_csv("Data/processed_train.csv", index=False)
proccessed_test.to_csv("Data/processed_test.csv", index=False)

