from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def target_encoding(df, features, target):
    for feature in features:
        df[feature + '_enc'] = df[feature].map(df.groupby(feature)[target].mean())
    return df

def scale_zero_one(feature, df):
  scaler = MinMaxScaler()
  df[feature] = scaler.fit_transform(df[[feature]])
  return df

def map_cols(columns, mapping, df):
  for column in columns :
    df[column] = df[column].map(mapping)

  return df

def one_hot_encode_columns(df, columns_to_encode):
    """
    Takes in a DataFrame and a list of columns to one-hot encode.
    Returns the DataFrame with one-hot encoded columns as integer (0/1) values.
    """
    # Apply one-hot encoding to specified columns
    df_encoded = pd.get_dummies(df, columns=columns_to_encode)

    # Ensure all one-hot encoded columns are in integer format (0/1)
    one_hot_columns = [col for col in df_encoded.columns if any(orig_col in col for orig_col in columns_to_encode)]
    df_encoded[one_hot_columns] = df_encoded[one_hot_columns].astype(int)

    return df_encoded


def process_dataset(df):

    COLUMNS_TO_DROP = ['CustomerID', 'Count', 'Country', 'State', 'Lat Long',"Churn Label", "Churn Reason","Latitude", "Longitude"]
    COLUMNS_FOR_SCALING = ['Churn Score', "Monthly Charges", "Tenure Months", "CLTV", "Total Charges"]
    COLUMNS_FOR_TARGET_ENCODING = ["Zip Code", "City"]
    LABEL_ENCODING_COLUMNS = ["Multiple Lines", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies" ]
    LABEL_ENCODING_MAPPING = {"Yes":1, "No": 0, "No internet service" : -1, "No phone service":-1}
    ONE_HOT_ENCODING_COLUMNS = ["Internet Service", "Contract", "Payment Method"]
    TWO_VAL_COLUMNS = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
    TWO_VAL_MAPPINGS = {"Male" : 0, "Female" : 1, "Yes" : 1, "No": 0}


    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors='coerce')
    df["Total Charges"] = df["Total Charges"].fillna(0)

    # Dropping Columns 
    df_processed = df.drop(COLUMNS_TO_DROP, axis=1)
    
    # Scale the numerical Columns 
    for column in COLUMNS_FOR_SCALING:
        print(f'    ** Scaling {column} **')
        df_processed = scale_zero_one(column, df_processed)
    print("** Scaling Done **")

    # One Hot Encoding the Categorical Columns
    df_processed = one_hot_encode_columns(df_processed, ONE_HOT_ENCODING_COLUMNS)
    print("** One Hot Encoding Done **")

    # Target Encoding the Categorical Columns
    for column in COLUMNS_FOR_TARGET_ENCODING:
        print(f'    ** Target Encoding {column} **')
        df_processed = target_encoding(df_processed, [column], 'Churn Value')
    
    print("** Target Encoding Done **")

    df_processed = df_processed.drop(COLUMNS_FOR_TARGET_ENCODING, axis=1)
    print("** Columns Dropped **")


    # Label Encoding for Categorical Columns
    df_processed = map_cols(LABEL_ENCODING_COLUMNS, LABEL_ENCODING_MAPPING, df_processed)
    df_processed = map_cols(TWO_VAL_COLUMNS, TWO_VAL_MAPPINGS, df_processed)

    return df_processed