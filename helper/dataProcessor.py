import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def read_data(self):
        # Read csv file and drop duplicate rows based on 'generation' column
        df = pd.read_csv(self.data_path)
        df = df.drop_duplicates(subset='generation')
        
        # Fill the missing values with empty string
        df = df.fillna('')
        
        # Concatenate prompt_body and gen_body columns and create a new column named 'full_text'
        df['full_text'] = df['prompt_body'] + "_SEP_" + df['gen_body']
        
        # Filter out rows with less than 10 tokens in 'full_text' column
        df = df[df.apply(lambda x: len(x['full_text'].split("_SEP_")) >= 10,axis=1)]
        self.df = df
    
    def split_data(self):
        # Split data into train, validation and test sets
        df_train, df_val_test = train_test_split(self.df, test_size=0.2, random_state=42069)
        df_val, df_test = train_test_split(df_val_test, test_size=0.5, random_state=42069)
        
        # Create numpy arrays of 'full_text' and 'true_boundary_index' columns for train, validation and test sets
        train_X, train_Y = np.array(df_train['full_text']), np.array(df_train["true_boundary_index"])
        val_X, val_Y = np.array(df_val['full_text']), np.array(df_val["true_boundary_index"])
        test_X, test_Y = np.array(df_test['full_text']), np.array(df_test["true_boundary_index"])
        
        # Create a dictionary to store the data for train, validation and test sets
        self.data_dict = {
            'train': {'X': train_X, 'Y': train_Y},
            'val': {'X': val_X, 'Y': val_Y},
            'test': {'X': test_X, 'Y': test_Y}
        }
    
    def process_data(self):
        # Call read_data() and split_data() methods to process the data
        self.read_data()
        self.split_data()
        
        # Return the dictionary containing train, validation and test data
        return self.data_dict
