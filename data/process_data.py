"""
    Preprocessing of Data
    Project: Udacity Nanodegree - Disaster Response Pipeline
    Sample Script Syntax:
    > python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>
    Sample Script Execution:
    > python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db
    Arguments Description:
        1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
        2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
        3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Load Messages data and Categories data, and merge then into unique dataframe.
        
        Arguments:
            messages_filepath -> Path to the CSV file containing messages
            categories_filepath -> Path to the CSV file containing categories
        Output:
            df -> Combined data containing messages and categories
    """
    
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    return pd.merge(df_messages,df_categories,on='id')


def clean_data(df):
    """
        Clean loaded data
        
        Arguments:
            df -> Combined data containing messages and categories
        Outputs:
            df -> Combined clean data
    """

    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Fix the categories columns name
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(np.int)
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
        Save Data to SQLite Database
        
        Arguments:
            df -> Combined clean data
            database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')  


def main():
    """
        Main function which will Load, Clean and Save the data.
    """

    # if have 4 arguments, then execute the ETL pipeline
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    #  Print the help message
    else:
       print("Please provide the arguments correctly: \n\
            Sample Script Execution:\n\
            > python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db \n\
            Arguments Description: \n\
            1) Path to the CSV file containing messages (e.g. disaster_messages.csv)\n\
            2) Path to the CSV file containing categories (e.g. disaster_categories.csv)\n\
            3) Path to SQLite destination database (e.g. disaster_response_db.db)")


if __name__ == '__main__':
    main()