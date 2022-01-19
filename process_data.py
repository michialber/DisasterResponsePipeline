import sys
import pandas as pd
from sqlalchemy import create_engine, inspect


def load_data(messages_filepath, categories_filepath):
    
    """
    Loads data from two .csv-files, merges and transforms these into a dataframe.

    Args:
        messages_filepath (str): url of the csv-file containing disaster messages
        categories_filepath (str): url of the csv-file containing disaster categories
               
    Returns:
        df (pd.DataFrame): data frame containing the merged message and category data
    """  
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,left_on='id',right_on='id')
       
    return df


def clean_data(df):
    
    """
    Transforms the data frame by splitting the category column in separate columns per category. Afterwards,
    binary values are assigned to these created columns.

    Args:
        df (pd.DataFrame): data frame which needs to be transformed.url of the csv-file containing disaster messages        
        
    Returns:
        df (pd.DataFrame): transformed data frame    
    """ 
    
    categories = df['categories'].str.split(';',expand=True)
    
    categories_raw = categories.head(1).values.tolist()[0]
    categories_lean = []
    for name in categories_raw:
        categories_lean.append(name[:len(name)-2])
    
    categories.columns = categories_lean
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        #replace values unequal to [0,1]
        categories[column] = categories[column].replace(2,1)
        
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    df.duplicated(keep='first').sum()
    df = df.drop_duplicates()
    df.duplicated(keep='first').sum()
    
    return df

def save_data(df, database_filename):
    
    """
    Saves the data of a data frame to a sql database

    Args:
        df (pd.DataFrame): data frame which needs to be saved to a sql database        
        database_filename (str): url to the database
    """ 
        
    url_string = 'sqlite:///' + database_filename
    engine = create_engine(url_string)
    df.to_sql('messages_categories',engine,if_exists='replace',index=False)

def main():
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
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
