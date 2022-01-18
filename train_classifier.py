import sys
import pandas as pd
from sqlalchemy import create_engine,inspect

import re

import nltk 
nltk.download(['punkt','wordnet','stopwords'])
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    
    """
    Loads message_categories data from the database into a dataframe.

    Args:
        database_filepath (str): url of the database containing relevant data      
        
    Returns:
        X (pd.DataFrame): data frame containing feature variables
        Y (pd.DataFrame): data frame containing target variables
        category_names (list): list containing names of target variables
    """  
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine)
    X = df['message'] 
    Y = df.drop(columns=['id','message','original','genre']) 
    category_names = Y.columns
    
    return X,Y,category_names

def tokenize(text):
    
    """
    Tokenizes a given text into separate tokens using normalization,lemmatizing, stringing.

    Args:
        text (str): string which needs to be tokenized.      
        
    Returns:
        clean_tokens (List): list of cleaned tokens
    """  
    
    #convert to lower case
    text = text.lower()
    
    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    #tokenize text
    tokens = word_tokenize(text)
    
    #remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    #initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iterate through each token
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    
    """
    Creates a pipeline containing the CountVectorizer, TFIDFtransformation and the MultiOutputClassifier while applying GridSearch.
        
    Returns:
        cv (GridSearchCV): Some classification model
    """  
    
    pipeline =  Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer()),
            ('mclf',MultiOutputClassifier(RandomForestClassifier()))
            ])
        
    parameters = {
        'vect__max_features':(None,18),
    }
        
    cv = GridSearchCV(pipeline, parameters)
        
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Checks how precise a given model predicts target variables by comparing these to the given test set.
        
    Args:
        model: Some classification model
        X_test: feature test set used for predicting target variables
        Y_test: variable test set used to compare to the predicted target variables
        category_names: list of available categories
    """  
    
    Y_pred = model.predict(X_test)
    count = 0
    for column in category_names:
        print('Category: ' + column)
        print(classification_report(Y_test[column], Y_pred[:, count]))
        count += 1

def save_model(model, model_filepath):
    
    """
    Serializes a given model into a pickle file.
        
    Args:
        model: Some classification model
        model_filepath: path where the pickle file needs to be saved
    """  
    
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()