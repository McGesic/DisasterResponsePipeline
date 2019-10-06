import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle 


def load_data(database_filepath):
    engine = create_engine('sqlite:///' +database_filepath)
    df = pd.read_sql_table('df', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    cat_names=Y.columns
    return X, Y, cat_names

def tokenize(text):
    
    raw_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens = []
    for token in raw_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip('^°!"§$%&/()=?\}][{+~+#-_.:,;<>|')
        tokens.append(clean_token)

    return tokens


def build_model():
    model = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75, ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('classifier', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
            ])
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred=pd.DataFrame(model.predict(X_test))
    results=pd.DataFrame()
    results['categories']=category_names ### hier stand etwas anderes!
    results.set_index('categories')
    results['precision'], results['recall'], results['fscore']='','',''
    for category in range(len(category_names)):
        results['precision'][category]= precision_recall_fscore_support(Y_test.iloc[:,category], Y_pred.iloc[:,category], average='weighted')[0]
        results['recall'][category]= precision_recall_fscore_support(Y_test.iloc[:,category], Y_pred.iloc[:,category], average='weighted')[1]
        results['fscore'][category]= precision_recall_fscore_support(Y_test.iloc[:,category], Y_pred.iloc[:,category], average='weighted')[2]
    print(results['precision'].mean())
    print(results['recall'].mean())
    print(results['fscore'].mean())
    print(results)
    

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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
    