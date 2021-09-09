#Imports
import pandas as pd
import pickle

from sklearn import datasets

def dataset():
    iris_data = datasets.load_iris() # Loads the Iris Dataset
    
    X = iris_data.data
    y = iris_data.target
    
    # Creating Dataframes
    df_data = pd.DataFrame(X, columns=['sepal_length', 'sepal_width','petal_length','petal_width'])
    df_target = pd.DataFrame(y, columns=['species'])
    
    # Final DataFrame 
    df = pd.concat([df_data,df_target],axis=1)
    return df

def save_pickle(object_,path,pickle_file_name):
    # This function Saves DataFrame in Pickled Format
    with open(path + pickle_file_name,'wb') as f:
        pickle.dump(object_,f)

if __name__ == '__main__':
    path = 'Dataset/'
    pickle_file_name = 'data.pkl'
    df = dataset()
    save_pickle(df, path, pickle_file_name)
    
    