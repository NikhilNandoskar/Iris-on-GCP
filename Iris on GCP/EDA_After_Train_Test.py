#Imports
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split

from EDA import *

np.random.seed(0)

def transform_data(df):
    X = df.iloc[:,:-1]  # Idependent Features 
    y = df.iloc[:,-1]   # Dependent Feature
    # convert from dataframes to numpy matrices
    X = np.array(X.values)  
    y = np.array(y.values)
    y = y.flatten()
    return X, y

def save_numpy(path,X_train, X_test, y_train, y_test):
    np.save(path+'X_Train.npy',X_train)
    np.save(path+'X_Test.npy',X_test)
    np.save(path+'y_Train.npy',y_train)
    np.save(path+'y_Test.npy',y_test)
    
def create_dataframe(X,y):
    df_data = pd.DataFrame(X, columns=['sepal_length', 'sepal_width','petal_length','petal_width'])
    df_target = pd.DataFrame(y, columns=['species'])
    
    # Final DataFrame 
    df = pd.concat([df_data,df_target],axis=1)
    return df

def task_function(path,task,image_path):
    X = np.load(path+'X_'+task+'.npy')
    y = np.load(path+'y_'+task+'.npy')
    df = create_dataframe(X,y)
    col_nm = df.columns
    data_stats(df)
    histogram_plot(df,col_nm,image_path,'Histogram_'+task+'.jpg')
    count_plot(df,image_path,'Class_Balance_'+task+'.jpg')
    box_plot(df,col_nm,image_path,'Boxplot_'+task+'.jpg')
    heatmap(df,image_path,'HeatMap_'+task+'.jpg')
    
if __name__ == '__main__':
    start = time.time()
    path = 'Dataset/'
    image_path = 'static/Images/'
    df = load_dataset('data.pkl')
    X, y = transform_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=27)
    save_numpy(path,X_train, X_test, y_train, y_test)
    task = input('Enter Task (Train or Test): ' )
    if task == 'Train':
        task_function(path,task,image_path)
    else:
        task_function(path,task,image_path)
    end = time.time()
    print('EDA Finised in:', (end-start))

