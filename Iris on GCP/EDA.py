# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

def load_dataset(pickle_file_name):
    # This functions loads the pickled dataframe
    with open('Dataset/'+pickle_file_name,'rb') as f:
        df = pickle.load(f)
    return df
    
def data_stats(df):
    print('Here')
    print('\nData Description:')
    print(df.describe())
        
    print('\nSkweness in Data:')
    print(df.skew())
    
    print('\nNull Enteries in Data:')
    print(df.isna().sum())

def histogram_plot(df,col_nm,image_path,plot_name):
    f, axes = plt.subplots(2,2,figsize=(10,5))
    f.suptitle("Histogram Plot of All Features")
    sns.histplot(x=col_nm[0],data=df,ax=axes[0,0])
    sns.histplot(x=col_nm[1],data=df,ax=axes[0,1])
    sns.histplot(x=col_nm[2],data=df,ax=axes[1,0])
    sns.histplot(x=col_nm[3],data=df,ax=axes[1,1])
    plt.tight_layout()
    plt.savefig(image_path+plot_name)
    plt.show()
    
def count_plot(df,image_path,plot_name):
    plt.figure(figsize=(10,5))
    plt.title('Per Class Distribution')
    sns.countplot(x='species',data=df)
    plt.savefig(image_path+plot_name)
    plt.show()
    
def box_plot(df,col_nm,image_path,plot_name):
    f, axes = plt.subplots(2,2,figsize=(10,5))
    f.suptitle("Box Plot of All Features")
    sns.boxplot(x=col_nm[0],data=df,ax=axes[0,0])
    sns.boxplot(x=col_nm[1],data=df,ax=axes[0,1])
    sns.boxplot(x=col_nm[2],data=df,ax=axes[1,0])
    sns.boxplot(x=col_nm[3],data=df,ax=axes[1,1])
    plt.tight_layout()
    plt.savefig(image_path+plot_name)
    plt.show()
    
def heatmap(df,image_path,plot_name):
    correlation = df.corr()
    plt.figure(figsize=(10,5))
    plt.title('Heatmap')
    sns.heatmap(data=df[correlation.index].corr(),cmap='RdYlGn')
    plt.savefig(image_path+plot_name)
    
if __name__ == '__main__':
    start = time.time()
    df = load_dataset('data.pkl')
    col_nm = df.columns
    image_path = 'static/Images/'
    data_stats(df)
    histogram_plot(df,col_nm,image_path,'Histogram.jpg')
    count_plot(df,image_path,'Class_Balance.jpg')
    box_plot(df,col_nm,image_path,'Boxplot.jpg')
    heatmap(df,image_path,'HeatMap.jpg')
    end = time.time()
    print('EDA Finised in:', (end-start))
    

    
    

