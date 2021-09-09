# Imports
import numpy as np
import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from Saved_Dataset import save_pickle
from LR_Regularization_Dropout_Adam import L_layer_model,predict

np.random.seed(0)

def load_numpy(dataset_path):
    X_train = np.load(dataset_path+'X_Train.npy')
    X_test = np.load(dataset_path+'X_Test.npy')
    y_train = np.load(dataset_path+'y_Train.npy')
    y_test = np.load(dataset_path+'y_Test.npy')
    return X_train,X_test,y_train,y_test
    

def Logistic_Regression_Model(X,y,Output_classes):
    learned_parameters = L_layer_model(X, y,Output_classes,
                                       layers_dims=[X.shape[1],6,np.unique(y).shape[0]], 
                                       predict_result=False,activation_type="multiclass", 
                                       reg_type="l2",keep_prob=1, 
                                       mini_batch_size=10, n=2, 
                                       learning_rate = 0.001,lambd=0.04, 
                                       num_epochs =500)
    return learned_parameters

def Support_Vector_Machone_Model(X,y):
    svm = SVC(C=0.6)
    svm.fit(X,y)
    return svm

# Training the Model
def run_model(model_choice,X,y):
    if model_choice == 'LR':
        Output_classes = np.unique(y).shape[0]
        learned_parameters = Logistic_Regression_Model(X,y,Output_classes)
        return learned_parameters
    elif model_choice == 'SVM':
        model = Support_Vector_Machone_Model(X, y)
        return model
    else: print('Entered Model Choice is Wrong, Enter LR or SVM')

# Testing the Model    
def test_model(model,X,y,model_choice):
    if model_choice == 'LR':
        Output_classes = np.unique(y).shape[0]
        y_pred = predict(X, model,y,
                         Output_classes=Output_classes, 
                         keep_prob=1,predict_result=False,
                         activation_type="multiclass",flags="y_is_present") 
        return y_pred
    elif model_choice == 'SVM':
        y_pred = model.predict(X)
        return y_pred
    else: print('Entered Model Name is Wrong, Enter LR or SVM')

def accuracy(y,y_pred):
    m = y.shape[0]
    acc = np.sum((y_pred == y)/m)*100
    print("Accuracy:%.2f%%" % acc)

if __name__ == '__main__':
    start = time.time()
    path = 'Trained_Models/'
    dataset_path = 'Dataset/'
    # Load Training and Testing Data
    X_train, X_test, y_train, y_test = load_numpy(dataset_path)
    
    model_choice = input('Enter Model Name: ')
    try:
        model = run_model(model_choice,X_train,y_train)
        save_pickle(model,path,model_choice+'_'+'Trained_Model.pkl')
        
        print("Training Results by {}".format(model_choice))
        y_pred_train = test_model(model,X_train,y_train,model_choice)
        accuracy(y_train,y_pred_train)
        print(classification_report(y_train,y_pred_train))
        print(confusion_matrix(y_train, y_pred_train))
            
        print('\n')
        
        print("Testing Results by {}".format(model_choice))
        y_pred_test = test_model(model,X_test,y_test,model_choice)
        accuracy(y_test,y_pred_test)
        print(classification_report(y_test,y_pred_test))
        print(confusion_matrix(y_test, y_pred_test))
        
        end = time.time()
        print('Finieshed Task in:',(end-start))
    except ValueError: print('Entered Model Choice is Wrong, Enter LR or SVM')
        
    
    