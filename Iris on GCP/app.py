# Imports
import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
from LR_Regularization_Dropout_Adam import api_prediction
import pickle

app = Flask(__name__) 

# Load Models
model_to_pickle = "Trained_Models/"
with open(model_to_pickle+"LR_Trained_Model.pkl", 'rb') as file:
    LR_model = pickle.load(file)
    print('Logistic Regression Model Loaded')

with open(model_to_pickle+"SVM_Trained_Model.pkl", 'rb') as file:
    SVM_model = pickle.load(file)
    print('Support Vector Machine Model Loaded')
    
@app.route('/',methods=['GET','POST'])
def main():
    
    if request.method == 'POST':
        
        test_features = [float(x) if x.isdigit() else x for x in flask.request.form.values()]
        final_features = np.array(test_features[:-1],dtype=float)
        final_features = final_features.reshape((1,final_features.shape[0]))
        model_name = test_features[-1]
        
        if model_name == 'Logistic Regression':
            print('Using Logistic Regression Model')
            prediction = api_prediction(final_features,LR_model, 1,
                      Output_classes=3,keep_prob=1,predict_result=True, 
                      activation_type="multiclass" ,flags="predict_y")
        elif model_name == 'Support Vector Machine':
            print('Using Support Vector Machine Model')
            prediction = SVM_model.predict(final_features)
        else:
            print('Using Default Model: Logistic Regression')
            prediction = api_prediction(final_features,LR_model, 1,
                      Output_classes=3,keep_prob=1,predict_result=True, 
                      activation_type="multiclass" ,flags="predict_y")
        
        prediction = np.squeeze(prediction).astype(int)
        
        if prediction==0: pred="Iris Setosa"
        elif prediction==1: pred="Iris Versicolor" 
        else: pred="Iris Virginica" 
        
        return render_template('index.html',
                               prediction_text=pred,model_name=model_name)
    else:
        return render_template('index.html')

@app.route('/test',methods=['GET'])
def test_api():
    try:
        sp = request.args.get('sepal_length',default=6.3,type=float)
        sw = request.args.get('sepal_width',default=3.5,type=float)
        pl = request.args.get('petal_length',default=5,type=float)
        pw = request.args.get('petal_width',default=1,type=float)
        
        model_name = request.args.get('model_name',default='Logistic Regression',type=str)
        
        test_features = [sp,sw,pl,pw]
        final_features = np.array(test_features)
        final_features = final_features.reshape((1,final_features.shape[0]))
        
        if model_name == 'Logistic Regression':
            print('Using Logistic Regression Model')
            prediction = api_prediction(final_features,LR_model, 1,
                      Output_classes=3,keep_prob=1,predict_result=True, 
                      activation_type="multiclass" ,flags="predict_y")
        elif model_name == 'Support Vector Machine':
            print('Using Support Vector Machine model')
            prediction = SVM_model.predict(final_features)
        else:
            print('Using Default Model: Logistic Regression')
            prediction = api_prediction(final_features,LR_model, 1,
                      Output_classes=3,keep_prob=1,predict_result=True, 
                      activation_type="multiclass" ,flags="predict_y")
        
        prediction = np.squeeze(prediction).astype(int)
        
        if prediction==0: pred="Iris Setosa"
        elif prediction==1: pred="Iris Versicolor" 
        else: pred="Iris Virginica" 
        
        return jsonify(status='Complete',prediction=pred,model_name=model_name)
    except:
        return jsonify('0')
        
if __name__ == "__main__":
    app.run(debug=True)