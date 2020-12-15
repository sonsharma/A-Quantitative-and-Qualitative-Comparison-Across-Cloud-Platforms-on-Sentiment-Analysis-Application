from flask import Flask
from flask import render_template,url_for,request
import json
import requests
import pandas as pd
import time,os, pickle
import keras
from keras.preprocessing import sequence

app = Flask(__name__)
port = int(os.getenv("PORT"))
#port = 8082

@app.route('/')
def home():
    return render_template('home.html')
azure_prediction=''
aws_prediction=''
test_accu=''
train_accu=''
train_loss=''
test_loss=''
train_time=''
deploy_time=''
test_time_taken=''
training_memory = ''
deploy_memory = ''
with open(os.path.join('tokenizer_model.pkl'), 'rb') as handle:
    tokenizer_model = pickle.load(handle)

@app.route('/predict_azure',methods=['POST'])
def predict_azure():
    if request.method == 'POST':
        start_time = time.time()
        print("Azure Request")
        test_accu = '0.8655'
        train_accu = '0.99'
        train_loss = '0.0246'
        test_loss = '0.4217'
        train_time = '2min 41s'
        deploy_time = '10.08min'
        training_memory='Peak memory: 3718.12 MiB\nIncrement: 224.91 MiB'
        deploy_memory='Peak memory: 3365.89 MiB\nIncrement: 0.62 MiB'
        Reviews = request.form['Azure Review']
        scoring_uri = 'http://bced7c1a-787b-49b7-9955-9a3c4c2412e5.westeurope.azurecontainer.io/score'
        twt_azure=[]
        twt_azure.append(Reviews)
        print("1-",twt_azure)
        twt_azure = tokenizer_model.texts_to_sequences(twt_azure)
        data = {"data": twt_azure}
        print("2",twt_azure)
        input_data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        # Make the request and display the response
        resp = requests.post(scoring_uri, input_data, headers=headers)
        print(resp.text)
        azure_prediction = resp.text.split(',')[0].split(':')[1][3:-2]
        print(azure_prediction)
        test_time_taken=str(time.time() - start_time)+'s'
        return render_template('home.html',azure_prediction = azure_prediction, aws_prediction=aws_prediction, azure_text = Reviews,
                               azure_test_accu=test_accu, azure_train_accu = train_accu, azure_train_loss = train_loss,
                               azure_test_loss = test_loss, azure_train_time = train_time, azure_deploy_time = deploy_time,
                               azure_time_taken=test_time_taken,azure_training_memory=training_memory,azure_deploy_memory=deploy_memory)

@app.route('/predict_aws',methods=['POST'])
def predict_aws():
    if request.method == 'POST':
        start_time = time.time()
        print("AWS Request")
        test_accu = '0.8935'
        train_accu = '1.0000'
        train_loss = '3.8156e-05'
        test_loss = '0.6477'
        train_time = '2min 19s'
        deploy_time = '6min 33s'
        training_memory='Peak memory: 2772.10 MiB\nIncrement: 1647.09 MiB'
        deploy_memory='Peak memory: 2051.92 MiB\nIncrement: 11.10 MiB'
        Reviews2 = request.form['AWS Review']
        headers = {'Content-Type': 'application/json'}
        scoring_uri_aws = 'https://4hlqzz7hz6.execute-api.us-east-2.amazonaws.com/test/sentiment_webapp'
        twt_aws=[]
        twt_aws.append(Reviews2)
        twt_aws = tokenizer_model.texts_to_sequences(twt_aws)
        twt_aws = str(twt_aws)
        data_aws = {"review": twt_aws}
        resp = requests.post(scoring_uri_aws, headers=headers, params=data_aws)
        print(resp.text)
        aws_prediction = resp.text.split(',')[0].split(':')[1][2:-1]
        print(aws_prediction)
        test_time_taken=str(time.time() - start_time)+'s'
        return render_template('home.html', aws_prediction=aws_prediction, azure_prediction = azure_prediction, aws_text=Reviews2,
                               aws_test_accu=test_accu, aws_train_accu = train_accu, aws_train_loss = train_loss,
                               aws_test_loss = test_loss, aws_train_time = train_time, aws_deploy_time = deploy_time,
                               aws_time_taken = test_time_taken,aws_training_memory=training_memory,aws_deploy_memory=deploy_memory)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=port)
