from flask import Flask, render_template, request
import joblib
from text_prepo import text_prepro

app = Flask(__name__)

model = joblib.load('lstm_19.sav')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    data=request.form.get('text')
    #print(data)
    num_data=0
    output=0
    data=text_prepro()
    num_data=data.preprocessing(data)

    output=model.predict(num_data)
    return render_template('index.html',prediction_text="This news seems to be fake with probablity of {}".format(output))




if __name__=='main':
    app.run(debug=True)