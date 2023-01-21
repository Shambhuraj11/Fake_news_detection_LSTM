
from flask import Flask, render_template, request

from text_prepo import text_prepro
from keras.models import load_model

app = Flask(__name__)

model=load_model('weights.02-0.17.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=request.form.get('text')
   #print(data)
    data1=text_prepro()
    num_data=data1.preprocessing(data)

    output=model.predict(num_data)
    return render_template('index.html',prediction_text="This news seems to be fake with probablity of {}".format(output))



#port = int(os.getenv("PORT",5000))
if __name__=='main':
    app.run(debug=True)