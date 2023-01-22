
from flask import Flask, render_template, request

from keras.models import load_model
import re

from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences

app = Flask(__name__)


model=load_model('weights.05-0.17.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=request.form.get('data')
   #print(data)
    corpus=[]
    lema=WordNetLemmatizer()
    review=re.sub('[^a-zA-Z]',' ',data)
    review=review.lower()
    review=review.split()

    review=' '.join(review)
    corpus.append(review)

    voc_size=15000
    one_hot_rep=[one_hot(word,voc_size)for word in corpus]
    sent_length=20
    embed_doc=pad_sequences(one_hot_rep,padding='pre',maxlen=sent_length)

    output=model.predict(embed_doc)
    return render_template('index.html',prediction_text="This news seems to be fake with probablity of {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)

