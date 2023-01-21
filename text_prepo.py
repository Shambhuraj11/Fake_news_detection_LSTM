import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences

class text_prepro:
    
    
    def preprocessing(self,data):

        self.data=data
        self.review=''
        self.one_hot_rep=[]
        self.embed_doc=[]
        lema=WordNetLemmatizer()
       
        for i in range(len(self.data)):
            self.review=re.sub('[^a-zA-Z]',' ',self.data)
            self.review=self.review.lower()
            self.review=self.review.split()
            self.review=[lema.lemmatize(word) for word in self.review if not word in set(stopwords.words('english'))]
            self.review=' '.join(self.review)
        voc_size=15000
        self.one_hot_rep=[one_hot(word,voc_size)for word in self.review]
        sent_length=20
        self.embed_doc=pad_sequences(self.one_hot_rep,padding='pre',maxlen=sent_length)
        return self.embed_doc
