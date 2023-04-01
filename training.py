import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow import *

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
# so it opens the json file n reads it lyk a text n then loads it nd it returns a json object tats stored as a dictionary
words = []
classes = []
documents = []
ignore_letters = ['!','@','#','$','%','^','&','*',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        #tokenize takes a sentence n tokeizes it into words , we get a list
        #ex - "hey how are u" is tokenized as "heh" "how" "are" "u"
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])



