import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import pickle
import json
import random

from keras.models import load_model
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_user_input(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
  return sentence_words

def get_bag_of_words(sentence, words):
  sentence_words = clean_up_user_input(sentence)
  bag = [0] * len(words)
  for s in sentence_words:
    for i, word in enumerate(words):
      if word == s:
        bag[i] = 1

  return np.array(bag)

def predict_class(sentence):
    # filter below  threshold predictions
    p = get_bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

print("\n\n")
while True:
  msg = input()
  if msg != '':
    ints = predict_class(msg)
    res = getResponse(ints, intents)

    print("Bot: " + res + "\n\n")

