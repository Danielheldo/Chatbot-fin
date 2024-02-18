import random


import json
import pickle
import numpy
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow

from flask import Flask, render_template, request
global model


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['JSON_AS_ASCII'] = False


with open('intents.json') as file:
    intents_json = json.load(file)

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tensorflow.keras.models.load_model('chatbot_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')

def get_bot_response():
    message = request.args.get('msg')
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
              bag[i] = 1
    return numpy.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    print("Bag of words:", bow)
    res = model.predict(numpy.array([bow]))[0]
    print("Predicted probabilities:", res)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    print("Predicted intents:", return_list)
    return return_list

def get_response(ints, intents):
    if len(ints) > 0:
        tag = ints[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['response'])
                return result
    return "Entschuldige, ich verstehe diese Frage nicht."



#def get_response(ints, intents):
    #result = None
    #if not intents:
    #    return "Entschuldige, ich verstehe diese Frage nicht."
    #if not ints:
    #   return "Entschuldige, ich verstehe diese Frage nicht."
    #tag = ints[0]['intent']
    #list_of_intents = intents['intents']
    #print(list_of_intents)
    #for i in list_of_intents:
    #    if i['tag'] == tag:
    #       result = random.choice(i['response'])
    #       break
    #    print(i['tag'], tag)
    #return result

print("Du kannst beginnen!")

if __name__ == '__main__':
    app.run(debug=True)