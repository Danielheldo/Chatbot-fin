import random
import json
import pickle
import numpy
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow
from database_utils import connect_to_database, save_rating
from flask import Flask, render_template, request
global model

import sqlite3

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



def bot_response():
    message = request.args.get('msg')
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res

@app.route('/rate', methods=['POST'])
def rate_response():
    try:
        # Benutzeranfrage, Bot-Antwort und Bewertung aus dem Formular extrahieren
        user_input = request.form['user_input']
        bot_response = request.form['bot_response']
        rating = int(request.form['rating'])
        
        # Bewertung in die Datenbank einfügen
        save_rating(user_input, bot_response, rating)
        
        return 'Bewertung erfolgreich gespeichert.'
    except Exception as e:
        return 'Fehler beim Speichern der Bewertung: ' + str(e)


def save_rating(user_input, bot_response, rating):
    try:
        # Verbindung zur Datenbank herstellen
        conn = sqlite3.connect('bewertungen.db')
        cursor = conn.cursor()
        
        # Bewertung in die Tabelle "Bewertungen" einfügen
        cursor.execute("INSERT INTO Bewertungen (Benutzeranfrage, BotAntwort, Bewertung) VALUES (?, ?, ?)", (user_input, bot_response, rating))
        
        # Änderungen bestätigen und Verbindung schließen
        conn.commit()
        conn.close()
    except Exception as e:
        print("Fehler beim Speichern der Bewertung:", str(e))



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

# Verbindung zur Datenbank herstellen
conn = connect_to_database("bewertungen.db")

# Laden der Modelldaten und anderer erforderlicher Dateien
# (Intents, Wörter, Klassen, Modell)
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tensorflow.keras.models.load_model('chatbot_model.h5')


if __name__ == '__main__':
    app.run(debug=True)

print("Du kannst beginnen!")

if __name__ == '__main__':
    app.run(debug=True)


