import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk #naturallanguagetoolkit  (word familien)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import sqlite3

def create_table_if_not_exists():
    try:
        conn = sqlite3.connect('bewertungen.db')
        cursor = conn.cursor()
        
        # Überprüfen, ob die Tabelle bereits vorhanden ist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Bewertungen'")
        table_exists = cursor.fetchone()
        
        # Wenn die Tabelle nicht vorhanden ist, erstelle sie
        if not table_exists:
            cursor.execute('''CREATE TABLE Bewertungen 
                              (ID INTEGER PRIMARY KEY AUTOINCREMENT, 
                              Benutzeranfrage TEXT, 
                              BotAntwort TEXT, 
                              Bewertung INTEGER)''')
            conn.commit()
            print("Tabelle 'Bewertungen' erfolgreich erstellt.")
        
        conn.close()
    except Exception as e:
        print("Fehler beim Erstellen der Tabelle:", str(e))


create_table_if_not_exists()



lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())  #unser Wörterbuch

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

#print(words)

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Laden der intents.json-Datei
with open('intents.json') as file:
    intents = json.load(file)

# Extrahieren der Trainingsdaten und zugehörigen Labels aus den intents
train_texts = []
train_labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        train_texts.append(pattern)
        train_labels.append(intent['tag'])

# Synonyme hinzufügen
def add_synonyms(text, n=1):
    words = nltk.word_tokenize(text)
    synonyms = []
    for word in words:
        syns = wordnet.synsets(word)
        if syns:
            syn_word = syns[0].lemmas()[0].name()
            if syn_word != word:
                synonyms.append(syn_word)
    return " ".join(words + synonyms[:n])

train_texts_with_synonyms = [add_synonyms(text) for text in train_texts]

# Initialisieren und Anpassen des TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(train_texts_with_synonyms)

# Vektorisierung der Trainingsdaten
train_vectors = tfidf_vectorizer.transform(train_texts_with_synonyms)

# Initialisieren und Anpassen des Klassifikationsmodells (z. B. Multinomial Naive Bayes)
classifier = MultinomialNB()
classifier.fit(train_vectors, train_labels)

# Pipeline erstellen
model = make_pipeline(tfidf_vectorizer, classifier)

# Trainieren des Modells
model.fit(train_texts_with_synonyms, train_labels)

#Machinelerning Teil:

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(len(words), 128, input_length=len(trainX[0])))
model.add(tf.keras.layers.SimpleRNN(128))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=100, batch_size=12, verbose=1)
model.save('chatbot_model.h5')
print('Done')

_, accuracy = model.evaluate(trainX, trainY)
print('Trainingsgenauigkeit: %.2f' % (accuracy*100))

