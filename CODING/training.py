import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk #naturallanguagetoolkit  (word familien)
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score

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

predicitons = ["Das ist ein Beispiel.", "Die Sonne scheint.", "Hallo, wie geht es dir?"]
labels = ["Das ist ein Beispiel.", "Die Sonne scheint.", "Hallo, wie geht es dir?"]

f1 = f1_score(labels, predicitons, average='weighted')
print("F1-Score", f1)