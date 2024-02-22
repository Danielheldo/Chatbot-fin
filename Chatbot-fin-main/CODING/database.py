import sqlite3

conn = sqlite3.connect('bewertungen.db')

c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS bewertungen
          (antwort TEXT, bewertung INTEGER)''')

conn.commit()
conn.close()

def save_rating(response, rating):
    conn = sqlite3.connect('bewertungen.db')
    c = conn.cursor()
    c.execute("INSERT INTO bewertungen (antwort, bewertung) VALUES (?, ?)",(response, rating))
    conn.commit()
    conn.close()

response = "Hello! How can I help you?"
rating = 4
save_rating(response, rating)