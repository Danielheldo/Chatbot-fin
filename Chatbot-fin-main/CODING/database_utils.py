import sqlite3

def connect_to_database(database_name):
    """
    Verbindung zur SQLite-Datenbank herstellen und die Verbindungsinstanz zurückgeben.
    """
    conn = sqlite3.connect(database_name)
    return conn

def save_rating(conn, response, rating):
    """
    Eine Bewertung (Antwort und Bewertung) in der Datenbank speichern.
    """
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ratings (response, rating) VALUES (?, ?)", (response, rating))
    conn.commit()

