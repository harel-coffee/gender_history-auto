import sqlite3
from pathlib import Path

from IPython import embed


def initialize_metadata_db():
    """
    Initializes the metadata database

    Fields:
    - doc_id (increments from 0, starting with the earliest document)
    - uri (available for all articles)
    - jstor_id
    - uri
    - doi
    - filename (the local filename assigned to metadata xml and ngrams txts

    - title

    - authors_list (a stringified list of author last_names, first_names, and genders
    - authors_str (a readable string of all author names)
    - authors_gender (gender across all authors, either male, female, mixed, or undetermined
    - authors_number

    - year      (int)
    - date_str  (format: YYYY-MM-DD)

    - journal_title
    - volume
    - issue
    - pages

    :return:
    """

    db = sqlite3.connect(str(Path('data', 'metadata.db')))
    cur = db.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS metadata (
                      article_id        int PRIMARY KEY, 
                      uri               text,
                      doi               text,
                      jstor_id          text,  
                      filename          text,
                      
                      title             text,
                      
                      authors_list      text,
                      authors_str       text,
                      authors_gender    text,
                      authors_number    int,
                      
                      year              int,
                      date_str          str,
                      
                      journal_title     text,
                      volume            int,
                      issue             int,
                      pages             text
                      )
              ''')

def get_article_metadata(jstor_id, cur=None):
    """

    :param jstor_id:
    :param cur: sqlite database cursor or None
    :return:
    """

    if not cur:
        db = sqlite3.connect(str(Path('data', 'metadata.db')))
        cur = db.cursor()

    cur.execute('SELECT * FROM metadata WHERE jstor_id = "{}"'.format(jstor_id))
    article = cur.fetchall()



if __name__ == '__main__':
    initialize_metadata_db()

