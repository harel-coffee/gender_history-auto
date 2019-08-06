import sqlite3
from pathlib import Path

#from scipy.sparse import dok_matrix
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ngrams_parser import load_ngrams_of_article

from IPython import embed

class Corpus:



    def __init__(self):
        self.db = sqlite3.connect(str(Path('data', 'metadata.db')))
        self.start_year = 1970
        self.end_year = 2013

        # initialize metadata
        self.metadata = [article for article in self.metadata_iterator()]


    def metadata_iterator(self):

        self.db.row_factory = sqlite3.Row
        cur = self.db.cursor()
        cur.execute('SELECT * FROM metadata ORDER BY article_id ASC')
        for article in cur.fetchall():
            article_dict = {}
            for key in article.keys():
                article_dict[key] = article[key]
            yield article_dict


    def plot_frequency_chart(self, term, smoothing=1):

        # create a dataframe
        df = pd.DataFrame({
            'year': range(self.start_year, self.end_year + 1),
            # number of times term appears in article written by women
            'count_term_female':    [0] * (self.end_year - self.start_year + 1),
            # total number of terms in articles written by women
            'count_total_female':   [0] * (self.end_year - self.start_year + 1),
            'count_term_male':      [0] * (self.end_year - self.start_year + 1),
            'count_total_male':     [0] * (self.end_year - self.start_year + 1),
        })
        df.index = range(self.start_year, self.end_year+1)

        ngram = len(term.split())

        for article in self.metadata:
            article_gender = article['authors_gender']
            article_ngrams = load_ngrams_of_article(article['filename'], ngram)

            term_count = 0
            for ngrams_term in article_ngrams:
                if ngrams_term.startswith(term):
                    print(ngrams_term, article_ngrams[ngrams_term])
                    term_count += article_ngrams[ngrams_term]

            if article_gender in {'female', 'male'}:
                df.at[article['year'], f'count_term_{article_gender}'] += term_count
                df.at[article['year'], f'count_total_{article_gender}'] += sum(article_ngrams.values())

        freq_female = df['count_term_female'] / df['count_total_female']
        df[f'{term} Frequency (female)'] = freq_female.rolling(window=smoothing*2+1, center=True).mean()
        freq_male = df['count_term_male'] / df['count_total_male']
        df[f'{term} Frequency (male)'] = freq_male.rolling(window=smoothing*2+1, center=True).mean()

        ax = plt.gca()
        df.plot(kind='line', x='year', y=f'{term} Frequency (female)', ax=ax)
        df.plot(kind='line', x='year', y=f'{term} Frequency (male)', ax=ax)
        plt.show()
        embed()


if __name__ == '__main__':
    c = Corpus()
    c.plot_frequency_chart('household')

