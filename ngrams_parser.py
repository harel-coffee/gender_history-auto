import os
from pathlib import Path
from collections import Counter
from IPython import embed

import pickle


def create_vocabulary(ngrams=1, vocabulary_size = 100000):

    data_path = Path('data', f'ngram{ngrams}')

    ngram_counter = Counter()

    for root, _, files in os.walk(data_path):
        for file in files:
            print(len(ngram_counter))
            if file.endswith('.txt'):
                with open(Path(root, file)) as f:
                    ngrams_data = f.read()
                for line in ngrams_data.split('\n'):
                    if not line:
                        continue
                    ngram, count = line.split('\t')
                    ngram_counter[ngram] += int(count)

    term_list = []
    for term, count in ngram_counter.most_common(vocabulary_size):
        term_list.append(term)
    term_list = sorted(term_list, reverse=False)

    term_dict = {}
    for idx, term in enumerate(term_list):
        term_dict[idx] = term
    with open(Path('data', f'ngram{ngrams}_vocabulary_dict.pickle'), 'wb') as outfile:
        pickle.dump(term_dict, outfile)


def load_ngrams_of_article(filename, ngrams):
    """
    Returns the ngrams of one filename as a Counter

    >>> ngrams = load_ngrams_of_article('journal-article-10.2307_1842193', 1)
    >>> ngrams['vice']
    76

    :param filename: filename without extension
    :param ngrams: 1-3
    :return:
    """

    if not ngrams in [1,2,3]:
        raise ValueError(f'Ngrams are only available for 1')

    ngram_counter = Counter()
    with open(Path('data', f'ngram{ngrams}', f'{filename}-ngram{ngrams}.txt')) as f:
        ngrams_data = f.read()
    for line in ngrams_data.split('\n'):
        if not line:
            continue
        ngram, count = line.split('\t')
        ngram_counter[ngram] += int(count)

    return ngram_counter

if __name__ == '__main__':
    create_vocabulary()