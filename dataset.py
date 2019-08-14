
from pathlib import Path
import pandas as pd

from topics import TOPICS
from IPython import embed

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words

from scipy.sparse import csr_matrix

from collections import Counter
from copy import deepcopy

class Dataset:


    def __init__(self, dataset='dissertations'):

        self.df = pd.read_csv(Path('data', 'doc_with_outcome_and_abstract_stm.csv'),
                         encoding='windows-1252')

        # earliest and latest year in dataset
        self.start_year = min(self.df.ThesisYear)
        self.end_year = max(self.df.ThesisYear)
        # male and female authors
        self.author_gender = 'both'
        # no term filter active. When term filter is active, only documents mentioning the term
        # in the abstract are retained.
        self.term_filter = None
        self.topic_percentile_score_filters = []
        self.vocabulary_set = None

        # creating the tokenized and lemmatized abstract takes time -> do it when the dataset
        # first gets opened and store all tokenized abstracts
        if not 'tokenized_abstract' in self.df.columns:
            wnl = WordNetLemmatizer()
            tokenizer = RegexpTokenizer(r'\b\w\w+\b')
            tokenized_abstracts = []
            for abstract in self.df['Abstract']:
                tokenized_abstract = " ".join([wnl.lemmatize(t) for t in tokenizer.tokenize(abstract)])
                tokenized_abstract = tokenized_abstract.lower()
                tokenized_abstracts.append(tokenized_abstract)
            self.df['tokenized_abstract'] = tokenized_abstracts
            self.df.to_csv(Path('data', 'doc_with_outcome_and_abstract_stm.csv'))

        # 8/14/19: load updated thesis field data from all_data.csv
        if not 'ThesisProQuestFields' in self.df.columns:
            fields_df = pd.read_csv(Path('data', 'all_data.csv'),
                         encoding='ISO-8859-1')

            # all_data.csv and the original csv file have different indexes
            # -> sort by ID and reindex
            fields_df = fields_df.sort_values(by='ID')
            self.df = self.df.sort_values(by='ID')
            fields_df['IDC'] = fields_df['ID']
            self.df['IDC'] = self.df['ID']
            fields_df = fields_df.set_index(keys=['IDC'])
            self.df = self.df.set_index(keys=['IDC'])
            assert np.all(self.df['ID'] == fields_df['ID'])
            for field in ['ThesisProQuestFields', 'ThesisNrcFields',
                          'Inferred NRC Department(NRC Area: SubField)']:
                self.df[field] = fields_df[field]

#            self.df['is_history'] = True
#            self.df['is_history'][self.df['ThesisNrcFields'] == 'Business community'] = False
#            self.df['is_history'][self.df['ThesisNrcFields'] == 'Home economics'] = False


            selector_1 = self.df['ThesisProQuestFields'] != 'Business community'
            selector_2 = self.df['ThesisProQuestFields'] != 'Home economics'
            selector_3 = self.df['ThesisProQuestFields'] != 'Business costs'
            selector_4 = self.df['Abstract'].str.contains(pat='histor', case=False) == True
            self.df['is_history'] = (selector_1 & selector_2 & selector_3 | selector_4)

            print("Eliminating home economics theses from dataset.")
            self.df = self.df[self.df['is_history'] == True]

            self.df.reset_index(inplace=True)


#            self.print_differences_between_filtered_and_unfiltered_datasets()







    def print_differences_between_filtered_and_unfiltered_datasets(self):
        """
        Prints a short analysis between the filtered and unfiltered datasets

        :return:
        """
        out_docs = self.df[self.df['is_history'] == False]
        in_docs = self.df[self.df['is_history'] == True]
        print(f"Currently filtering out {len(out_docs)} non-historical dissertations.")

        difs = []
        topics_list = [f'topic.{id}' for id in range(1, 71)]
        for topic in topics_list:
            difs.append(abs(in_docs[topic].mean() - out_docs[topic].mean()))
        for topic_id in np.argsort(np.array(difs))[::-1][:10]:
            topic_str = topics_list[topic_id]
            print(f'{topic_str}. Dif: {difs[topic_id]}.')

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f'Dissertation Dataset, {self.name_full}'

    @property
    def name(self):
        return f'{self.start_year}-{self.end_year}, {self.author_gender}'

    @property
    def name_full(self):
        n = self.name
        if self.topic_percentile_score_filters:
            for f in self.topic_percentile_score_filters:
                topic_id = int(f['topic'][6:])
                n += f' percentile score between {f["min_percentile_score"]}th and ' \
                     f'{f["max_percentile_score"]}th for {TOPICS[topic_id]["name"]}'
        return n

    @property
    def vocabulary(self):
        """
        Lazy-loaded vocabulary set to check if a word appears in the dataset
        :return: set

        >>> d = Dataset()
        >>> 'family' in d.vocabulary
        True

        >>> 'familoeu' in d.vocabulary
        False

        """
        if not self.vocabulary_set:
            self.vocabulary_set = self.get_vocabulary(exclude_stopwords=False)
        return self.vocabulary_set

    def copy(self):
        c = Dataset()
        attrs = self.__dict__.copy()
        del attrs['df']
        for attr in attrs:
            setattr(c, attr, deepcopy(getattr(self, attr)))

        c.df = self.df.copy(deep=True)
        return c


    def topic_percentile_score_filter(self, topic, min_percentile_score=0, max_percentile_score=100):
        """
        Filter the dataset such that it only contains documents that score between
        min_percentile_score and max_percentile_score for a topic

        :param topic: topic id (int) or str ('topic.XX')
        :param min_percentile_score: 0 to 100
        :param max_percentile_score: 0 to 100
        :return:

        >>> d = Dataset()
        >>> len(d)
        23246

        # select dissertations that score in the top decile (strongest) for the gender topic (28)
        >>> d.topic_percentile_score_filter(topic=28, min_percentile_score=90)
        >>> print(len(d), min(d.df['percentile_score_topic.28']))
        2325 90.0

        # select 50th-70th decile
        >>> d2 = Dataset()
        >>> d2.topic_percentile_score_filter('topic.28', min_percentile_score=50, max_percentile_score=70)
        >>> len(d2)
        6974


        """

        if isinstance(topic, int):
            topic = f'topic.{topic}'
        if not f'percentile_score_{topic}' in self.df.columns:
            self.df[f'percentile_score_{topic}'] = self.df[topic].rank(pct=True) * 100 // 10 * 10
        self.df = self.df[self.df[f'percentile_score_{topic}'] >= min_percentile_score]
        self.df = self.df[self.df[f'percentile_score_{topic}'] <= max_percentile_score]
        self.topic_percentile_score_filters.append({
            'topic': topic,
            'min_percentile_score': min_percentile_score,
            'max_percentile_score': max_percentile_score
        })


    def filter(self, start_year=None, end_year=None, author_gender=None,
               term_filter=None):
        """

        :param start_year:    (int between 1976 and 2015)  earliest year to include
        :param end_year:      (int between 1976 and 2015)  last year to include
        :param author_gender: (male or female)             only include male or female authors
        :param term_filter:   string or raw string         only include documents mentioning term
        :return:

        >>> d = Dataset()
        >>> len(d)
        23246

        # filter for years (inclusive) between 1985 and 1995
        >>> d.filter(start_year=1985, end_year=1995)
        >>> len(d)
        6978

        # filter by author gender
        >>> d.filter(author_gender='male')
        >>> len(d)
        4195

        # filter by term or regex
        # regex example: r'\bgender\b'
        >>> d.filter(term_filter='gender')
        >>> len(d)
        136

        """

        df = self.df

        if start_year:
            df = df[df.ThesisYear >= start_year]
            self.start_year = start_year
        if end_year:
            df = df[df.ThesisYear <= end_year]
            self.end_year = end_year
        if author_gender:
            if not author_gender in ['male', 'female']:
                raise ValueError(f'Author gender needs to be male or female but not {author_gender}')
            df = df[df.AdviseeGender == author_gender]
            self.author_gender = author_gender

        if term_filter:
            df = df[df['tokenized_abstract'].str.contains(pat=term_filter, regex=True) == True]
            self.term_filter = term_filter

        self.df = df
        return self

    def get_vocabulary(self, exclude_stopwords=True, max_terms=None, min_appearances=None):
        """
        Returns a list of all terms that appear in the tokenized_abstract column

        :return: list

        # stop words are excluded by default
        >>> d = Dataset()
        >>> len(d.get_vocabulary(exclude_stopwords=True))
        176458
        >>> d.get_vocabulary().count('are')
        0

        # you can also limit the number of terms and require a minimum number of appearances
        >>> voc = d.get_vocabulary(max_terms=1000, min_appearances=2)
        >>> len(voc)
        1000

        """
        vocabulary = Counter()
        for abstract in self.df['tokenized_abstract']:
            for word in abstract.split():
                vocabulary[word] += 1

        if exclude_stopwords:
            for term in stop_words.ENGLISH_STOP_WORDS.union({'wa', 'ha'}):
                try:
                    del vocabulary[term]
                except KeyError:
                    pass

        vocab_list = []
        if not max_terms:
            max_terms = 1000000
        for word, count in vocabulary.most_common(max_terms):
            if min_appearances and count < min_appearances:
                continue
            else:
                vocab_list.append(word)

        return sorted(vocab_list)


    def get_document_topic_matrix(self):
        """
        Returns a document-topic sparse matrix. each row represents a document and each column a
        topic
        Note: all topics are one-off because there is no topic.0, i.e. topic.1 is stored in the
        0th column.

        >>> d = Dataset()
        >>> dtm = d.get_document_topic_matrix()
        >>> dtm.shape
        (23246, 70)

        :return: csr_matrix
        """

        topics_str_list = [f'topic.{i}' for i in range(1, 71)]
        dtm = csr_matrix(self.df[topics_str_list].to_numpy())
        return dtm


    def get_document_term_matrix(self, vocabulary, store_in_df=False):
        """

        Returns a document-term sparse matrix. each row represents a document and each column a
        topic

        If store_in_df is selected, the dtm will be stored in the dataframe, i.e. every term
        is stored as a row in the dataframe (useful for quick term selection)

        :param vocabulary: list
        :param store_in_df: bool
        :return:

        >>> d = Dataset()
        >>> vocabulary = d.get_vocabulary(max_terms=10000)
        >>> dtm = d.get_document_term_matrix(vocabulary)
        >>> dtm.shape
        (23246, 10000)

        """

        vectorizer = CountVectorizer(vocabulary=vocabulary)
        dtm = vectorizer.fit_transform(self.df['tokenized_abstract'].to_list())

        if store_in_df:
            dtm_df = pd.DataFrame(dtm.toarray(), columns=vocabulary)
            self.df = pd.concat([self.df, dtm_df], axis=1)
        else:
            return dtm





if __name__ == '__main__':
    d = Dataset('dissertations')
    dtm = d.get_document_term_matrix(d.get_vocabulary(max_terms=10000), store_in_df=True)

    embed()

