
import random
random.seed(0)

import pickle

from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline, BSpline

from scipy.interpolate import interp1d

import matplotlib.patches as mpatches
from pathlib import Path
import pandas as pd

from IPython import embed
import html

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from scipy.sparse import csr_matrix

from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gender_history.utilities import WORD_SPLIT_REGEX, BASE_PATH, STOP_WORDS

import seaborn as sns

import re
import hashlib




class Dataset:

    def __init__(self, **kwargs):

        # earliest and latest year in dataset
        self.start_year = min(self.df.m_year)
        self.end_year = max(self.df.m_year)
        # male and female authors
        self.author_gender = 'both'
        # no term filter active. When term filter is active, only documents mentioning the term
        # in the abstract are retained.
        self.term_filter = None
        self.institution_filter = None
        self.advisor_gender_filter = None
        self.descendants_filter = None
        self.topic_score_filters = []
        self.vocabulary_set = None

        self._df_with_equal_samples_per_5_year_period = None

        if not kwargs.get('skip_loading_topics'):
            self.topics = self.load_topic_data()

        self.store_aggregate_approach_and_geographical_info_in_df()


    def __len__(self):
        return len(self.df)

    def __hash__(self) -> str:
        """
        We use hash to create a unique identifier for a configuration. Hence, this uses md5 not
        hash() because it has to be deterministic. ( hash() uses a random seed).
        :return: str
        """
        string_to_hash = (
            f'{self.dataset_type}{self.start_year}{self.end_year}{self.author_gender}'
            f'{self.term_filter}{self.institution_filter}{self.advisor_gender_filter}'
            f'{self.descendants_filter}{self.topic_score_filters}'
            f'{self.use_equal_samples_dataset}'
        )
        md5 = hashlib.md5(string_to_hash.encode('utf8')).hexdigest()
        return md5

    def create_df_1000_texts_per_5_year_interval(self):
        """
        Creates a df with a randomly selected sample of 1000 texts per 5 year period.

        :return:
        """
        df_with_equal_samples = None

        for start_year in range(1950, 2014, 5):
            print(start_year)
            date_docs = self.df[(start_year <= self.df.m_year) & (self.df.m_year < start_year + 5)]
            print(f'Docs from {start_year} to {start_year + 4}: {len(date_docs)}')

            date_docs_sample = date_docs.sample(n=2000, replace=True, random_state=0)
            if df_with_equal_samples is None:
                df_with_equal_samples = date_docs_sample
            else:
                df_with_equal_samples = df_with_equal_samples.append(date_docs_sample)

        # for start_year in range(1950, 2014, 5):
        #     print(start_year)
        #     for gender in ['female', 'male']:
        #         date_docs = self.df[(start_year <= self.df.m_year) &
        #                             (self.df.m_year < start_year + 5) &
        #                             (self.df.m_author_genders == gender)]
        #         print(f'{gender} docs from {start_year} to {start_year + 4}: {len(date_docs)}')
        #         date_docs_sample = date_docs.sample(n=5000, replace=True, random_state=0)
        #         if df_with_equal_samples is None:
        #             df_with_equal_samples = date_docs_sample
        #         else:
        #             df_with_equal_samples = df_with_equal_samples.append(date_docs_sample)

        df_with_equal_samples.reset_index(drop=True, inplace=True)

        return df_with_equal_samples


#     def __repr__(self):
#         return f'Dissertation Dataset, {self.name_full}'
#
#     @property
#     def name(self):
#         return f'{self.start_year}-{self.end_year}, {self.author_gender}'
#
#     @property
#     def name_full(self):
#         n = self.name
#         if self.institution_filter:
#             n += f' {self.institution_filter}'
#         if self.advisor_gender_filter:
#             n += f' {self.advisor_gender_filter}'
#         if self.topic_percentile_score_filters:
#             for f in self.topic_percentile_score_filters:
#                 topic_id = int(f['topic'][6:])
#                 n += f' percentile score between {f["min_percentile_score"]}th and ' \
#                      f'{f["max_percentile_score"]}th for {self.topics[topic_id]["name"]}'
#         return n
#
#     @property
#     def vocabulary(self):
#         """
#         Lazy-loaded vocabulary set to check if a word appears in the dataset
#         :return: set
#
#         >>> d = Dataset()
#         >>> 'family' in d.vocabulary
#         True
#
#         >>> 'familoeu' in d.vocabulary
#         False
#
#         """
#         if not self.vocabulary_set:
#             self.vocabulary_set = self.get_vocabulary(exclude_stopwords=False)
#         return self.vocabulary_set
#
    def copy(self):

        return deepcopy(self)


    def load_topic_data(self):
        """
        Return a dict that maps from topic id to a dict with names, gen/spec approaches,
        gen/spec area, and terms prob/frex for all 90 topics

        :param file_path: Path
        :return: dict
        """
        topics = {}
        df = pd.read_csv(Path(BASE_PATH, 'data', 'journal_csv',
                              'topic_titles_and_terms.csv'),
                         encoding='utf-8')


        overall_freq_score_path = Path(BASE_PATH, 'data', 'journal_csv',
                                                'overall_freq_scores.pickle')
        if overall_freq_score_path.exists():
            with open(overall_freq_score_path, 'r') as infile:
                overall_freq_scores = pickle.load(infile)
        else:
            pass

        for _, row in df.iterrows():

            topic_id = int(row['ID'])
            name_descriptive = row['Name (descriptive)']
            name_as_field = row['Name as Field (subfield)']
            gen_approach = row['General Approach']
            spec_approach = row['Specific Approach']
            gen_area = row['General Area']
            spec_area = row['Specific Area']

            terms_prob = row['Terms Prob'].split(", ")
            terms_frex = row['Terms Frex'].split(", ")
            topics[topic_id] = {
                'name': name_descriptive,
                'name_as_field': name_as_field,
                'gen_approach': gen_approach,
                'spec_approach': spec_approach,
                'gen_area': gen_area,
                'spec_area': spec_area,
                'terms_prob': terms_prob,
                'terms_frex': terms_frex,
                'terms_both': terms_prob + terms_frex
            }
        return topics

    def store_aggregate_approach_and_geographical_info_in_df(self):
        """
        To aggregate, e.g. the multiple political history general approach topics,
        we calculate the average for each article.
        e.g. if topics 30 and 55 are the political history topics, we set
        df['gen_approach_Political History'] to the average of the columns topic.30 and topic.55.

        TODO: expand for specific approaches and geographical areas

        :return:
        """

        gen_approaches_to_id = defaultdict(set)
        for topic_id, topic in self.topics.items():
            if topic['gen_approach'] == 'None' or not isinstance(topic['gen_approach'], str):
                continue
            gen_approaches = topic['gen_approach']
            for approach in gen_approaches.split(';'):
                gen_approaches_to_id[approach.strip()].add(topic_id)

        sum_of_means = 0
        for gen_approach in gen_approaches_to_id:
            selectors = [f'topic.{i}' for i in gen_approaches_to_id[gen_approach]]
            self.df[f'gen_approach_{gen_approach}'] = self.df[selectors].sum(axis=1)
            m = self.df[f'gen_approach_{gen_approach}'].mean()
            sum_of_means += m




    def topic_score_filter(self, topic_id, min_percentile_score=0, max_percentile_score=100,
                           min_topic_weight=0, max_topic_weight=1):
        """
        Filter the dataset such that it only contains documents that score between
        min_percentile_score and max_percentile_score for a topic

        :param topic_id: topic id (int) or str ('topic.XX')
        :param min_percentile_score: 0 to 100
        :param max_percentile_score: 0 to 100
        :return:

        >>> d = Dataset()
        >>> len(d)
        21634

        # select dissertations that score in the top decile (strongest) for the gender topic (28)
        >>> d.topic_score_filter(topic_id=28, min_percentile_score=90)
        >>> print(len(d), min(d.df['percentile_score_topic.28']))
        2164 90.0

        # select 50th-70th decile
        >>> d2 = Dataset()
        >>> d2.topic_score_filter('topic.28', min_percentile_score=50, max_percentile_score=70)
        >>> len(d2)
        6491

        # filters can be combined
        >>> d3 = Dataset()
        >>> d3.topic_score_filter(14, min_percentile_score=80)
        >>> d3.topic_score_filter(28, min_percentile_score=80)
        >>> len(d3)
        866


        """
        if not isinstance(topic_id, str):
            topic_id = f'topic.{topic_id}'

        if not f'percentile_score_{topic_id}' in self.df.columns:
            print("adding all percentile scores")
            # add all of the topics at once because if we filter topics twice, the ranks would be
            # influenced by the first selection
            for i in range(1, 91):
                t = f'topic.{i}'
                self.df[f'percentile_score_{t}'] = self.df[t].rank(pct=True) * 100

        if min_percentile_score > 0 or max_percentile_score < 100:

            self.df = self.df[self.df[f'percentile_score_{topic_id}'] >= min_percentile_score]
            self.df = self.df[self.df[f'percentile_score_{topic_id}'] <= max_percentile_score]
            self.df = self.df.reset_index(drop=True)

            self.topic_score_filters.append({
                'topic': topic_id,
                'min_percentile_score': min_percentile_score,
                'max_percentile_score': max_percentile_score
            })

        if min_topic_weight > 0:
            self.df = self.df[self.df[topic_id] > min_topic_weight]
            self.df = self.df.reset_index(drop=True)
            self.topic_score_filters.append({
                'topic': topic_id,
                'min_weight': min_topic_weight
            })

        if max_topic_weight < 1:
            self.df = self.df[self.df[topic_id] < max_topic_weight]
            self.df = self.df.reset_index(drop=True)
            self.topic_score_filters.append({
                'topic': topic_id,
                'max_weight': max_topic_weight
            })

        return self


    def filter(self, start_year=None, end_year=None, author_gender=None,
               term_filter=None, institution_filter=None, advisor_gender=None,
               has_descendants=None):
        """

        :param start_year:    (int between 1976 and 2015)  earliest year to include
        :param end_year:      (int between 1976 and 2015)  last year to include
        :param author_gender: (male or female)             only include male or female authors
        :param term_filter:   string or raw string         only include documents mentioning term
        :return:

        >>> d = Dataset()
        >>> len(d)
        21634

        # filter for years (inclusive) between 1985 and 1995
        >>> d.filter(start_year=1985, end_year=1995)
        >>> len(d)
        6532

        # filter by author gender
        >>> d.filter(author_gender='male')
        >>> len(d)
        4055

        # filter by advisor gender
        >>> d.filter(advisor_gender='female')
        >>> len(d)
        220

        # filter by term or regex
        # regex example: r'\bgender\b'
        >>> d.filter(term_filter={'term':'gender', 'min_count':'5'})
        >>> len(d)
        13

        # filter by institution
        >>> d = Dataset()
        >>> d.filter(institution_filter='harvard')
        >>> len(d)
        778

        >>> d = Dataset()
        >>> d.filter(institution_filter='not_harvard')
        >>> len(d)
        20856

        >>> d = Dataset()
        >>> d.filter(has_descendants=True)
        >>> len(d)
        1583

        """

        df = self.df

        if start_year:
            df = df[df.m_year >= start_year]
            self.start_year = start_year
        if end_year:
            df = df[df.m_year <= end_year]
            self.end_year = end_year
        if author_gender:
            if not author_gender in ['male', 'female', 'mixed']:
                raise ValueError(f'Author gender needs to be male/female/mixed but not {author_gender}')
            df = df[df.m_author_genders == author_gender]
            self.author_gender = author_gender

        if advisor_gender:
            if not advisor_gender in ['male', 'female', 'unknown']:
                raise ValueError(f'Author gender needs to be male or female but not {advisor_gender}')
            if self.dataset_type == 'journals':
                raise ValueError('advisor gender filter is only available for dissertations')

            df = df[df.m_advisor_gender == advisor_gender]
            self.advisor_gender = advisor_gender


        if term_filter:


            term = term_filter['term']
            if 'min_count' in term_filter:
                min_count = term_filter['min_count']
            else:
                min_count = 0
            if 'max_count' in term_filter:
                max_count = term_filter['max_count']
            else:
                max_count = 10000000

            pattern = r'\b' + term + r'\b'

            # store count of search term
            df['m_tf_count'] = df.m_text.str.count(pat=pattern)

            df = df[(df.m_tf_count >= min_count) & (df.m_tf_count <= max_count)]

            self.term_filter = term_filter


        if institution_filter:
            if self.dataset_type == 'journals':
                raise ValueError('institution filter is only available for dissertations')
            if institution_filter.startswith('not_'):
                institution_filter = institution_filter[4:]
                df = df[df['m_ThesisInstitution'].str.contains(institution_filter, case=False) == False]
            else:
                df = df[df['m_ThesisInstitution'].str.contains(institution_filter, case=False) == True]
            self.institution_filter = institution_filter

        if has_descendants == True:
            if self.dataset_type == 'journals':
                raise ValueError('descendant filter is only available for dissertations')
            df = df[df.m_descendants > 0]
            self.descendants_filter = True

        if has_descendants == False:
            if self.dataset_type == 'journals':
                raise ValueError('descendant filter is only available for dissertations')
            df = df[df.m_descendants == 0]
            self.descendants_filter = False

        self.df = df.reset_index(drop=True)
        return self


    def get_document_topic_matrix(self):
        """
        Returns a document-topic sparse matrix. each row represents a document and each column a
        topic
        Note: all topics are one-off because there is no topic.0, i.e. topic.1 is stored in the
        0th column.

        Vocabulary are the columns to select

        >>> d = Dataset()
        >>> vocabulary = [f'topic.{i}' for i in range(1, 101)]
        >>> dtm = d.get_document_topic_matrix()
        >>> dtm.shape
        (23246, 70)

        :return: csr_matrix
        """

        selectors = [f'topic.{i}' for i in range(1, 91)]
        dtm = csr_matrix(self.df[selectors].to_numpy())
        return dtm


    def get_default_vocabulary(self, no_terms=1000):
        """
        Loads a default vocabulary of no_terms, i.e. a vocabulary generated with a journal dataset
        without any filters

        :param no_terms:
        :return:
        """

        vocabulary_path = Path(BASE_PATH, 'data', 'dtms',
                               f'vocabulary_{no_terms}.pickle')
        if vocabulary_path.exists():
            with open(vocabulary_path, 'rb') as infile:
                return pickle.load(infile)

        else:
            print(f"Generating new standard vocabulary with {no_terms} terms.")
            from gender_history.datasets.dataset_journals import JournalsDataset
            jd = JournalsDataset()
            _, vocabulary = jd.get_vocabulary_and_document_term_matrix(max_features=no_terms,)
            with open(vocabulary_path, 'wb') as outfile:
                pickle.dump(vocabulary, outfile)
            return vocabulary


    def get_percentage_of_stopwords(self):


        vectorizer_no_stopwords = CountVectorizer(max_features=1000000)
        dtm_no_stopwords = vectorizer_no_stopwords.fit_transform(self.df['m_text'].to_list())

        vectorizer_stopwords = CountVectorizer(max_features=999700, stop_words=STOP_WORDS)
        dtm_stopwords = vectorizer_stopwords.fit_transform(self.df['m_text'].tolist())

        embed()

    def get_vocabulary_and_document_term_matrix(self, vocabulary=None, max_features=100000,
                                 use_frequencies=False, ngram_range=(1, 1), split_apostrophes=True,
                                 store_in_df=False, exclude_stop_words=False):
        """

        Returns a document-term sparse matrix. each row represents a document and each column a
        topic

        If store_in_df is selected, the dtm will be stored in the dataframe, i.e. every term
        is stored as a row in the dataframe (useful for quick term selection)

        :param vocabulary: list
        :param store_in_df: bool
        :return:

        >>> d = Dataset()
        >>> dtm, vocabulary = d.get_vocabulary_and_document_term_matrix(max_features=10000)
        >>> dtm.shape
        (23246, 10000)

        """
        dtm = None
        dtm_path = None

        if isinstance(vocabulary, list) and len(vocabulary) % 1000 == 0:
            dtm_path = Path(BASE_PATH, 'data', 'dtms',
                            f'dtm_{self.__hash__()}_{len(vocabulary)}_{use_frequencies}_'
                            f'_{max_features}_{ngram_range}'
                            f'{split_apostrophes}.pickle')
            vocabulary_path = Path(BASE_PATH, 'data', 'dtms',
                            f'vocabulary_{self.__hash__()}_{len(vocabulary)}_{use_frequencies}_'
                            f'{max_features}_{ngram_range}'
                            f'{split_apostrophes}.pickle')
            if False:
                with open(dtm_path, 'rb') as infile:
                    dtm = pickle.load(infile)
                with open(vocabulary_path, 'rb') as infile:
                    vocabulary = pickle.load(infile)

        if dtm is None:
            print(f"Generating document term matrix with {max_features} features.")
            token_pattern = WORD_SPLIT_REGEX
            if split_apostrophes:
                # token_pattern = r'\b\w\w+\b'
                token_pattern = r'\b[a-z][a-z]+\b'
            stop_words = None
            if exclude_stop_words:
                stop_words = STOP_WORDS
            vectorizer = CountVectorizer(vocabulary=vocabulary, token_pattern=token_pattern,
                                         ngram_range=ngram_range,
                                         max_features=max_features,
                                         stop_words=stop_words)
                                         # stop_words='english')
            dtm = vectorizer.fit_transform(self.df['m_text'].to_list())
            # for frequencies, don't use TfidfVectorizer because not including stop words will
            # inflate the frequencies of the non-stopword-terms
            if use_frequencies:
                text_len_arr = np.array(self.df['m_text_len'])
                dtm_freq = (dtm.T / text_len_arr).T
                dtm = csr_matrix(dtm_freq)


            vocabulary = vectorizer.get_feature_names()

            if dtm_path:
                with open(dtm_path, 'wb') as outfile:
                    pickle.dump(dtm, outfile)
                with open(vocabulary_path, 'wb') as outfile:
                    pickle.dump(vocabulary, outfile)

        if store_in_df:
            print("Storing dtm in df")
            dtm_df = pd.DataFrame(dtm.toarray(), columns=vocabulary)
            self.df = pd.concat([self.df, dtm_df], axis=1)
        else:
            return dtm, vocabulary



if __name__ == '__main__':



    pass


