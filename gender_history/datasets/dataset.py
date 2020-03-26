
import random
random.seed(0)

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words

from scipy.sparse import csr_matrix

from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns


class Dataset:

    def __init__(self):

        # earliest and latest year in dataset
        self.start_year = min(self.df.year)
        self.end_year = max(self.df.year)
        # male and female authors
        self.author_gender = 'both'
        # no term filter active. When term filter is active, only documents mentioning the term
        # in the abstract are retained.
        self.term_filter = None
        self.institution_filter = None
        self.advisor_gender_filter = None
        self.descendants_filter = None
        self.topic_percentile_score_filters = []
        self.vocabulary_set = None

        self._df_with_equal_samples_per_5_year_period = None


    def __len__(self):
        return len(self.df)

    def create_df_1000_texts_per_5_year_interval(self):
        """
        Creates a df with a randomly selected sample of 1000 texts per 5 year period.

        :return:
        """
        df_with_equal_samples = None

        # for start_year in range(1950, 2014, 5):
        #     print(start_year)
        #     date_docs = self.df[(start_year <= self.df.year) & (self.df.year < start_year + 5)]
        #     print(f'Docs from {start_year} to {start_year + 4}: {len(date_docs)}')
        #
        #     date_docs_sample = date_docs.sample(n=1000, replace=True, random_state=0)
        #     if df_with_equal_samples is None:
        #         df_with_equal_samples = date_docs_sample
        #     else:
        #         df_with_equal_samples = df_with_equal_samples.append(date_docs_sample)


        for start_year in range(1950, 2014, 5):
            print(start_year)
            for gender in ['female', 'male']:
                date_docs = self.df[(start_year <= self.df.year) &
                                    (self.df.year < start_year + 5) &
                                    (self.df.author_genders == gender)]
                print(f'{gender} docs from {start_year} to {start_year + 4}: {len(date_docs)}')
                date_docs_sample = date_docs.sample(n=500, replace=True, random_state=0)
                if df_with_equal_samples is None:
                    df_with_equal_samples = date_docs_sample
                else:
                    df_with_equal_samples = df_with_equal_samples.append(date_docs_sample)

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


    def load_topic_data(self, file_path):
        """
        Return a dict that maps from topic id to a dict with names, gen/spec approaches,
        gen/spec area, and terms prob/frex for all 90 topics

        :param file_path: Path
        :return: dict
        """
        topics = {}
        df = pd.read_csv(file_path, encoding='utf-8')

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
            if topic['gen_approach'] == '' or not isinstance(topic['gen_approach'], str):
                continue
            gen_approaches_to_id[topic['gen_approach']].add(topic_id)

        sum_of_means = 0
        for gen_approach in gen_approaches_to_id:
            selectors = [f'topic.{i}' for i in gen_approaches_to_id[gen_approach]]
            try:
                self.df[f'gen_approach_{gen_approach}'] = self.df[selectors].sum(axis=1)
            except:
                embed()

            m = self.df[f'gen_approach_{gen_approach}'].mean()
            sum_of_means += m



#     def topic_percentile_score_filter(self, topic, min_percentile_score=0, max_percentile_score=100):
#         """
#         Filter the dataset such that it only contains documents that score between
#         min_percentile_score and max_percentile_score for a topic
#
#         :param topic: topic id (int) or str ('topic.XX')
#         :param min_percentile_score: 0 to 100
#         :param max_percentile_score: 0 to 100
#         :return:
#
#         >>> d = Dataset()
#         >>> len(d)
#         21634
#
#         # select dissertations that score in the top decile (strongest) for the gender topic (28)
#         >>> d.topic_percentile_score_filter(topic=28, min_percentile_score=90)
#         >>> print(len(d), min(d.df['percentile_score_topic.28']))
#         2164 90.0
#
#         # select 50th-70th decile
#         >>> d2 = Dataset()
#         >>> d2.topic_percentile_score_filter('topic.28', min_percentile_score=50, max_percentile_score=70)
#         >>> len(d2)
#         6491
#
#         # filters can be combined
#         >>> d3 = Dataset()
#         >>> d3.topic_percentile_score_filter(14, min_percentile_score=80)
#         >>> d3.topic_percentile_score_filter(28, min_percentile_score=80)
#         >>> len(d3)
#         866
#
#
#         """
#         if not isinstance(topic, str):
#             topic = f'topic.{topic}'
#
#         if not f'percentile_score_{topic}' in self.df.columns:
#             # add all of the topics at once because if we filter topics twice, the ranks would be
#             # influenced by the first selection
#             for i in range(1, 71):
#                 t = f'topic.{i}'
#                 self.df[f'percentile_score_{t}'] = self.df[t].rank(pct=True) * 100 // 10 * 10
#         self.df = self.df[self.df[f'percentile_score_{topic}'] >= min_percentile_score]
#         self.df = self.df[self.df[f'percentile_score_{topic}'] <= max_percentile_score]
#         self.df = self.df.reset_index()
#         self.topic_percentile_score_filters.append({
#             'topic': topic,
#             'min_percentile_score': min_percentile_score,
#             'max_percentile_score': max_percentile_score
#         })
#
#
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
        >>> d.filter(term_filter='gender')
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
            df = df[df.year >= start_year]
            self.start_year = start_year
        if end_year:
            df = df[df.year <= end_year]
            self.end_year = end_year
        if author_gender:
            if not author_gender in ['male', 'female', 'mixed']:
                raise ValueError(f'Author gender needs to be male/female/mixed but not {author_gender}')
            df = df[df.author_genders == author_gender]
            self.author_gender = author_gender

        if advisor_gender:
            if not advisor_gender in ['male', 'female', 'unknown']:
                raise ValueError(f'Author gender needs to be male or female but not {advisor_gender}')
            if self.dataset_type == 'journals':
                raise ValueError('advisor gender filter is only available for dissertations')

            df = df[df.AdvisorGender == advisor_gender]
            self.advisor_gender = advisor_gender


        if term_filter:
            if term_filter.startswith('not_'):
                term_filter = term_filter[4:]
                df = df[df['text'].str.contains(pat=term_filter, regex=True) == False]
            else:
                df = df[df['text'].str.contains(pat=term_filter, regex=True) == True]
            self.term_filter = term_filter

        if institution_filter:
            if self.dataset_type == 'journals':
                raise ValueError('institution filter is only available for dissertations')
            if institution_filter.startswith('not_'):
                institution_filter = institution_filter[4:]
                df = df[df['ThesisInstitution'].str.contains(institution_filter, case=False) == False]
            else:
                df = df[df['ThesisInstitution'].str.contains(institution_filter, case=False) == True]
            self.institution_filter = institution_filter

        if has_descendants == True:
            if self.dataset_type == 'journals':
                raise ValueError('descendant filter is only available for dissertations')
            df = df[df.AnyDescendants == 1]
            self.descendants_filter = True

        if has_descendants == False:
            if self.dataset_type == 'journals':
                raise ValueError('descendant filter is only available for dissertations')
            df = df[df.AnyDescendants == 0]
            self.descendants_filter = False

        self.df = df.reset_index()
        return self


    def get_vocabulary(self, exclude_stopwords=True, max_terms=None, min_appearances=None,
                       include_2grams=False):
        """
        Returns a list of all terms that appear in the text column

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

        if not max_terms:
            max_terms = 1000000

        vocabulary = Counter()
        for abstract in self.df['text']:
            a = abstract.split()
            for idx, word in enumerate(a):
                vocabulary[word] += 1
                if include_2grams:
                    try:
                        gram = '{} {}'.format(word, a[idx+1])
                        vocabulary[gram] += 1
                    except IndexError:
                        pass

        if exclude_stopwords:
            clean_vocabulary = Counter()
            stopwords = stop_words.ENGLISH_STOP_WORDS.union({'wa', 'ha',
                                                     'óé', 'dotbelow', 'cyrillic'})

            for ngram in vocabulary:
                valid = True
                for term in ngram.split():
                    if term in stopwords:
                        valid = False
                if valid:
                    clean_vocabulary[ngram] = vocabulary[ngram]
            vocabulary = clean_vocabulary


        vocab_list = []
        for word, count in vocabulary.most_common(max_terms):
            if min_appearances and count < min_appearances:
                continue
            else:
                vocab_list.append(word)

        return sorted(vocab_list)


    def get_document_topic_matrix(self, vocabulary):
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

        dtm = csr_matrix(self.df[vocabulary].to_numpy())
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
        dtm = vectorizer.fit_transform(self.df['text'].to_list())

        if store_in_df:
            dtm_df = pd.DataFrame(dtm.toarray(), columns=vocabulary)
            self.df = pd.concat([self.df, dtm_df], axis=1)
        else:
            return dtm

#     def print_dissertations_mentioning_terms_or_topics(self, terms, no_dissertations=5):
#         """
#         print dissertations that mention specific terms or topics,
#         can be weighted or unweighted
#
#         :param terms: list or dict
#         :param no_dissertations: number of dissertations to print, default: 5
#
#         >>> d = Dataset()
#         >>> terms = ['woman', 'female', 'women', 'feminist', 'gender']
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
#         2014 Author: female  Advisor: female  Authors, Activists, Apostles: Women's Leadership in the New Christian Right
#         2006 Author: female  Advisor: male    Women on the march: Gender and anti-fascism in American communism, 1935--1939
#
#         # by default, each term has weight=1. However, terms can also be a dict of term weights
#         >>> terms = {'nurse': 1, 'drugs': 10}
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
#         1997 Author: female  Advisor: male    Regulating beauty: Cosmetics in American culture from the 1906 Pure Food and Drugs Act to the 1938 Food, Drug and Cosmetic Act
#         1994 Author: female  Advisor: female  G.I. nurses at war: Gender and professionalization in the Army Nurse Corps during World War II
#
#         # topics also work:
#         >>> terms = ['topic.28']
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
#         1989 Author: female  Advisor: unknown Day nurseries and wage-earning mothers in the United States, 1890-1930
#         1988 Author: female  Advisor: female  "Women adrift" and "urban pioneers": Self-supporting working women in America, 1880-1930
#
#         >>> terms = ['gay', 'homosexual', 'homosexuality', 'masculinity']
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=20)
#
#         """
#
#         # if input is a list, give each term weight 1
#         if isinstance(terms, list):
#             terms = {t: 1 for t in terms}
#
#         out = ''
#
#         if list(terms.keys())[0].startswith('topic.'):
#             self.df['dissertation_weights'] = 0
#             for topic, weight in terms.items():
#                 self.df['dissertation_weights'] += weight * self.df[topic]
#             for _, row in self.df.sort_values(by='dissertation_weights', ascending=False)[:no_dissertations].iterrows():
#                 out += ('\n{} Author: {:7s} Advisor: {:7s} {}'.format(
#                     row['Year'], row['author_genders'],
#                     row['AdvisorGender'], row['ThesisTitle']
#                 ))
#
#         elif list(terms.keys())[0].startswith('X'):
#             topic = list(terms.keys())[0]
#             self.df['article_weights'] = 0
#             for _, row in self.df.sort_values(by=topic, ascending=False)[:no_dissertations].iterrows():
#                 out += ('\n{} {:7s}. Title: {}. Authors: {}.'.format(row.year, row.author_genders,
#                                                                  row.title, row.authors))
#
#
#
#         else:
#             dtm = self.get_document_term_matrix(vocabulary=terms.keys())
#             weights = np.array(list(terms.values()))
#             scores = dtm * weights
#
#             for i in scores.argsort()[::-1][:no_dissertations]:
#
#
#                 out +=('{} Author: {:7s} Advisor: {:7s} {}'.format(
#                     self.df['Year'][i], self.df['author_genders'][i],
#                     self.df['AdvisorGender'][i], self.df['ThesisTitle'][i]
#                 ))
#         return out
#
#
#     def print_topics_and_text_samples(self, file_path):
#         """
#         prints ids, names, terms, sample docs for each topic
#
#         :param file_path:
#         :return:
#         """
#         print("here")
#         out = ''
#         for i in range(1, 101):
#             topic = self.topics[i]
#             out += '\n\nTopic ID: {:3s}. Topic Name: {}'.format(str(i), topic['name'])
#             out += f'\nterms, prob: {topic["terms_prob"][:10]}'
#             out += f'\nterms, frex: {topic["terms_frex"][:10]}\nExamples:'
#
#             out += self.print_dissertations_mentioning_terms_or_topics([f'X{i}'], no_dissertations=10)
#
#         print(out)
#         with open('tset.txt', 'w') as outf: outf.write(out)
#
#
#
#
#     def print_examples_of_term_in_context(self, term, no_examples=10):
#         """
#         Finds examples of a term in the abstracts and prints them
#
#         >>> d = Dataset()
#         >>> d.print_examples_of_term_in_context('hegel', 2)
#         1987 PHILIP SCHAFF (1819-1893): PORTRAIT OF AN IMMIGRANT THEOLOGIAN         heology that accommodated such figure a hegel and schleiermacher tolerated liberal position yet rema
#         1995 State, society, and the market: Karl Sigmund Altenstein and the langua n idealism first by fichte and later by hegel thus the importance of his relationship to fichte and
#
#         :param term:
#         :return:
#         """
#
#         df = self.df[self.df['text'].str.contains(pat=r'\b{}\b'.format(term))]
#         if len(df) == 0:
#             print(f'No texts mention {term}.')
#             return
#
#         while True:
#             try:
#                 samples = df.sample(no_examples)
#                 break
#             except ValueError:
#                 no_examples -= 1
#
#         print(f'\n Found {len(df)} examples of {term}.')
#         for  _, row in samples.iterrows():
#
#             pos = row['text'].find(term)
#             if pos > -1:
#                 text = row['text'][max(0, pos-40):pos+60]
#                 print('{} {:7s} {:70s} {}'.format(
#                     row['year'], row['author_genders'], row['title'][:70], text,
#                 ))
#
#
#
#
#     def grid_plot_topics(self, sorted_topic_ids, hue,
#                          y_max=None, df=None, show_plot=True, store_as_filename=None):
#         """
#         Can be used to plot a 10x7 grid of all topics
#
#         :param df:
#         :param sorted_topic_ids: list(int)
#         :param hue:
#         :param y_max:
#         :return:
#
#         # show topic distribution from most female to most male
#         >>> d = Dataset()
#         >>> male = d.copy().filter(author_gender='male')
#         >>> female = d.copy().filter(author_gender='female')
#         >>> difs = {}
#         >>> for topic_id in range(1, 71):
#         ...     dif = np.mean(female.df[f'topic.{topic_id}']) - np.mean(male.df[f'topic.{topic_id}'])
#         ...     difs[topic_id] = dif
#         >>> sorted_topic_ids =  [t[0] for t in sorted(difs.items(), key = lambda x: x[1], reverse=True)]
#         >>> d.grid_plot_topics(sorted_topic_ids, hue='author_genders')
#
#         """
#
#         print("Creating topic gridplot")
#
#         if not df:
#             df = self.df
#
#         fig = plt.figure(figsize=(50, 50))
#         gs = gridspec.GridSpec(nrows=10, ncols=10, figure=fig)
#
#         for ax_id, topic_id in enumerate(sorted_topic_ids):
#
#             print(ax_id, topic_id)
#             row = ax_id // 10
#             col = ax_id % 10
#             ax = fig.add_subplot(gs[row, col])
#             ax = sns.lineplot(x='year', y=f'X{topic_id}', hue=hue,
#                               data=df, ax=ax)
#             ax.set_title(f'{topic_id}: {self.topics[topic_id]["name"]}')
#             ax.set_xlim(self.start_year, self.end_year)
#             if y_max:
#                 ax.set_ylim(0, y_max)
#
#         if show_plot:
#             plt.show()
#         if store_as_filename:
#             fig.savefig(Path('data', 'plots', store_as_filename))
#
#     def get_data(self, data_type, token_list, smoothing):
#
#         # load text info and turn it into term frequencies
#         if data_type == 'terms':
#             self.get_document_term_matrix(vocabulary=token_list, store_in_df=True)
#             for idx, row in self.df.iterrows():
#                 text_len = len(row.text.split())
#                 self.df.at[idx, 'text_len'] = text_len
#             for t in token_list:
#                 self.df[t] = self.df[t] / self.df['text_len']
#
#         data = {}
#         for t in token_list:
#             data[t] = defaultdict(list)
#
#         df = self.df
#
#         for idx, year in enumerate(range(self.start_year, self.end_year)):
#             time_slice = df[(df.year >= year - smoothing) & (df.year <= year + smoothing)]
#             time_slice_female = time_slice[time_slice.author_genders == 'female']
#             time_slice_male = time_slice[time_slice.author_genders == 'male']
#
#             for t in token_list:
#                 freq_both = time_slice[t].mean()
#                 freq_female = time_slice_female[t].mean()
#                 freq_male = time_slice_male[t].mean()
#
#
#                 # if a term doesn't appear, it is neutral
#                 if (freq_male + freq_female) == 0:
#                     freq_score = 0.5
#                 else:
#                     freq_score = freq_female / (freq_female + freq_male)
#
#                 data[t]['year'].append(year)
#                 data[t]['freq_score'].append(freq_score)
#                 data[t]['freq_both'].append(freq_both)
#
#         for t in token_list:
#             data[t]['mean_freq_score'] = np.mean(data[t]['freq_score'])
#             data[t]['mean_freq_both'] = np.mean(data[t]['freq_both'])
#             data[t]['freq_score_range'] = max(data[t]['freq_score']) - min(data[t]['freq_score'])
#
#         return data
#
#
#
#
#
#
#     def plot_topic_grid(self, smoothing=5):
#
#         from divergence_analysis import divergence_analysis
#         c1 = self.copy().filter(author_gender='male')
#         c2 = self.copy().filter(author_gender='female')
#         topic_df = divergence_analysis(self, c1, c2, topics_or_terms='topics',
#                             c1_name='male', c2_name='female', sort_by='dunning',
#                             number_of_terms_to_print=50)
#
#         topic_ids_sorted = [r['index'] + 1 for _, r in topic_df.iterrows()]
#         topic_names_sorted = [f'X{tid}' for tid in topic_ids_sorted]
#
#         data = self.get_data(data_type='topics', token_list=topic_names_sorted,
#                              smoothing=smoothing)
#
#         fig = plt.figure(figsize=(50, 50))
#         gs = gridspec.GridSpec(nrows=10, ncols=10, figure=fig)
#
#         for ax_id, topic_id in enumerate(topic_ids_sorted):
#             print(ax_id, topic_id)
#             row = ax_id // 10
#             col = ax_id % 10
#             ax = fig.add_subplot(gs[row, col])
#
#             t = f'X{topic_id}'
#             x = data[t]['year']
#             y = data[t]['freq_both']
#             freq_scores = data[t]['freq_score']
#             x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
#             spl = make_interp_spline(x, y, k=2)
#             y_lin = spl(x_lin)
#             spl_freq_score = make_interp_spline(x, freq_scores, k=1)
#             freq_score_lin = spl_freq_score(x_lin)
#
#             points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
#             segments = np.concatenate([points[:-1], points[1:]], axis=1)
#             norm = plt.Normalize(0.2, 0.8)
#
#             lc = LineCollection(segments, cmap='coolwarm', norm=norm)
#             # Set the values used for colormapping
#             lc.set_array(freq_score_lin)
#             lc.set_linewidth(2)
#             line = ax.add_collection(lc)
#             ax.set_xlim(x_lin.min(), x_lin.max())
#             ax.set_ylim(0, y_lin.max() * 1.1)
#
#             ax.set_title(topic_df.at[ax_id, 'term'])
#
#         plt.savefig(Path('data', 'plots', f'topic_plots.png'))
#         plt.show()
# #
#
#
#     def plot_topic_grid_of_selected_topics(self, smoothing=5):
#
#         # topic_ids_sorted = [r['index'] + 1 for _, r in topic_df.iterrows()]
#
#         topic_names_sorted = [x for x in self.df.columns if x.startswith('gen_approach_')]
#
#         data = self.get_data(data_type='topics', token_list=topic_names_sorted,
#                              smoothing=smoothing)
#
#         n_rows = 6
#         n_cols = 6
#         fig = plt.figure(figsize=(50, 50))
#         gs = gridspec.GridSpec(nrows=n_rows, ncols=n_cols, figure=fig)
#
#         for ax_id, name in enumerate(topic_names_sorted):
#             print(ax_id, name)
#             row = ax_id // n_cols
#             col = ax_id % n_cols
#             ax = fig.add_subplot(gs[row, col])
#
#             t = name
#             x = data[t]['year']
#             y = data[t]['freq_both']
#             freq_scores = data[t]['freq_score']
#             x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
#             spl = make_interp_spline(x, y, k=2)
#             y_lin = spl(x_lin)
#             spl_freq_score = make_interp_spline(x, freq_scores, k=1)
#             freq_score_lin = spl_freq_score(x_lin)
#
#             points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
#             segments = np.concatenate([points[:-1], points[1:]], axis=1)
#             norm = plt.Normalize(0.2, 0.8)
#
#             lc = LineCollection(segments, cmap='coolwarm', norm=norm)
#             # Set the values used for colormapping
#             lc.set_array(freq_score_lin)
#             lc.set_linewidth(2)
#             line = ax.add_collection(lc)
#             ax.set_xlim(x_lin.min(), x_lin.max())
#             ax.set_ylim(0, max(y_lin) * 1.1)
#
#             ax.set_title(name)
#             # ax.set_title(f'({topic_id}) {self.topics[topic_id]["name"]}')
#
#         plt.savefig(Path('data', 'plots', f'general_approaches.png'))
#         plt.show()
#
#
#
#
    def plot_londa(self,
                   data_type,
                   term_or_topic_list,
                   smoothing=3):

        data = self.get_data(data_type=data_type, token_list=term_or_topic_list,
                             smoothing=smoothing)

        # 2: Set up plot
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(nrows=1,
                               ncols=1,
                               figure=fig,
                               width_ratios=[1],
                               height_ratios=[1],
                               wspace=0.2, hspace=0.05
                               )

        ax = fig.add_subplot(gs[0, 0])
        ax.set_ylim(0, 1)
        ax.set_xlim(self.start_year + 2, self.end_year - 2)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both')

        for t in term_or_topic_list:
            x = data[t]['year']
            y = data[t]['freq_both']
            freq_scores = data[t]['freq_score']
            x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
            spl = make_interp_spline(x, y, k=2)
            y_lin = spl(x_lin)
            spl_freq_score = make_interp_spline(x, freq_scores, k=1)
            freq_score_lin = spl_freq_score(x_lin)

            # points[0] = (x[0], y[0]
            points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
            # segments[0] = 2x2 matrix. segments[0][0] = points[0]; segments[0][1] = points[1]
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0.2, 0.8)

            lc = LineCollection(segments, cmap='coolwarm', norm=norm)
            # Set the values used for colormapping
            lc.set_array(freq_score_lin)
            lc.set_linewidth(4)
            line = ax.add_collection(lc)


        fig.colorbar(line, ax=ax)

        ax.set_xlim(x_lin.min(), x_lin.max())
        # ax.set_ylim(0, y_lin.max() * 1.1)
        ax.set_ylim(0, 0.4)

        plt.savefig(Path('data', 'plots', f'plot_londa.png'))
        plt.show()
#
#
#
#     def plot_gender_development_over_time(self,
#             no_terms_or_topics_to_show=8,
#             data='topics',
#             display_selector='most_frequent',
#             selected_terms_or_topics=None,
#             show_plot=True,
#             store_to_filename=None,
#             title=None,
#             smoothing=3
#
#           ):
#
#         """
#
#         :param no_terms_or_topics_to_show: int
#         :param data: 'topics', 'terms', 'terms_of_topics'
#         :param display_selector: 'most_frequent', 'most_divergent', 'most_variable'
#         :param selected_terms_or_topics: topic_id or list of terms
#         :param show_plot: bool
#         :param store_to_filename: bool or str
#         :return:
#         """
#
#         if data == 'terms_of_topic':
#             if not isinstance(selected_terms_or_topics, int):
#                 raise ValueError("When displaying 'terms_of_topic', please pass a topic_id for param"
#                                  "selected_terms_or_topics")
#
#         # 0: find terms or topics to display
#         if data == 'topics':
#             selected_terms_or_topics = [f'topic.{id}' for id in range(1, 71)]
#             title_name = 'topics'
#         elif data == 'terms':
#             vocab = []
#             for t in selected_terms_or_topics:
#                 vocab.append(t)
#             self.get_document_term_matrix(vocabulary=vocab, store_in_df=True)
#             title_name = 'terms'
#         elif data == 'terms_of_topic':
#             vocab = []
#             topic_id = selected_terms_or_topics
#             for term in self.topics[topic_id]['terms_prob']:
#                 if term in self.vocabulary:
#                     vocab.append(term)
#             selected_terms_or_topics = vocab
#             self.get_document_term_matrix(vocabulary=vocab, store_in_df=True)
#             title_name = f'terms of topic {topic_id}'
#         else:
#             raise ValueError('"data" has to be "terms" "topics" or "terms_of_topic"')
#
#         if not title:
#             if display_selector == 'most_frequent':
#                 title = f'Most frequent {title_name} for female (top) and male authors (bottom)'
#             elif display_selector == 'most_divergent':
#                 title = f'Most divergent {title_name} for female (top) and male authors (bottom)'
#             else:
#                 title = f'Most variable {title_name} for female (top) and male authors (bottom)'
#
#         df = self.df
#
#         # 1: Load data
#         data = {}
#         for t in selected_terms_or_topics:
#             data[t] = defaultdict(list)
#         min_freq_total = 1
#         max_freq_total = 0
#
#         for idx, year in enumerate(range(self.start_year, self.end_year)):
#
#             time_slice = df[(df.year >= year - smoothing) & (df.year <= year + smoothing)]
#             time_slice_female = time_slice[time_slice.author_genders == 'female']
#             time_slice_male = time_slice[time_slice.author_genders == 'male']
#
#             for t in selected_terms_or_topics:
#                 freq_total = time_slice[t].mean()
#                 freq_female = time_slice_female[t].mean()
#                 freq_male = time_slice_male[t].mean()
#
#                 # if a term doesn't appear, it is neutral
#                 if (freq_male + freq_female) == 0:
#                     freq_score = 0.5
#                 else:
#                     freq_score = freq_female / (freq_female + freq_male)
#
#                 data[t]['year'].append(year)
#                 data[t]['freq_score'].append(freq_score)
#                 data[t]['freq_total'].append(freq_total)
#
#                 if freq_total < min_freq_total:
#                     min_freq_total = freq_total
#                 if freq_total > max_freq_total:
#                     max_freq_total = freq_total
#
#         for t in terms:
#             data[t]['mean_freq_score'] = np.mean(data[t]['freq_score'])
#             data[t]['mean_freq_total'] = np.mean(data[t]['freq_total'])
#             data[t]['freq_score_range'] = max(data[t]['freq_score']) - min(data[t]['freq_score'])
#
#         # 2: Set up plot
#         fig = plt.figure(figsize=(12, 12))
#         gs = gridspec.GridSpec(nrows=1,
#                                ncols=1,
#                                figure=fig,
#                                width_ratios=[1],
#                                height_ratios=[1],
#                                wspace=0.2, hspace=0.05
#                                )
#
#         ax = fig.add_subplot(gs[0, 0])
#         ax.set_ylim(0, 1)
#         ax.set_xlim(self.start_year + 2, self.end_year -2)
#         ax.set_axisbelow(True)
#         ax.grid(which='major', axis='both')
#
#         dot_scaler = MinMaxScaler((0.0, 50.0))
#         dot_scaler.fit(np.array([min_freq_total, max_freq_total]).reshape(-1, 1))
#         legends = []
#
#         def draw_line(t, t_data, df):
#             """
#             Draws one line depending on t (term or topic string) and t_data (dict of data belonging
#             to t)
#
#             :param t: str
#             :param t_data: dict
#             :return:
#             """
#             y = t_data['freq_score']
#             x = t_data['year']
#             frequencies = t_data['freq_total']
#             if t.startswith('topic.'):
#                 legend = self.topics[int(t[6:])]['name']
#             else:
#                 legend = '{:10s} ({})'.format(t, df[t].sum())
#
#             x_spline = np.linspace(min(x), max(x), ((self.end_year - 2) - (self.start_year + 2)) * 1000)
#             spl = make_interp_spline(x, y, k=1)  # BSpline object
#             y_spline = spl(x_spline)
#
#             line_interpolater = interp1d(x, frequencies)
#             line_widths = line_interpolater(x_spline)
#             line_widths = dot_scaler.transform(line_widths.reshape(-1, 1)).flatten()
#
#             try:
#                 color = sns.color_palette()[len(legends)]
#             except IndexError:
#                 color = sns.cubehelix_palette(100, start=2, rot=0, dark=0, light=.95)[len(legends)]
#
#             ax.scatter(x_spline, y_spline, s=line_widths, antialiased=True,
#                        color=color)
#             legends.append(mpatches.Patch(color=color, label=legend))
#
#         # 3: Plot
#         if display_selector == 'most_frequent':
#             ax.set_title(title, weight='bold', fontsize=18)
#             sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['mean_freq_total'], reverse=True)
#             for t, t_data in sorted_items[:no_terms_or_topics_to_show]:
#                 draw_line(t, t_data, df)
#         elif display_selector == 'most_divergent':
#             ax.set_title(title, weight='bold', fontsize=18)
#             sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['mean_freq_score'], reverse=True)
#             no_disp = no_terms_or_topics_to_show // 2
#             for t, t_data in sorted_items[:no_disp] + sorted_items[::-1][:no_disp]:
#                 draw_line(t, t_data, df)
#         elif display_selector == 'most_variable':
#             ax.set_title(title, weight='bold', fontsize=18)
#             # sort by mean_freq_range second to preserve colors between plots
#             sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['freq_score_range'], reverse=True)
#             sorted_items = sorted_items[:no_terms_or_topics_to_show]
#             sorted_items = sorted(sorted_items, key=lambda k_v: k_v[1]['mean_freq_score'], reverse=True)
#             for t, t_data in sorted_items:
#                 draw_line(t, t_data, df)
#
#         else:
#             raise ValueError('display_selector has to be most_frequent, most_variable, or most_divergent')
#
#         ax.legend(handles=legends, loc=4)
#
#         if show_plot:
#             plt.show()
#         if store_to_filename:
#             fig.savefig(Path('data', store_to_filename))
#
#
# def plot_adviser_gender():
#
#     d = Dataset()
#     d.filter(start_year=1980, end_year=2015)
#
#     fig = plt.figure(figsize=(50, 50))
#     gs = gridspec.GridSpec(nrows=10, ncols=7, figure=fig)
#
#     for ax_id, topic_id in enumerate(range(1, 71)):
#         print(ax_id)
#
#         row = ax_id // 7
#         col = ax_id % 7
#         ax = fig.add_subplot(gs[row, col])
#
#         male = np.zeros(2015-1980+1)
#         female = np.zeros(2015 - 1980 + 1)
#         unknown = np.zeros(2015 - 1980 + 1)
#
#         male_a = np.zeros(2015 - 1980 + 1)
#         female_a = np.zeros(2015 - 1980 + 1)
#
#
#         top = d.copy()
#         top.topic_percentile_score_filter(topic_id, min_percentile_score=80)
#         for _, row in top.df.iterrows():
#             year = row['Year']
#             advisor_gender = row['AdvisorGender']
#             advisee_gender = row['author_genders']
#
#             if row['AdvisorGender'] == 'female':
#                 female[row['Year'] - 1980] += 1
#             elif row['AdvisorGender'] == 'male':
#                 male[row['Year'] - 1980] += 1
#             else:
#                 unknown[row['Year'] - 1980] += 1
#
#             if advisee_gender == 'female':
#                 female_a[year - 1980] += 1
#             elif advisee_gender == 'male':
#                 male_a[year - 1980] += 1
#
#         # sns.lineplot(x=range(1980, 2016), y=unknown, ax=ax, label='unknown')
#         # sns.lineplot(x=range(1980, 2016), y=male, ax=ax, label='male')
#         # sns.lineplot(x=range(1980, 2016), y=female, ax=ax, label='female')
#         # sns.lineplot(x=range(1980, 2016), y=female_a, ax=ax, label='female_a')
#         # sns.lineplot(x=range(1980, 2016), y=male_a, ax=ax, label='male_a')
#         sns.lineplot(x=range(1980, 2016), y=female_a / (female_a+male_a), label='Advisee')
#         sns.lineplot(x=range(1980, 2016), y=female / (female+male), label='Advisor')
#
#
#         ax.legend()
#         ax.set_title(f'{topic_id}: {d.topics[topic_id]["name"]}')
#
#     plt.show()
#
#
#
#
#

#     def topic_percentile_score_filter(self, topic, min_percentile_score=0, max_percentile_score=100):
#         """
#         Filter the dataset such that it only contains documents that score between
#         min_percentile_score and max_percentile_score for a topic
#
#         :param topic: topic id (int) or str ('topic.XX')
#         :param min_percentile_score: 0 to 100
#         :param max_percentile_score: 0 to 100
#         :return:
#
#         >>> d = Dataset()
#         >>> len(d)
#         21634
#
#         # select dissertations that score in the top decile (strongest) for the gender topic (28)
#         >>> d.topic_percentile_score_filter(topic=28, min_percentile_score=90)
#         >>> print(len(d), min(d.df['percentile_score_topic.28']))
#         2164 90.0
#
#         # select 50th-70th decile
#         >>> d2 = Dataset()
#         >>> d2.topic_percentile_score_filter('topic.28', min_percentile_score=50, max_percentile_score=70)
#         >>> len(d2)
#         6491
#
#         # filters can be combined
#         >>> d3 = Dataset()
#         >>> d3.topic_percentile_score_filter(14, min_percentile_score=80)
#         >>> d3.topic_percentile_score_filter(28, min_percentile_score=80)
#         >>> len(d3)
#         866
#
#
#         """
#         if not isinstance(topic, str):
#             topic = f'topic.{topic}'
#
#         if not f'percentile_score_{topic}' in self.df.columns:
#             # add all of the topics at once because if we filter topics twice, the ranks would be
#             # influenced by the first selection
#             for i in range(1, 71):
#                 t = f'topic.{i}'
#                 self.df[f'percentile_score_{t}'] = self.df[t].rank(pct=True) * 100 // 10 * 10
#         self.df = self.df[self.df[f'percentile_score_{topic}'] >= min_percentile_score]
#         self.df = self.df[self.df[f'percentile_score_{topic}'] <= max_percentile_score]
#         self.df = self.df.reset_index()
#         self.topic_percentile_score_filters.append({
#             'topic': topic,
#             'min_percentile_score': min_percentile_score,
#             'max_percentile_score': max_percentile_score
#         })
#
#
#     def filter(self, start_year=None, end_year=None, author_gender=None,
#                term_filter=None, institution_filter=None, advisor_gender=None,
#                has_descendants=None):
#         """
#
#         :param start_year:    (int between 1976 and 2015)  earliest year to include
#         :param end_year:      (int between 1976 and 2015)  last year to include
#         :param author_gender: (male or female)             only include male or female authors
#         :param term_filter:   string or raw string         only include documents mentioning term
#         :return:
#
#         >>> d = Dataset()
#         >>> len(d)
#         21634
#
#         # filter for years (inclusive) between 1985 and 1995
#         >>> d.filter(start_year=1985, end_year=1995)
#         >>> len(d)
#         6532
#
#         # filter by author gender
#         >>> d.filter(author_gender='male')
#         >>> len(d)
#         4055
#
#         # filter by advisor gender
#         >>> d.filter(advisor_gender='female')
#         >>> len(d)
#         220
#
#
#         # filter by term or regex
#         # regex example: r'\bgender\b'
#         >>> d.filter(term_filter='gender')
#         >>> len(d)
#         13
#
#         # filter by institution
#         >>> d = Dataset()
#         >>> d.filter(institution_filter='harvard')
#         >>> len(d)
#         778
#
#         >>> d = Dataset()
#         >>> d.filter(institution_filter='not_harvard')
#         >>> len(d)
#         20856
#
#         >>> d = Dataset()
#         >>> d.filter(has_descendants=True)
#         >>> len(d)
#         1583
#
#         """
#
#         df = self.df
#
#         if start_year:
#             df = df[df.year >= start_year]
#             self.start_year = start_year
#         if end_year:
#             df = df[df.year <= end_year]
#             self.end_year = end_year
#         if author_gender:
#             if not author_gender in ['male', 'female']:
#                 raise ValueError(f'Author gender needs to be male or female but not {author_gender}')
#             df = df[df.author_genders == author_gender]
#             self.author_gender = author_gender
#
#         if advisor_gender:
#             if not advisor_gender in ['male', 'female', 'unknown']:
#                 raise ValueError(f'Author gender needs to be male or female but not {advisor_gender}')
#             df = df[df.AdvisorGender == advisor_gender]
#             self.advisor_gender = advisor_gender
#
#
#         if term_filter:
#             if term_filter.startswith('not_'):
#                 term_filter = term_filter[4:]
#                 df = df[df['text'].str.contains(pat=term_filter, regex=True) == False]
#             else:
#                 df = df[df['text'].str.contains(pat=term_filter, regex=True) == True]
#             self.term_filter = term_filter
#
#         if institution_filter:
#             if institution_filter.startswith('not_'):
#                 institution_filter = institution_filter[4:]
#                 df = df[df['ThesisInstitution'].str.contains(institution_filter, case=False) == False]
#             else:
#                 df = df[df['ThesisInstitution'].str.contains(institution_filter, case=False) == True]
#             self.institution_filter = institution_filter
#
#         if has_descendants == True:
#             df = df[df.AnyDescendants == 1]
#             self.descendants_filter = True
#
#         if has_descendants == False:
#             df = df[df.AnyDescendants == 0]
#             self.descendants_filter = False
#
#         self.df = df.reset_index()
#         return self
#
#
#     def get_vocabulary(self, exclude_stopwords=True, max_terms=None, min_appearances=None,
#                        include_2grams=False):
#         """
#         Returns a list of all terms that appear in the text column
#
#         :return: list
#
#         # stop words are excluded by default
#         >>> d = Dataset()
#         >>> len(d.get_vocabulary(exclude_stopwords=True))
#         176458
#         >>> d.get_vocabulary().count('are')
#         0
#
#         # you can also limit the number of terms and require a minimum number of appearances
#         >>> voc = d.get_vocabulary(max_terms=1000, min_appearances=2)
#         >>> len(voc)
#         1000
#
#         """
#
#         if not max_terms:
#             max_terms = 1000000
#
#         vocabulary = Counter()
#         for abstract in self.df['text']:
#             a = abstract.split()
#             for idx, word in enumerate(a):
#                 vocabulary[word] += 1
#                 if include_2grams:
#                     try:
#                         gram = '{} {}'.format(word, a[idx+1])
#                         vocabulary[gram] += 1
#                     except IndexError:
#                         pass
#
#
#
#         if exclude_stopwords:
#             clean_vocabulary = Counter()
#             stopwords = stop_words.ENGLISH_STOP_WORDS.union({'wa', 'ha',
#                                                      'óé', 'dotbelow', 'cyrillic'})
#
#             for ngram in vocabulary:
#                 valid = True
#                 for term in ngram.split():
#                     if term in stopwords:
#                         valid = False
#                 if valid:
#                     clean_vocabulary[ngram] = vocabulary[ngram]
#             vocabulary = clean_vocabulary
#
#
#         vocab_list = []
#         for word, count in vocabulary.most_common(max_terms):
#             if min_appearances and count < min_appearances:
#                 continue
#             else:
#                 vocab_list.append(word)
#
#         return sorted(vocab_list)
#
#
#     def get_document_topic_matrix(self, vocabulary):
#         """
#         Returns a document-topic sparse matrix. each row represents a document and each column a
#         topic
#         Note: all topics are one-off because there is no topic.0, i.e. topic.1 is stored in the
#         0th column.
#
#         Vocabulary are the columns to select
#
#         >>> d = Dataset()
#         >>> vocabulary = [f'X{i}' for i in range(1, 101)]
#         >>> dtm = d.get_document_topic_matrix()
#         >>> dtm.shape
#         (23246, 70)
#
#         :return: csr_matrix
#         """
#
#         dtm = csr_matrix(self.df[vocabulary].to_numpy())
#         return dtm
#
#
#     def get_document_term_matrix(self, vocabulary, store_in_df=False):
#         """
#
#         Returns a document-term sparse matrix. each row represents a document and each column a
#         topic
#
#         If store_in_df is selected, the dtm will be stored in the dataframe, i.e. every term
#         is stored as a row in the dataframe (useful for quick term selection)
#
#         :param vocabulary: list
#         :param store_in_df: bool
#         :return:
#
#         >>> d = Dataset()
#         >>> vocabulary = d.get_vocabulary(max_terms=10000)
#         >>> dtm = d.get_document_term_matrix(vocabulary)
#         >>> dtm.shape
#         (23246, 10000)
#
#         """
#
#         vectorizer = CountVectorizer(vocabulary=vocabulary)
#         dtm = vectorizer.fit_transform(self.df['text'].to_list())
#
#         if store_in_df:
#             dtm_df = pd.DataFrame(dtm.toarray(), columns=vocabulary)
#             self.df = pd.concat([self.df, dtm_df], axis=1)
#         else:
#             return dtm
#
#     def print_dissertations_mentioning_terms_or_topics(self, terms, no_dissertations=5):
#         """
#         print dissertations that mention specific terms or topics,
#         can be weighted or unweighted
#
#         :param terms: list or dict
#         :param no_dissertations: number of dissertations to print, default: 5
#
#         >>> d = Dataset()
#         >>> terms = ['woman', 'female', 'women', 'feminist', 'gender']
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
#         2014 Author: female  Advisor: female  Authors, Activists, Apostles: Women's Leadership in the New Christian Right
#         2006 Author: female  Advisor: male    Women on the march: Gender and anti-fascism in American communism, 1935--1939
#
#         # by default, each term has weight=1. However, terms can also be a dict of term weights
#         >>> terms = {'nurse': 1, 'drugs': 10}
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
#         1997 Author: female  Advisor: male    Regulating beauty: Cosmetics in American culture from the 1906 Pure Food and Drugs Act to the 1938 Food, Drug and Cosmetic Act
#         1994 Author: female  Advisor: female  G.I. nurses at war: Gender and professionalization in the Army Nurse Corps during World War II
#
#         # topics also work:
#         >>> terms = ['topic.28']
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
#         1989 Author: female  Advisor: unknown Day nurseries and wage-earning mothers in the United States, 1890-1930
#         1988 Author: female  Advisor: female  "Women adrift" and "urban pioneers": Self-supporting working women in America, 1880-1930
#
#         >>> terms = ['gay', 'homosexual', 'homosexuality', 'masculinity']
#         >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=20)
#
#         """
#
#         # if input is a list, give each term weight 1
#         if isinstance(terms, list):
#             terms = {t: 1 for t in terms}
#
#         out = ''
#
#         if list(terms.keys())[0].startswith('topic.'):
#             self.df['dissertation_weights'] = 0
#             for topic, weight in terms.items():
#                 self.df['dissertation_weights'] += weight * self.df[topic]
#             for _, row in self.df.sort_values(by='dissertation_weights', ascending=False)[:no_dissertations].iterrows():
#                 out += ('\n{} Author: {:7s} Advisor: {:7s} {}'.format(
#                     row['Year'], row['author_genders'],
#                     row['AdvisorGender'], row['ThesisTitle']
#                 ))
#
#         elif list(terms.keys())[0].startswith('X'):
#             topic = list(terms.keys())[0]
#             self.df['article_weights'] = 0
#             for _, row in self.df.sort_values(by=topic, ascending=False)[:no_dissertations].iterrows():
#                 out += ('\n{} {:7s}. Title: {}. Authors: {}.'.format(row.year, row.author_genders,
#                                                                  row.title, row.authors))
#
#
#
#         else:
#             dtm = self.get_document_term_matrix(vocabulary=terms.keys())
#             weights = np.array(list(terms.values()))
#             scores = dtm * weights
#
#             for i in scores.argsort()[::-1][:no_dissertations]:
#
#
#                 out +=('{} Author: {:7s} Advisor: {:7s} {}'.format(
#                     self.df['Year'][i], self.df['author_genders'][i],
#                     self.df['AdvisorGender'][i], self.df['ThesisTitle'][i]
#                 ))
#         return out
#
#
#     def print_topics_and_text_samples(self, file_path):
#         """
#         prints ids, names, terms, sample docs for each topic
#
#         :param file_path:
#         :return:
#         """
#         print("here")
#         out = ''
#         for i in range(1, 101):
#             topic = self.topics[i]
#             out += '\n\nTopic ID: {:3s}. Topic Name: {}'.format(str(i), topic['name'])
#             out += f'\nterms, prob: {topic["terms_prob"][:10]}'
#             out += f'\nterms, frex: {topic["terms_frex"][:10]}\nExamples:'
#
#             out += self.print_dissertations_mentioning_terms_or_topics([f'X{i}'], no_dissertations=10)
#
#         print(out)
#         with open('tset.txt', 'w') as outf: outf.write(out)
#
#
#
#
#     def print_examples_of_term_in_context(self, term, no_examples=10):
#         """
#         Finds examples of a term in the abstracts and prints them
#
#         >>> d = Dataset()
#         >>> d.print_examples_of_term_in_context('hegel', 2)
#         1987 PHILIP SCHAFF (1819-1893): PORTRAIT OF AN IMMIGRANT THEOLOGIAN         heology that accommodated such figure a hegel and schleiermacher tolerated liberal position yet rema
#         1995 State, society, and the market: Karl Sigmund Altenstein and the langua n idealism first by fichte and later by hegel thus the importance of his relationship to fichte and
#
#         :param term:
#         :return:
#         """
#
#         df = self.df[self.df['text'].str.contains(pat=r'\b{}\b'.format(term))]
#         if len(df) == 0:
#             print(f'No texts mention {term}.')
#             return
#
#         while True:
#             try:
#                 samples = df.sample(no_examples)
#                 break
#             except ValueError:
#                 no_examples -= 1
#
#         print(f'\n Found {len(df)} examples of {term}.')
#         for  _, row in samples.iterrows():
#
#             pos = row['text'].find(term)
#             if pos > -1:
#                 text = row['text'][max(0, pos-40):pos+60]
#                 print('{} {:7s} {:70s} {}'.format(
#                     row['year'], row['author_genders'], row['title'][:70], text,
#                 ))
#
#
#     def normalize_dataset_by_5year_interval(self, no_docs_per_5year_interval=5000):
#
# #        dfs = {}
#
#         docs = []
#
#         for years_range in [(1976, 1984), (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2004),
#                             (2005, 2009), (2010, 2015)]:
#             y1, y2 = years_range
#             if self.start_year >= y2 or self.end_year < y2:
#                 continue
# #            dfs[years_range] = self.df[(self.df['Year'] >= y1) & (self.df['Year'] <= y2)]
#             df = self.df[(self.df['year'] >= y1) & (self.df['year'] <= y2)]
#             df = df.to_dict('records')
#
#             if len(df) == 0:
#                 raise IndexError(f'Cannot generate dataset of {no_docs_per_5year_interval} docs for {y1}-{y2} for'
#                       f' {self.name_full} with 0 docs.')
#             if len(df) < 50:
#                 print(f'WARNING. Generating dataset of {no_docs_per_5year_interval} docs for {y1}-{y2} for'
#                       f' {self.name} with only {len(df)} docs.')
#
#             for i in range(no_docs_per_5year_interval):
#                 docs.append(random.sample(df, 1)[0])
#         self.df = pd.DataFrame(docs)
#
#
#     def grid_plot_topics(self, sorted_topic_ids, hue,
#                          y_max=None, df=None, show_plot=True, store_as_filename=None):
#         """
#         Can be used to plot a 10x7 grid of all topics
#
#         :param df:
#         :param sorted_topic_ids: list(int)
#         :param hue:
#         :param y_max:
#         :return:
#
#         # show topic distribution from most female to most male
#         >>> d = Dataset()
#         >>> male = d.copy().filter(author_gender='male')
#         >>> female = d.copy().filter(author_gender='female')
#         >>> difs = {}
#         >>> for topic_id in range(1, 71):
#         ...     dif = np.mean(female.df[f'topic.{topic_id}']) - np.mean(male.df[f'topic.{topic_id}'])
#         ...     difs[topic_id] = dif
#         >>> sorted_topic_ids =  [t[0] for t in sorted(difs.items(), key = lambda x: x[1], reverse=True)]
#         >>> d.grid_plot_topics(sorted_topic_ids, hue='author_genders')
#
#         """
#
#         print("Creating topic gridplot")
#
#         if not df:
#             df = self.df
#
#         fig = plt.figure(figsize=(50, 50))
#         gs = gridspec.GridSpec(nrows=10, ncols=10, figure=fig)
#
#         for ax_id, topic_id in enumerate(sorted_topic_ids):
#
#             print(ax_id, topic_id)
#             row = ax_id // 10
#             col = ax_id % 10
#             ax = fig.add_subplot(gs[row, col])
#             ax = sns.lineplot(x='year', y=f'X{topic_id}', hue=hue,
#                               data=df, ax=ax)
#             ax.set_title(f'{topic_id}: {self.topics[topic_id]["name"]}')
#             ax.set_xlim(self.start_year, self.end_year)
#             if y_max:
#                 ax.set_ylim(0, y_max)
#
#         if show_plot:
#             plt.show()
#         if store_as_filename:
#             fig.savefig(Path('data', 'plots', store_as_filename))
#

    def get_data(self, data_type, token_list, smoothing):

        # load text info and turn it into term frequencies
        if data_type == 'terms':
            self.get_document_term_matrix(vocabulary=token_list, store_in_df=True)
            for idx, row in self.df.iterrows():
                text_len = len(row.text.split())
                self.df.at[idx, 'text_len'] = text_len
            for t in token_list:
                self.df[t] = self.df[t] / self.df['text_len']

        data = {}
        for t in token_list:
            data[t] = defaultdict(list)

        df = self.df

        for idx, year in enumerate(range(self.start_year, self.end_year)):
            time_slice = df[(df.year >= year - smoothing) & (df.year <= year + smoothing)]
            time_slice_female = time_slice[time_slice.author_genders == 'female']
            time_slice_male = time_slice[time_slice.author_genders == 'male']

            for t in token_list:
                freq_both = time_slice[t].mean()
                freq_female = time_slice_female[t].mean()
                freq_male = time_slice_male[t].mean()


                # if a term doesn't appear, it is neutral
                if (freq_male + freq_female) == 0:
                    freq_score = 0.5
                else:
                    freq_score = freq_female / (freq_female + freq_male)

                data[t]['year'].append(year)
                data[t]['freq_score'].append(freq_score)
                data[t]['freq_both'].append(freq_both)

        for t in token_list:
            data[t]['mean_freq_score'] = np.mean(data[t]['freq_score'])
            data[t]['mean_freq_both'] = np.mean(data[t]['freq_both'])
            data[t]['freq_score_range'] = max(data[t]['freq_score']) - min(data[t]['freq_score'])

        return data
#
#
#
#
#
#     def plot_topic_grid(self, smoothing=5):
#
#         from divergence_analysis import divergence_analysis
#         c1 = self.copy().filter(author_gender='male')
#         c2 = self.copy().filter(author_gender='female')
#         topic_df = divergence_analysis(self, c1, c2, topics_or_terms='topics',
#                             c1_name='male', c2_name='female', sort_by='dunning',
#                             number_of_terms_to_print=50)
#
#         topic_ids_sorted = [r['index'] + 1 for _, r in topic_df.iterrows()]
#         topic_names_sorted = [f'X{tid}' for tid in topic_ids_sorted]
#
#         data = self.get_data(data_type='topics', token_list=topic_names_sorted,
#                              smoothing=smoothing)
#
#         fig = plt.figure(figsize=(50, 50))
#         gs = gridspec.GridSpec(nrows=10, ncols=10, figure=fig)
#
#         for ax_id, topic_id in enumerate(topic_ids_sorted):
#             print(ax_id, topic_id)
#             row = ax_id // 10
#             col = ax_id % 10
#             ax = fig.add_subplot(gs[row, col])
#
#             t = f'X{topic_id}'
#             x = data[t]['year']
#             y = data[t]['freq_both']
#             freq_scores = data[t]['freq_score']
#             x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
#             spl = make_interp_spline(x, y, k=2)
#             y_lin = spl(x_lin)
#             spl_freq_score = make_interp_spline(x, freq_scores, k=1)
#             freq_score_lin = spl_freq_score(x_lin)
#
#             points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
#             segments = np.concatenate([points[:-1], points[1:]], axis=1)
#             norm = plt.Normalize(0.2, 0.8)
#
#             lc = LineCollection(segments, cmap='coolwarm', norm=norm)
#             # Set the values used for colormapping
#             lc.set_array(freq_score_lin)
#             lc.set_linewidth(2)
#             line = ax.add_collection(lc)
#             ax.set_xlim(x_lin.min(), x_lin.max())
#             ax.set_ylim(0, y_lin.max() * 1.1)
#
#             ax.set_title(topic_df.at[ax_id, 'term'])
#
#         plt.savefig(Path('data', 'plots', f'topic_plots.png'))
#         plt.show()
#
#
#
#     def plot_topic_grid_of_selected_topics(self, smoothing=5):
#
#         # topic_ids_sorted = [r['index'] + 1 for _, r in topic_df.iterrows()]
#
#         topic_names_sorted = [x for x in self.df.columns if x.startswith('gen_approach_')]
#
#         data = self.get_data(data_type='topics', token_list=topic_names_sorted,
#                              smoothing=smoothing)
#
#         n_rows = 6
#         n_cols = 6
#         fig = plt.figure(figsize=(50, 50))
#         gs = gridspec.GridSpec(nrows=n_rows, ncols=n_cols, figure=fig)
#
#         for ax_id, name in enumerate(topic_names_sorted):
#             print(ax_id, name)
#             row = ax_id // n_cols
#             col = ax_id % n_cols
#             ax = fig.add_subplot(gs[row, col])
#
#             t = name
#             x = data[t]['year']
#             y = data[t]['freq_both']
#             freq_scores = data[t]['freq_score']
#             x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
#             spl = make_interp_spline(x, y, k=2)
#             y_lin = spl(x_lin)
#             spl_freq_score = make_interp_spline(x, freq_scores, k=1)
#             freq_score_lin = spl_freq_score(x_lin)
#
#             points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
#             segments = np.concatenate([points[:-1], points[1:]], axis=1)
#             norm = plt.Normalize(0.2, 0.8)
#
#             lc = LineCollection(segments, cmap='coolwarm', norm=norm)
#             # Set the values used for colormapping
#             lc.set_array(freq_score_lin)
#             lc.set_linewidth(2)
#             line = ax.add_collection(lc)
#             ax.set_xlim(x_lin.min(), x_lin.max())
#             ax.set_ylim(0, max(y_lin) * 1.1)
#
#             ax.set_title(name)
#             # ax.set_title(f'({topic_id}) {self.topics[topic_id]["name"]}')
#
#         plt.savefig(Path('data', 'plots', f'general_approaches.png'))
#         plt.show()
#
#
#
#
#     def plot_londa(self,
#                    data_type,
#                    term_or_topic_list,
#                    smoothing=3):
#
#         data = self.get_data(data_type=data_type, token_list=term_or_topic_list,
#                              smoothing=smoothing)
#
#         # 2: Set up plot
#         fig = plt.figure(figsize=(12, 12))
#         gs = gridspec.GridSpec(nrows=1,
#                                ncols=1,
#                                figure=fig,
#                                width_ratios=[1],
#                                height_ratios=[1],
#                                wspace=0.2, hspace=0.05
#                                )
#
#         ax = fig.add_subplot(gs[0, 0])
#         ax.set_ylim(0, 1)
#         ax.set_xlim(self.start_year + 2, self.end_year - 2)
#         ax.set_axisbelow(True)
#         ax.grid(which='major', axis='both')
#
#         for t in term_or_topic_list:
#             x = data[t]['year']
#             y = data[t]['freq_both']
#             freq_scores = data[t]['freq_score']
#             x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
#             spl = make_interp_spline(x, y, k=2)
#             y_lin = spl(x_lin)
#             spl_freq_score = make_interp_spline(x, freq_scores, k=1)
#             freq_score_lin = spl_freq_score(x_lin)
#
#             # points[0] = (x[0], y[0]
#             points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
#             # segments[0] = 2x2 matrix. segments[0][0] = points[0]; segments[0][1] = points[1]
#             segments = np.concatenate([points[:-1], points[1:]], axis=1)
#             norm = plt.Normalize(0.2, 0.8)
#
#             lc = LineCollection(segments, cmap='coolwarm', norm=norm)
#             # Set the values used for colormapping
#             lc.set_array(freq_score_lin)
#             lc.set_linewidth(4)
#             line = ax.add_collection(lc)
#
#
#         fig.colorbar(line, ax=ax)
#
#         ax.set_xlim(x_lin.min(), x_lin.max())
#         # ax.set_ylim(0, y_lin.max() * 1.1)
#         ax.set_ylim(0, 0.4)
#
#         plt.savefig(Path('data', 'plots', f'plot_londa.png'))
#         plt.show()
#
#
#
#     def plot_gender_development_over_time(self,
#             no_terms_or_topics_to_show=8,
#             data='topics',
#             display_selector='most_frequent',
#             selected_terms_or_topics=None,
#             show_plot=True,
#             store_to_filename=None,
#             title=None,
#             smoothing=3
#
#           ):
#
#         """
#
#         :param no_terms_or_topics_to_show: int
#         :param data: 'topics', 'terms', 'terms_of_topics'
#         :param display_selector: 'most_frequent', 'most_divergent', 'most_variable'
#         :param selected_terms_or_topics: topic_id or list of terms
#         :param show_plot: bool
#         :param store_to_filename: bool or str
#         :return:
#         """
#
#         if data == 'terms_of_topic':
#             if not isinstance(selected_terms_or_topics, int):
#                 raise ValueError("When displaying 'terms_of_topic', please pass a topic_id for param"
#                                  "selected_terms_or_topics")
#
#         # 0: find terms or topics to display
#         if data == 'topics':
#             selected_terms_or_topics = [f'topic.{id}' for id in range(1, 71)]
#             title_name = 'topics'
#         elif data == 'terms':
#             vocab = []
#             for t in selected_terms_or_topics:
#                 vocab.append(t)
#             self.get_document_term_matrix(vocabulary=vocab, store_in_df=True)
#             title_name = 'terms'
#         elif data == 'terms_of_topic':
#             vocab = []
#             topic_id = selected_terms_or_topics
#             for term in self.topics[topic_id]['terms_prob']:
#                 if term in self.vocabulary:
#                     vocab.append(term)
#             selected_terms_or_topics = vocab
#             self.get_document_term_matrix(vocabulary=vocab, store_in_df=True)
#             title_name = f'terms of topic {topic_id}'
#         else:
#             raise ValueError('"data" has to be "terms" "topics" or "terms_of_topic"')
#
#         if not title:
#             if display_selector == 'most_frequent':
#                 title = f'Most frequent {title_name} for female (top) and male authors (bottom)'
#             elif display_selector == 'most_divergent':
#                 title = f'Most divergent {title_name} for female (top) and male authors (bottom)'
#             else:
#                 title = f'Most variable {title_name} for female (top) and male authors (bottom)'
#
#         df = self.df
#
#         # 1: Load data
#         data = {}
#         for t in selected_terms_or_topics:
#             data[t] = defaultdict(list)
#         min_freq_total = 1
#         max_freq_total = 0
#
#         for idx, year in enumerate(range(self.start_year, self.end_year)):
#
#             time_slice = df[(df.year >= year - smoothing) & (df.year <= year + smoothing)]
#             time_slice_female = time_slice[time_slice.author_genders == 'female']
#             time_slice_male = time_slice[time_slice.author_genders == 'male']
#
#             for t in selected_terms_or_topics:
#                 freq_total = time_slice[t].mean()
#                 freq_female = time_slice_female[t].mean()
#                 freq_male = time_slice_male[t].mean()
#
#                 # if a term doesn't appear, it is neutral
#                 if (freq_male + freq_female) == 0:
#                     freq_score = 0.5
#                 else:
#                     freq_score = freq_female / (freq_female + freq_male)
#
#                 data[t]['year'].append(year)
#                 data[t]['freq_score'].append(freq_score)
#                 data[t]['freq_total'].append(freq_total)
#
#                 if freq_total < min_freq_total:
#                     min_freq_total = freq_total
#                 if freq_total > max_freq_total:
#                     max_freq_total = freq_total
#
#         for t in terms:
#             data[t]['mean_freq_score'] = np.mean(data[t]['freq_score'])
#             data[t]['mean_freq_total'] = np.mean(data[t]['freq_total'])
#             data[t]['freq_score_range'] = max(data[t]['freq_score']) - min(data[t]['freq_score'])
#
#         # 2: Set up plot
#         fig = plt.figure(figsize=(12, 12))
#         gs = gridspec.GridSpec(nrows=1,
#                                ncols=1,
#                                figure=fig,
#                                width_ratios=[1],
#                                height_ratios=[1],
#                                wspace=0.2, hspace=0.05
#                                )
#
#         ax = fig.add_subplot(gs[0, 0])
#         ax.set_ylim(0, 1)
#         ax.set_xlim(self.start_year + 2, self.end_year -2)
#         ax.set_axisbelow(True)
#         ax.grid(which='major', axis='both')
#
#         dot_scaler = MinMaxScaler((0.0, 50.0))
#         dot_scaler.fit(np.array([min_freq_total, max_freq_total]).reshape(-1, 1))
#         legends = []
#
#         def draw_line(t, t_data, df):
#             """
#             Draws one line depending on t (term or topic string) and t_data (dict of data belonging
#             to t)
#
#             :param t: str
#             :param t_data: dict
#             :return:
#             """
#             y = t_data['freq_score']
#             x = t_data['year']
#             frequencies = t_data['freq_total']
#             if t.startswith('topic.'):
#                 legend = self.topics[int(t[6:])]['name']
#             else:
#                 legend = '{:10s} ({})'.format(t, df[t].sum())
#
#             x_spline = np.linspace(min(x), max(x), ((self.end_year - 2) - (self.start_year + 2)) * 1000)
#             spl = make_interp_spline(x, y, k=1)  # BSpline object
#             y_spline = spl(x_spline)
#
#             line_interpolater = interp1d(x, frequencies)
#             line_widths = line_interpolater(x_spline)
#             line_widths = dot_scaler.transform(line_widths.reshape(-1, 1)).flatten()
#
#             try:
#                 color = sns.color_palette()[len(legends)]
#             except IndexError:
#                 color = sns.cubehelix_palette(100, start=2, rot=0, dark=0, light=.95)[len(legends)]
#
#             ax.scatter(x_spline, y_spline, s=line_widths, antialiased=True,
#                        color=color)
#             legends.append(mpatches.Patch(color=color, label=legend))
#
#         # 3: Plot
#         if display_selector == 'most_frequent':
#             ax.set_title(title, weight='bold', fontsize=18)
#             sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['mean_freq_total'], reverse=True)
#             for t, t_data in sorted_items[:no_terms_or_topics_to_show]:
#                 draw_line(t, t_data, df)
#         elif display_selector == 'most_divergent':
#             ax.set_title(title, weight='bold', fontsize=18)
#             sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['mean_freq_score'], reverse=True)
#             no_disp = no_terms_or_topics_to_show // 2
#             for t, t_data in sorted_items[:no_disp] + sorted_items[::-1][:no_disp]:
#                 draw_line(t, t_data, df)
#         elif display_selector == 'most_variable':
#             ax.set_title(title, weight='bold', fontsize=18)
#             # sort by mean_freq_range second to preserve colors between plots
#             sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['freq_score_range'], reverse=True)
#             sorted_items = sorted_items[:no_terms_or_topics_to_show]
#             sorted_items = sorted(sorted_items, key=lambda k_v: k_v[1]['mean_freq_score'], reverse=True)
#             for t, t_data in sorted_items:
#                 draw_line(t, t_data, df)
#
#         else:
#             raise ValueError('display_selector has to be most_frequent, most_variable, or most_divergent')
#
#         ax.legend(handles=legends, loc=4)
#
#         if show_plot:
#             plt.show()
#         if store_to_filename:
#             fig.savefig(Path('data', store_to_filename))
#
#
# def plot_adviser_gender():
#
#     d = Dataset()
#     d.filter(start_year=1980, end_year=2015)
#
#     fig = plt.figure(figsize=(50, 50))
#     gs = gridspec.GridSpec(nrows=10, ncols=7, figure=fig)
#
#     for ax_id, topic_id in enumerate(range(1, 71)):
#         print(ax_id)
#
#         row = ax_id // 7
#         col = ax_id % 7
#         ax = fig.add_subplot(gs[row, col])
#
#         male = np.zeros(2015-1980+1)
#         female = np.zeros(2015 - 1980 + 1)
#         unknown = np.zeros(2015 - 1980 + 1)
#
#         male_a = np.zeros(2015 - 1980 + 1)
#         female_a = np.zeros(2015 - 1980 + 1)
#
#
#         top = d.copy()
#         top.topic_percentile_score_filter(topic_id, min_percentile_score=80)
#         for _, row in top.df.iterrows():
#             year = row['Year']
#             advisor_gender = row['AdvisorGender']
#             advisee_gender = row['author_genders']
#
#             if row['AdvisorGender'] == 'female':
#                 female[row['Year'] - 1980] += 1
#             elif row['AdvisorGender'] == 'male':
#                 male[row['Year'] - 1980] += 1
#             else:
#                 unknown[row['Year'] - 1980] += 1
#
#             if advisee_gender == 'female':
#                 female_a[year - 1980] += 1
#             elif advisee_gender == 'male':
#                 male_a[year - 1980] += 1
#
#         # sns.lineplot(x=range(1980, 2016), y=unknown, ax=ax, label='unknown')
#         # sns.lineplot(x=range(1980, 2016), y=male, ax=ax, label='male')
#         # sns.lineplot(x=range(1980, 2016), y=female, ax=ax, label='female')
#         # sns.lineplot(x=range(1980, 2016), y=female_a, ax=ax, label='female_a')
#         # sns.lineplot(x=range(1980, 2016), y=male_a, ax=ax, label='male_a')
#         sns.lineplot(x=range(1980, 2016), y=female_a / (female_a+male_a), label='Advisee')
#         sns.lineplot(x=range(1980, 2016), y=female / (female+male), label='Advisor')
#
#
#         ax.legend()
#         ax.set_title(f'{topic_id}: {d.topics[topic_id]["name"]}')
#
#     plt.show()


if __name__ == '__main__':

    pass


