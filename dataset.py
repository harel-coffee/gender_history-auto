
import random
random.seed(0)

from pathlib import Path
import pandas as pd

from topics import TOPICS
from IPython import embed
import html

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
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


    def __init__(self, dataset='dissertations'):


        try:
            self.df = pd.read_csv(Path('data', 'combined_data.csv'), encoding='utf-8')
        except FileNotFoundError:
            self.create_merged_and_cleaned_dataset()
            self.df = pd.read_csv(Path('data', 'combined_data.csv'), encoding='utf-8')


        # earliest and latest year in dataset
        self.start_year = min(self.df.ThesisYear)
        self.end_year = max(self.df.ThesisYear)
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
        if self.institution_filter:
            n += f' {self.institution_filter}'
        if self.advisor_gender_filter:
            n += f' {self.advisor_gender_filter}'
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

    def create_merged_and_cleaned_dataset(self):

        self.df = pd.read_csv(Path('data', 'doc_with_outcome_and_abstract_stm.csv'),
                              #                              encoding='utf-8')
                              encoding='windows-1252')

        print("creating and storing a merged, cleaned dataset at combined_data.csv")

        # creating the tokenized and lemmatized abstract takes time -> do it when the dataset
        # first gets opened and store all tokenized abstracts
        print("tokenizing abstracts")
        wnl = WordNetLemmatizer()
        tokenizer = RegexpTokenizer(r'\b\w\w+\b')
        tokenized_abstracts = []
        for abstract in self.df['Abstract']:
            # abstract contains html entities like &eacute -> remove
            abstract = html.unescape(abstract)
            # I have no idea what encoding proquest / the csv uses but apostrophes are parsed very
            # weirdly -> replace
            abstract = abstract.replace('?óé?¿é?ó', "'")
            tokenized_abstract = " ".join([wnl.lemmatize(t) for t in tokenizer.tokenize(abstract)])
            tokenized_abstract = tokenized_abstract.lower()
            tokenized_abstracts.append(tokenized_abstract)
        self.df['tokenized_abstract'] = tokenized_abstracts

        # 8/14/19: load updated thesis field data from all_data.csv
        fields_df = pd.read_csv(Path('data', 'all_data.csv'), encoding='ISO-8859-1')

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

        # selector_1 = self.df['ThesisProQuestFields'] != 'Business community'
        # selector_2 = self.df['ThesisProQuestFields'] != 'Home economics'
        # selector_3 = self.df['ThesisProQuestFields'] != 'Business costs'
        # selector_4 = self.df['Abstract'].str.contains(pat='histor', case=False) == True
        # self.df['home_econ'] = (selector_1 & selector_2 & selector_3 | selector_4)

        selector_1 = self.df['ThesisProQuestFields'].str.contains('histor', case=False) == True
        selector_2 = self.df['Abstract'].str.contains('histor', case=False) == True
        selector_3 = self.df['ThesisProQuestFields'].str.contains('Middle Ages') == True
        selector_4 = self.df['ThesisProQuestFields'].str.contains('Ancient civilizations') == True
        self.df['is_history'] = (selector_1 | selector_2 | selector_3 | selector_4)

        # plot differences between historical and non-historical data
        hist = self.df[self.df['is_history'] == True]
        non_hist = self.df[self.df['is_history'] == False]
        print(f'Historical dissertations: {len(hist)}. Non-historical dissertations: {len(non_hist)}')
        difs = {}
        for topic_id in range(1, 71):
            dif = abs(np.mean(hist[f'topic.{topic_id}']) - np.mean(non_hist[f'topic.{topic_id}']))
            difs[topic_id] = dif
        sorted_topic_ids = [t[0] for t in sorted(difs.items(), key=lambda x: x[1], reverse=True)]
        #        self.grid_plot_topics(sorted_topic_ids, hue='is_history', store_as_filename='history_filter.png')

        print("Eliminating home economics theses from dataset.")
        self.df = self.df[self.df['is_history'] == True]
        self.df.reset_index(inplace=True)

        self.df.to_csv(Path('data', 'combined_data.csv'))

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
        21634

        # select dissertations that score in the top decile (strongest) for the gender topic (28)
        >>> d.topic_percentile_score_filter(topic=28, min_percentile_score=90)
        >>> print(len(d), min(d.df['percentile_score_topic.28']))
        2164 90.0

        # select 50th-70th decile
        >>> d2 = Dataset()
        >>> d2.topic_percentile_score_filter('topic.28', min_percentile_score=50, max_percentile_score=70)
        >>> len(d2)
        6491

        # filters can be combined
        >>> d3 = Dataset()
        >>> d3.topic_percentile_score_filter(14, min_percentile_score=80)
        >>> d3.topic_percentile_score_filter(28, min_percentile_score=80)
        >>> len(d3)
        866


        """

        if isinstance(topic, int):
            topic = f'topic.{topic}'
        if not f'percentile_score_{topic}' in self.df.columns:
            self.df[f'percentile_score_{topic}'] = self.df[topic].rank(pct=True) * 100 // 10 * 10
        self.df = self.df[self.df[f'percentile_score_{topic}'] >= min_percentile_score]
        self.df = self.df[self.df[f'percentile_score_{topic}'] <= max_percentile_score]
        self.df = self.df.reset_index()
        self.topic_percentile_score_filters.append({
            'topic': topic,
            'min_percentile_score': min_percentile_score,
            'max_percentile_score': max_percentile_score
        })


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

        if advisor_gender:
            if not advisor_gender in ['male', 'female', 'unknown']:
                raise ValueError(f'Author gender needs to be male or female but not {advisor_gender}')
            df = df[df.AdvisorGender == advisor_gender]
            self.advisor_gender = advisor_gender


        if term_filter:
            if term_filter.startswith('not_'):
                term_filter = term_filter[4:]
                df = df[df['tokenized_abstract'].str.contains(pat=term_filter, regex=True) == False]
            else:
                df = df[df['tokenized_abstract'].str.contains(pat=term_filter, regex=True) == True]
            self.term_filter = term_filter

        if institution_filter:
            if institution_filter.startswith('not_'):
                institution_filter = institution_filter[4:]
                df = df[df['ThesisInstitution'].str.contains(institution_filter, case=False) == False]
            else:
                df = df[df['ThesisInstitution'].str.contains(institution_filter, case=False) == True]
            self.institution_filter = institution_filter

        if has_descendants == True:
            df = df[df.AnyDescendants == 1]
            self.descendants_filter = True

        if has_descendants == False:
            df = df[df.AnyDescendants == 0]
            self.descendants_filter = False

        self.df = df.reset_index()
        return self


    def grid_plot_topics(self, sorted_topic_ids, hue,
                         y_max=None, df=None, show_plot=True, store_as_filename=None):

        """
        Can be used to plot a 10x7 grid of all topics

        :param df:
        :param sorted_topic_ids: list(int)
        :param hue:
        :param y_max:
        :return:

        # show topic distribution from most female to most male
        >>> d = Dataset()
        >>> male = d.copy().filter(author_gender='male')
        >>> female = d.copy().filter(author_gender='female')
        >>> difs = {}
        >>> for topic_id in range(1, 71):
        ...     dif = np.mean(female.df[f'topic.{topic_id}']) - np.mean(male.df[f'topic.{topic_id}'])
        ...     difs[topic_id] = dif
        >>> sorted_topic_ids =  [t[0] for t in sorted(difs.items(), key = lambda x: x[1], reverse=True)]
        >>> d.grid_plot_topics(sorted_topic_ids, hue='AdviseeGender')

        """

        if not df:
            df = self.df

        fig = plt.figure(figsize=(50,50))
        gs = gridspec.GridSpec(nrows=10, ncols=7, figure=fig)

        for ax_id, topic_id in enumerate(sorted_topic_ids):
            print(ax_id, topic_id)
            row = ax_id // 7
            col = ax_id % 7
            ax = fig.add_subplot(gs[row, col])
            ax = sns.lineplot(x='ThesisYear', y=f'topic.{topic_id}', hue=hue,
                              data=df, ax=ax)
            ax.set_title(f'{topic_id}: {TOPICS[topic_id]["name"]}')
            ax.set_xlim(1980, 2015)
            if y_max:
                ax.set_ylim(0, y_max)

        if show_plot:
            plt.show()
        if store_as_filename:
            fig.savefig(Path('data', 'plots', store_as_filename))




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

    def get_vocabulary(self, exclude_stopwords=True, max_terms=None, min_appearances=None,
                       include_2grams=False):
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

        if not max_terms:
            max_terms = 1000000

        vocabulary = Counter()
        for abstract in self.df['tokenized_abstract']:
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

            # for term in :
            #     try:
            #         del vocabulary[term]
            #     except KeyError:
            #         pass
            #
            #
            #     # TODO: fix this ugly implementation
            #     for term2 in stop_words.ENGLISH_STOP_WORDS.union({'wa', 'ha', 'óé', 'dotbelow'}):
            #         try:
            #             del vocabulary['{} {}'.format(term, term2)]
            #         except KeyError:
            #             pass


        vocab_list = []
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

    def print_dissertations_mentioning_terms_or_topics(self, terms, no_dissertations=5):
        """
        print dissertations that mention specific terms or topics,
        can be weighted or unweighted

        :param terms: list or dict
        :param no_dissertations: number of dissertations to print, default: 5

        >>> d = Dataset()
        >>> terms = ['woman', 'female', 'women', 'feminist', 'gender']
        >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
        2014 Author: female  Advisor: female  Authors, Activists, Apostles: Women's Leadership in the New Christian Right
        2006 Author: female  Advisor: male    Women on the march: Gender and anti-fascism in American communism, 1935--1939

        # by default, each term has weight=1. However, terms can also be a dict of term weights
        >>> terms = {'nurse': 1, 'drugs': 10}
        >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
        1997 Author: female  Advisor: male    Regulating beauty: Cosmetics in American culture from the 1906 Pure Food and Drugs Act to the 1938 Food, Drug and Cosmetic Act
        1994 Author: female  Advisor: female  G.I. nurses at war: Gender and professionalization in the Army Nurse Corps during World War II

        # topics also work:
        >>> terms = ['topic.28']
        >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=2)
        1989 Author: female  Advisor: unknown Day nurseries and wage-earning mothers in the United States, 1890-1930
        1988 Author: female  Advisor: female  "Women adrift" and "urban pioneers": Self-supporting working women in America, 1880-1930

        >>> terms = ['gay', 'homosexual', 'homosexuality', 'masculinity']
        >>> d.print_dissertations_mentioning_terms_or_topics(terms=terms, no_dissertations=20)

        """

        # if input is a list, give each term weight 1
        if isinstance(terms, list):
            terms = {t: 1 for t in terms}

        if list(terms.keys())[0].startswith('topic.'):
            self.df['dissertation_weights'] = 0
            for topic, weight in terms.items():
                self.df['dissertation_weights'] += weight * self.df[topic]
            for _, row in self.df.sort_values(by='dissertation_weights', ascending=False)[:no_dissertations].iterrows():
                print('{} Author: {:7s} Advisor: {:7s} {}'.format(
                    row['ThesisYear'], row['AdviseeGender'],
                    row['AdvisorGender'], row['ThesisTitle']
                ))
        else:
            dtm = self.get_document_term_matrix(vocabulary=terms.keys())
            weights = np.array(list(terms.values()))
            scores = dtm * weights

            for i in scores.argsort()[::-1][:no_dissertations]:


                print('{} Author: {:7s} Advisor: {:7s} {}'.format(
                    self.df['ThesisYear'][i], self.df['AdviseeGender'][i],
                    self.df['AdvisorGender'][i], self.df['ThesisTitle'][i]
                ))

    def print_examples_of_term_in_context(self, term, no_examples=10):
        """
        Finds examples of a term in the abstracts and prints them

        >>> d = Dataset()
        >>> d.print_examples_of_term_in_context('hegel', 2)
        1987 PHILIP SCHAFF (1819-1893): PORTRAIT OF AN IMMIGRANT THEOLOGIAN         heology that accommodated such figure a hegel and schleiermacher tolerated liberal position yet rema
        1995 State, society, and the market: Karl Sigmund Altenstein and the langua n idealism first by fichte and later by hegel thus the importance of his relationship to fichte and

        :param term:
        :return:
        """

        df = self.df[self.df['tokenized_abstract'].str.contains(pat=r'\b{}\b'.format(term))]
        if len(df) == 0:
            print(f'No dissertation abstracts mention {term}.')
            return

        while True:
            try:
                samples = df.sample(no_examples)
                break
            except ValueError:
                no_examples -= 1

        print(f'\n Found {len(df)} examples of {term}.')
        for  _, row in samples.iterrows():

            pos = row['tokenized_abstract'].find(term)
            if pos > -1:
                text = row['tokenized_abstract'][max(0, pos-40):pos+60]
                print('{} {:7s} {:70s} {}'.format(
                    row['ThesisYear'], row['AdviseeGender'], row['ThesisTitle'][:70], text,
                ))


    def normalize_dataset_by_5year_interval(self, no_docs_per_5year_interval=5000):

#        dfs = {}

        docs = []

        for years_range in [(1976, 1984), (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2004),
                            (2005, 2009), (2010, 2015)]:
            y1, y2 = years_range
            if self.start_year >= y2 or self.end_year < y2:
                continue
#            dfs[years_range] = self.df[(self.df['ThesisYear'] >= y1) & (self.df['ThesisYear'] <= y2)]
            df = self.df[(self.df['ThesisYear'] >= y1) & (self.df['ThesisYear'] <= y2)]
            df = df.to_dict('records')

            if len(df) == 0:
                raise IndexError(f'Cannot generate dataset of {no_docs_per_5year_interval} docs for {y1}-{y2} for'
                      f' {self.name_full} with 0 docs.')
            if len(df) < 50:
                print(f'WARNING. Generating dataset of {no_docs_per_5year_interval} docs for {y1}-{y2} for'
                      f' {self.name} with only {len(df)} docs.')

            for i in range(no_docs_per_5year_interval):
                docs.append(random.sample(df, 1)[0])
        self.df = pd.DataFrame(docs)




def plot_adviser_gender():

    d = Dataset()
    d.filter(start_year=1980, end_year=2015)

    fig = plt.figure(figsize=(50, 50))
    gs = gridspec.GridSpec(nrows=10, ncols=7, figure=fig)

    for ax_id, topic_id in enumerate(range(1, 71)):
        print(ax_id)

        row = ax_id // 7
        col = ax_id % 7
        ax = fig.add_subplot(gs[row, col])

        male = np.zeros(2015-1980+1)
        female = np.zeros(2015 - 1980 + 1)
        unknown = np.zeros(2015 - 1980 + 1)

        male_a = np.zeros(2015 - 1980 + 1)
        female_a = np.zeros(2015 - 1980 + 1)


        top = d.copy()
        top.topic_percentile_score_filter(topic_id, min_percentile_score=80)
        for _, row in top.df.iterrows():
            year = row['ThesisYear']
            advisor_gender = row['AdvisorGender']
            advisee_gender = row['AdviseeGender']

            if row['AdvisorGender'] == 'female':
                female[row['ThesisYear'] - 1980] += 1
            elif row['AdvisorGender'] == 'male':
                male[row['ThesisYear'] - 1980] += 1
            else:
                unknown[row['ThesisYear'] - 1980] += 1

            if advisee_gender == 'female':
                female_a[year - 1980] += 1
            elif advisee_gender == 'male':
                male_a[year - 1980] += 1

        # sns.lineplot(x=range(1980, 2016), y=unknown, ax=ax, label='unknown')
        # sns.lineplot(x=range(1980, 2016), y=male, ax=ax, label='male')
        # sns.lineplot(x=range(1980, 2016), y=female, ax=ax, label='female')
        # sns.lineplot(x=range(1980, 2016), y=female_a, ax=ax, label='female_a')
        # sns.lineplot(x=range(1980, 2016), y=male_a, ax=ax, label='male_a')
        sns.lineplot(x=range(1980, 2016), y=female_a / (female_a+male_a), label='Advisee')
        sns.lineplot(x=range(1980, 2016), y=female / (female+male), label='Advisor')


        ax.legend()
        ax.set_title(f'{topic_id}: {TOPICS[topic_id]["name"]}')

    plt.show()








    def grid_plot_topics(self, sorted_topic_ids, hue,
                         y_max=None, df=None, show_plot=True, store_as_filename=None):

        """
        Can be used to plot a 10x7 grid of all topics

        :param df:
        :param sorted_topic_ids: list(int)
        :param hue:
        :param y_max:
        :return:

        # show topic distribution from most female to most male
        >>> d = Dataset()
        >>> male = d.copy().filter(author_gender='male')
        >>> female = d.copy().filter(author_gender='female')
        >>> difs = {}
        >>> for topic_id in range(1, 71):
        ...     dif = np.mean(female.df[f'topic.{topic_id}']) - np.mean(male.df[f'topic.{topic_id}'])
        ...     difs[topic_id] = dif
        >>> sorted_topic_ids =  [t[0] for t in sorted(difs.items(), key = lambda x: x[1], reverse=True)]
        >>> d.grid_plot_topics(sorted_topic_ids, hue='AdviseeGender')

        """

        if not df:
            df = self.df

        fig = plt.figure(figsize=(50,50))
        gs = gridspec.GridSpec(nrows=10, ncols=7, figure=fig)

        for ax_id, topic_id in enumerate(sorted_topic_ids):
            print(ax_id, topic_id)
            row = ax_id // 7
            col = ax_id % 7
            ax = fig.add_subplot(gs[row, col])
            ax = sns.lineplot(x='ThesisYear', y=f'topic.{topic_id}', hue=hue,
                              data=df, ax=ax)
            ax.set_title(f'{topic_id}: {TOPICS[topic_id]["name"]}')
            ax.set_xlim(1980, 2015)
            if y_max:
                ax.set_ylim(0, y_max)

        if show_plot:
            plt.show()
        if store_as_filename:
            fig.savefig(Path('data', 'plots', store_as_filename))






if __name__ == '__main__':
    d = Dataset('dissertations')
    d.filter(institution_filter='harvard')


    plot_adviser_gender()


