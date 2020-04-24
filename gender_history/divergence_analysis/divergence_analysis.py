
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

from gender_history.datasets.dataset import Dataset
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.divergence_analysis.stats import StatisticalAnalysis



from scipy.sparse import vstack, csr_matrix
from IPython import embed

import re

class DivergenceAnalysis():

    def __init__(self,
                 master_corpus: Dataset,
                 sub_corpus1: Dataset,
                 sub_corpus2: Dataset,
                 sub_corpus1_name: str=None,
                 sub_corpus2_name: str=None,
                 analysis_type: str = 'terms',
                 sort_by: str='dunning',
                 compare_to_overall_weights: bool=False):

        self.mc = master_corpus
        self.c1 = sub_corpus1
        self.c2 = sub_corpus2
        if not sub_corpus1_name:
            sub_corpus1_name = sub_corpus1.name
        self.c1_name = sub_corpus1_name
        if not sub_corpus2_name:
            sub_corpus2_name = sub_corpus2.name
        self.c2_name = sub_corpus2_name
        self.analysis_type = analysis_type
        self.sort_by = sort_by
        self.compare_to_overall_weights = compare_to_overall_weights

    def run_divergence_analysis(self,
                                number_of_terms_or_topics_to_print: int=30,
                                print_results: bool=True,
                                min_appearances_per_term: int=50):

        if not self.analysis_type in {'terms', 'topics', 'gen_approach'}:
            raise ValueError(f'analysis_type has to be "terms", "topics", or "gen_approach".')

        self.min_appearances_per_term = min_appearances_per_term

        if self.sort_by is None:
            if self.analysis_type == 'terms':
                self.sort_by = 'dunning'
            else:
                self.sort_by = 'frequency_score'

        self._initialize_analysis_and_dtms()
        self.output_data_df = self._generate_output_data()

        if print_results:
            self._print_results(number_of_terms_or_topics_to_print)

        return self.output_data_df

    def _initialize_analysis_and_dtms(self):

        if self.analysis_type == 'terms':
            # self.vocabulary = self.mc.get_vocabulary(max_terms=10000,
            #                                  min_appearances=self.min_appearances_per_term,
            #                                  include_2grams=True)
            # self.c1_dtm = self.c1.get_document_term_matrix(vocabulary=self.vocabulary)
            # self.c2_dtm = self.c2.get_document_term_matrix(vocabulary=self.vocabulary)

            # self.vocabulary = self.mc.get_default_vocabulary(no_terms=10000)
            mc_dtm, vocabulary = self.mc.get_vocabulary_and_document_term_matrix(
                max_features=10000, exclude_stop_words=True)


            # only retain up to 1000 words that appear at least 1000 times in the corpus
            # appear at least 1000 times -> a single article on a topic cannot cause a huge spike
            # 1000 words -> we're interested in key terms
            count_sums = np.array(mc_dtm.sum(axis=0)).flatten()
            assert len(vocabulary) == len(count_sums)
            count_sorted = sorted([(count_sums[i], vocabulary[i]) for i in range(len(vocabulary))], reverse=True)
            self.vocabulary = []
            for count, term in count_sorted:
                if (count > 1000 and len(self.vocabulary) < 1000) or term == 'gay':
                    self.vocabulary.append(term)

            print(f'Vocabulary length: {len(self.vocabulary)}')

            self.c1_dtm, _ = self.c1.get_vocabulary_and_document_term_matrix(vocabulary=self.vocabulary)
            self.c2_dtm, _ = self.c2.get_vocabulary_and_document_term_matrix(vocabulary=self.vocabulary)

        else:
            if self.analysis_type == 'topics':
                self.vocabulary = [f'topic.{i}' for i in range(1, 91)]
            elif self.analysis_type == 'gen_approach':
                self.vocabulary = [x for x in self.mc.df.columns if x.startswith('gen_approach_')]
            else:
                raise NotImplemented(f"analysis for {self.analysis_type} not yet implemented.")

            # multiply topic weights with text length to get an estimate for number of words
            # belonging to each topic.
            c1_text_len_arr = np.array(self.c1.df.m_text_len)
            c2_text_len_arr = np.array(self.c2.df.m_text_len)
            self.c1_dtm = self.c1.get_document_topic_matrix(vocabulary=self.vocabulary)
            self.c2_dtm = self.c2.get_document_topic_matrix(vocabulary=self.vocabulary)
            assert self.c1_dtm.shape[0] == len(c1_text_len_arr)
            assert self.c2_dtm.shape[0] == len(c2_text_len_arr)
            # multiply each dtm row with the number of terms
            self.c1_dtm = csr_matrix((self.c1_dtm.T.toarray() * c1_text_len_arr).T)
            self.c2_dtm = csr_matrix((self.c2_dtm.T.toarray() * c2_text_len_arr).T)

            c1_col_sums = np.array(self.c1_dtm.sum(axis=1)).flatten()
            c2_col_sums = np.array(self.c2_dtm.sum(axis=1)).flatten()


            # one topic can belong to multiple general approaches, so don't run for that.
            if self.analysis_type in ['terms', 'topics']:
                assert np.allclose(c1_col_sums, c1_text_len_arr, rtol=1e-04, atol=1e-06)
                assert np.allclose(c2_col_sums, c2_text_len_arr, rtol=1e-04, atol=1e-06)

        self.master_dtm = vstack([self.c1_dtm, self.c2_dtm])



    def _generate_output_data(self):

        s = StatisticalAnalysis(self.master_dtm, self.c1_dtm, self.c2_dtm, self.vocabulary)
        dunning, _ = s.dunning_log_likelihood()
        frequency_score, _ = s.frequency_score()

        total_terms_all = self.master_dtm.sum()
        total_terms_c1 = self.c1_dtm.sum()
        total_terms_c2 = self.c2_dtm.sum()

        column_sums_all = np.array(self.master_dtm.sum(axis=0))[0]
        column_sums_c1 = np.array(self.c1_dtm.sum(axis=0))[0]
        column_sums_c2 = np.array(self.c2_dtm.sum(axis=0))[0]

        if self.analysis_type == 'topics' and self.compare_to_overall_weights:
            d = JournalsDataset()
            dc1 = d.copy().filter(author_gender='female')
            dc2 = d.copy().filter(author_gender='male')
            jdiv = DivergenceAnalysis(d, dc1, dc2, sub_corpus1_name='women', sub_corpus2_name='men',
                                      analysis_type='topics', sort_by='dunning')
            ddf = jdiv.run_divergence_analysis(print_results=False)

        data = []
        for term_idx in range(len(self.vocabulary)):

            count_all = column_sums_all[term_idx]
            count_c1 = column_sums_c1[term_idx]
            count_c2 = column_sums_c2[term_idx]

            if self.analysis_type == 'terms':
                term = self.vocabulary[term_idx]
                if count_all < self.min_appearances_per_term:
                    continue
            else:
                topic_idx = term_idx + 1
                if self.analysis_type == 'topics':
                    topic_name = self.c1.topics[topic_idx]["name"]
                    if topic_name.startswith('Noise'):
                        continue
                    else:
                        term = f'({topic_idx}) {topic_name}'
                elif self.analysis_type == 'gen_approach':
                    term = self.vocabulary[term_idx]

            datum = {
                f'{self.analysis_type}': term,
                'dunning': dunning[term_idx],
                'frequency_score': frequency_score[term_idx],
                'count both': count_all,
                f'c {self.c1_name}': count_c1,
                f'c {self.c2_name}': count_c2,
                'freq both': count_all / total_terms_all,
                f'f {self.c1_name}': count_c1 / total_terms_c1,
                f'f {self.c2_name}': count_c2 / total_terms_c2,
            }
            if self.analysis_type == 'topics':
                datum['topic_id'] = topic_idx
            if self.analysis_type == 'topics' and self.compare_to_overall_weights:
                r = ddf[ddf.topic_id == topic_idx].iloc[0]
                datum['fs_comp_to_overall'] = datum['frequency_score'] - r['frequency_score']


            data.append(datum)

            if self.analysis_type == 'terms':
                if datum['terms'].find('child') > -1:
                    print(datum)
                if datum['terms'] == 'gay':
                    print(datum)


        df = pd.DataFrame(data)
        df.sort_values(by=self.sort_by, inplace=True)
        df.reset_index(inplace=True)

        return df

    def _print_results(self, number_of_terms_or_topics_to_print):
        """
        Prints the results in a table

        :param number_of_terms_or_topics_to_print:
        :return:
        """

        if self.analysis_type == 'terms':
            headers = ['terms', 'dunning', 'frequency_score',
                       'count both', f'c {self.c1_name}', f'c {self.c2_name}']
        else:
            if self.analysis_type == 'topics' and self.compare_to_overall_weights:
                headers = [f'{self.analysis_type}', 'dunning', 'frequency_score',
                           'fs_comp_to_overall',
                           'freq both', f'f {self.c1_name}', f'f {self.c2_name}']
            else:
                headers = [f'{self.analysis_type}', 'dunning', 'frequency_score', 'freq both',
                       f'f {self.c1_name}', f'f {self.c2_name}']

        year_df = {}

        for years_range in [
            (1951, 1954), (1955, 1959), (1960, 1964), (1965, 1969),
            (1970, 1974), (1975, 1979), (1980, 1984),
            (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2004),
             (2005, 2009), (2010, 2015)]:
            y1, y2 = years_range
            c1_count = len(self.c1.df[(self.c1.df['m_year'] >= y1) & (self.c1.df['m_year'] <= y2)])
            c2_count = len(self.c2.df[(self.c2.df['m_year'] >= y1) & (self.c2.df['m_year'] <= y2)])
            if c1_count > 0 or c2_count > 0:
                year_df[f'{y1}-{y2}'] = {
                    f'{self.c1_name}': c1_count,
                    f'{self.c1_name} freq': c1_count / len(self.c1),
                    f'{self.c2_name}': c2_count,
                    f'{self.c2_name} freq': c2_count / len(self.c2),
                }
        year_df = pd.DataFrame(year_df).transpose()
        print(tabulate(year_df, headers='keys'))


        print(f'\n\n{self.analysis_type} distinctive for Corpus 1: {self.c1_name}. {len(self.c1)} Documents\n')
        print(tabulate(self.output_data_df[headers][::-1][0:number_of_terms_or_topics_to_print], headers='keys'))

        print(f'\n\n{self.analysis_type} distinctive for Corpus 2: {self.c2_name}. {len(self.c2)} Documents\n')
        print(tabulate(self.output_data_df[headers][0:number_of_terms_or_topics_to_print], headers='keys'))

    def print_articles_for_top_topics(self, top_terms_or_topics=3, articles_per_term_or_topic=3):

        if self.analysis_type == 'terms':
            self.c1.get_vocabulary_and_document_term_matrix(vocabulary=self.vocabulary, store_in_df=True,
                                                            use_frequencies=False)
            self.c2.get_vocabulary_and_document_term_matrix(vocabulary=self.vocabulary, store_in_df=True,
                                                            use_frequencies=False)


        print(f'\nSample articles of distinctive {self.analysis_type} for {self.c1_name}')
        for _, row in self.output_data_df.iloc[::-1].iloc[:top_terms_or_topics].iterrows():


            if self.analysis_type == 'topics':
                topic_id = row['topic_id']
                column_str = f'topic.{topic_id}'
                print(f'\nTopic {topic_id} ({self.mc.topics[topic_id]["name"]}). Highest scoring items:')

            else:
                column_str = row['term']
                print(f'\n Term: {column_str}. Highest scoring items:')

            for _, row in self.c1.df.sort_values(by=column_str, ascending=False).iloc[:articles_per_term_or_topic].iterrows():

                if self.analysis_type == 'terms':
                    count_term = re.findall(r'\b\w\w+\b', row.m_text.lower()).count(column_str)
                    print(f'   Count {column_str}: {count_term}. ({row.m_year}) {row.m_authors}: {row.m_title}')
                else:
                    print(f'   ({row.m_year}) {row.m_authors}: {row.m_title}')


        print(f'\n\nSample articles for distinctive {self.analysis_type} for {self.c2_name}')
        for _, row in self.output_data_df.iloc[:top_terms_or_topics].iterrows():

            if self.analysis_type == 'topics':
                topic_id = row['topic_id']
                column_str = f'topic.{topic_id}'
                print(f'\nTopic {topic_id} ({self.mc.topics[topic_id]["name"]}). Highest scoring items:')

            else:
                column_str = row['term']
                print(f'\n Term: {column_str}. Highest scoring items:')

            for _, row in self.c2.df.sort_values(by=column_str, ascending=False).iloc[:articles_per_term_or_topic].iterrows():
                if self.analysis_type == 'terms':
                    count_term = re.findall(r'\b\w\w+\b', row.m_text.lower()).count(column_str)
                    print(f'   Count {column_str}: {count_term}. ({row.m_year}) {row.m_authors}: {row.m_title}')
                else:
                    print(f'   ({row.m_year}) {row.m_authors}: {row.m_title}')


if __name__ == '__main__':

    # generate_average_overall_freq_scores()


    # d = DissertationDataset()
    d = JournalsDataset()
    # d.filter(start_year=1990)

    # d.topic_score_filter(31, min_topic_weight=0.1)
    #d.topic_score_filter(21, min_percentile_score=95)

    # d.filter(term_filter={'term':'wom[ae]n', 'min_count': 10})
    # d.filter(term_filter={'term':'gender', 'min_count': 10})


    # d = d.topic_percentile_score_filter(topic_id=61, min_percentile_score=80)
    # d = d.filter(term_filter='childhood')

    # Create two sub-datasets, one for female authors and one for male authors
    c1 = d.copy().filter(term_filter={'term': 'gender', 'min_count': 10})
    c2 = d.copy().filter(term_filter={'term': 'women', 'min_count': 10})
    #
    # c1 = d.copy().topic_score_filter(71, min_percentile_score=90)
    # c2 = d.copy().topic_score_filter(71, max_percentile_score=89)
    #
    # c1 = d.copy().filter(term_filter='gay')
    # c2 = d.copy().filter(term_filter='not_gay')

    print(len(c1), len(c2), len(d))


    # Run the divergence analysis
    div = DivergenceAnalysis(d, c1, c2, sub_corpus1_name='gender', sub_corpus2_name='women',
                             analysis_type='topics', sort_by='dunning', compare_to_overall_weights=True)
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)

    div.print_articles_for_top_topics(top_terms_or_topics=10, articles_per_term_or_topic=5)

    embed()


#     d = JournalsDataset(use_equal_samples_dataset=False)
#     # c1 = d.copy().filter(author_gender='female')
#     # c2 = d.copy().filter(author_gender='male')
# #    c1 = d.copy().filter(term_filter='women')
# #    c2 = d.copy().filter(term_filter='not_women')
#     c1 = d.copy().topic_percentile_score_filter(topic_id=25, min_percentile_score=80)
#     c2 = d.copy().topic_percentile_score_filter(topic_id=45, min_percentile_score=80)
#
#     div = DivergenceAnalysis(d, c1, c2, sub_corpus1_name='men', sub_corpus2_name='women')
#     div.run_divergence_analysis()


