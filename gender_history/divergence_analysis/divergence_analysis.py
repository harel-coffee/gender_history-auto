
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

from gender_history.datasets.dataset import Dataset
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.divergence_analysis.stats import StatisticalAnalysis

from scipy.sparse import vstack
from IPython import embed


class DivergenceAnalysis():

    def __init__(self,
                 master_corpus: Dataset,
                 sub_corpus1: Dataset,
                 sub_corpus2: Dataset,
                 sub_corpus1_name: str=None,
                 sub_corpus2_name: str=None):

        self.mc = master_corpus
        self.c1 = sub_corpus1
        self.c2 = sub_corpus2
        if not sub_corpus1_name:
            sub_corpus1_name = sub_corpus1.name
        self.c1_name = sub_corpus1_name
        if not sub_corpus2_name:
            sub_corpus2_name = sub_corpus2.name
        self.c2_name = sub_corpus2_name
        self.analysis_type = None

    def run_divergence_analysis(self,
                                analysis_type: str='terms',
                                number_of_terms_or_topics_to_print: int=30,
                                print_results: bool=True,
                                sort_by: str=None,
                                min_appearances_per_term: int=50):

        if self.analysis_type is not None:
            print("WARNING! Running divergence analysis with pre-initialized analysis config. "
                  "You are likely re-running an analysis with a pre-initialized "
                  "DivergenceAnalysis object. Are you certain that's what you want to do?")

        if not analysis_type in {'terms', 'topics', 'gen_approach'}:
            raise ValueError(f'analysis_type has to be "terms", "topics", or "gen_approach".')

        self.analysis_type = analysis_type
        self.min_appearances_per_term = min_appearances_per_term
        self.sort_by = sort_by

        if self.sort_by is None:
            if analysis_type == 'terms':
                self.sort_by = 'dunning'
            else:
                self.sort_by = 'frequency_score'

        self._initialize_analysis_and_dtms()
        output_data_df = self._generate_output_data()
        if print_results:
            self._print_results(output_data_df, number_of_terms_or_topics_to_print)

        return output_data_df

    def _initialize_analysis_and_dtms(self):

        if self.analysis_type == 'terms':
            # self.vocabulary = self.mc.get_vocabulary(max_terms=10000,
            #                                  min_appearances=self.min_appearances_per_term,
            #                                  include_2grams=True)
            # self.c1_dtm = self.c1.get_document_term_matrix(vocabulary=self.vocabulary)
            # self.c2_dtm = self.c2.get_document_term_matrix(vocabulary=self.vocabulary)

            # self.vocabulary = self.mc.get_default_vocabulary(no_terms=10000)
            _, self.vocabulary = self.mc.get_vocabulary_and_document_term_matrix(max_features=50000)
            self.c1_dtm, _ = self.c1.get_vocabulary_and_document_term_matrix(vocabulary=self.vocabulary)
            self.c2_dtm, _ = self.c2.get_vocabulary_and_document_term_matrix(vocabulary=self.vocabulary)

        else:
            if self.analysis_type == 'topics':
                self.vocabulary = [f'topic.{i}' for i in range(1, 91)]
            elif self.analysis_type == 'gen_approach':
                self.vocabulary = [x for x in self.mc.df.columns if x.startswith('gen_approach_')]
            else:
                raise NotImplemented(f"analysis for {self.analysis_type} not yet implemented.")

            # TODO: don't use 4000 word default
            self.c1_dtm = self.c1.get_document_topic_matrix(vocabulary=self.vocabulary) * 4000
            self.c2_dtm = self.c2.get_document_topic_matrix(vocabulary=self.vocabulary) * 4000

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
                    term = f'({topic_idx}) {self.c1.topics[topic_idx]["name"]}'
                elif self.analysis_type == 'gen_approach':
                    term = self.vocabulary[term_idx]

            data.append({
                'term': term,
                'dunning': dunning[term_idx],
                'frequency_score': frequency_score[term_idx],
                'count both': count_all,
                f'c {self.c1_name}': count_c1,
                f'c {self.c2_name}': count_c2,
                'freq both': count_all / total_terms_all,
                f'f {self.c1_name}': count_c1 / total_terms_c1,
                f'f {self.c2_name}': count_c2 / total_terms_c2,
            })

        df = pd.DataFrame(data)
        df.sort_values(by=self.sort_by, inplace=True)
        df.reset_index(inplace=True)

        return df

    def _print_results(self, df, number_of_terms_or_topics_to_print):
        """
        Prints the results in a table

        :param number_of_terms_or_topics_to_print:
        :return:
        """

        if self.analysis_type == 'terms':
            headers = ['term', 'dunning', 'frequency_score',
                       'count both', f'c {self.c1_name}', f'c {self.c2_name}']
        else:
            headers = ['term', 'dunning', 'frequency_score', 'freq both',
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

        # embed()


        print(f'\n\nTerms distinctive for Corpus 1: {self.c1_name}. {len(self.c1)} Documents\n')
        print(tabulate(df[headers][::-1][0:number_of_terms_or_topics_to_print], headers='keys'))

        print(f'\n\nTerms distinctive for Corpus 2: {self.c2_name}. {len(self.c2)} Documents\n')
        print(tabulate(df[headers][0:number_of_terms_or_topics_to_print], headers='keys'))

#
# def divergence_analysis(master_dataset:Dataset,
#                         c1:Dataset,                 # sub corpus 1
#                         c2:Dataset,                 # sub corpus 2
#                         analysis_type='terms',
#                         number_of_terms_to_print=30,
#                         c1_name=None, c2_name=None,
#                         print_results=True, sort_by=None,
#                         min_appearances_per_term=50):
#     if not c1_name:
#         c1_name = c1.name
#     if not c2_name:
#         c2_name = c2.name
#
#     if not sort_by:
#         if analysis_type == 'terms':
#             sort_by = 'dunning'
#         else:
#             sort_by = 'frequency_score'
#
#
#     if analysis_type == 'terms':
#         vocabulary = master_dataset.get_vocabulary(max_terms=10000,
#                                                    min_appearances=min_appearances_per_term,
#                                                    include_2grams=True)
#         c1_dtm = c1.get_document_term_matrix(vocabulary=vocabulary)
#         c2_dtm = c2.get_document_term_matrix(vocabulary=vocabulary)
#     else:
#         if analysis_type == 'topics':
#             vocabulary = [f'X{i}' for i in range(1, 101)]
#         elif analysis_type == 'gen_approach':
#             vocabulary = [x for x in master_dataset.df.columns if x.startswith('gen_approach_')]
#         else:
#             raise NotImplemented(f"analysis for {analysis_type} not yet implemented.")
#         c1_dtm = c1.get_document_topic_matrix(vocabulary=vocabulary) * 4000
#         c2_dtm = c2.get_document_topic_matrix(vocabulary=vocabulary) * 4000
#
#     master_dtm = vstack([c1_dtm, c2_dtm])
#
#     s = StatisticalAnalysis(master_dtm, c1_dtm, c2_dtm, vocabulary)
#     dunning, _ = s.dunning_log_likelihood()
#     frequency_score, _ = s.frequency_score()
# #    mwr, _ = s.mann_whitney_rho()
# #    correlated_terms = s.correlation_coefficient()
#
#     total_terms_all = master_dtm.sum()
#     total_terms_c1 = c1_dtm.sum()
#     total_terms_c2 = c2_dtm.sum()
#
#     column_sums_all = np.array(master_dtm.sum(axis=0))[0]
#     column_sums_c1 = np.array(c1_dtm.sum(axis=0))[0]
#     column_sums_c2 = np.array(c2_dtm.sum(axis=0))[0]
#
#     data = []
#     for term_idx in range(len(vocabulary)):
#
#         count_all = column_sums_all[term_idx]
#         count_c1 = column_sums_c1[term_idx]
#         count_c2 = column_sums_c2[term_idx]
#
#
#         if analysis_type == 'terms':
#             term = vocabulary[term_idx]
#             if count_all < min_appearances_per_term:
#                 continue
#         else:
#             topic_idx = term_idx + 1
#             if analysis_type == 'topics':
#                 term = f'({topic_idx}) {c1.topics[topic_idx]["name"]}'
#             elif analysis_type == 'gen_approach':
#                 term = vocabulary[term_idx]
#
#         data.append({
#             'term': term,
#             'dunning': dunning[term_idx],
#             'frequency_score': frequency_score[term_idx],
#             'count_total': count_all,
#             f'count {c1_name}': count_c1,
#             f'count {c2_name}': count_c2,
#             'frequency_total': count_all / total_terms_all,
#             f'frequency {c1_name}': count_c1 / total_terms_c1,
#             f'frequency {c2_name}': count_c2 / total_terms_c2,
#         })
#
#
#     df = pd.DataFrame(data)
#
#
#     df.sort_values(by=sort_by, inplace=True)
#     df.reset_index(inplace=True)
#
#     if print_results:
#
#         if analysis_type == 'terms':
#             headers = ['term', 'dunning', 'frequency_score', 'count_total',
#                        f'count {c1_name}', f'count {c2_name}']
#         else:
#             headers = ['term', 'dunning', 'frequency_score', 'frequency_total',
#                        f'frequency {c1_name}', f'frequency {c2_name}']
#
#         year_df = {}
#
#         for years_range in [(1976, 1984), (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2004),
#                              (2005, 2009), (2010, 2015)]:
#             y1, y2 = years_range
#             c1_count = len(c1.df[(c1.df['year'] >= y1) & (c1.df['year'] <= y2)])
#             c2_count = len(c2.df[(c2.df['year'] >= y1) & (c2.df['year'] <= y2)])
#             if c1_count > 0 or c2_count > 0:
#                 year_df[f'{y1}-{y2}'] = {
#                     f'{c1_name}': c1_count,
#                     f'{c1_name} freq': c1_count / len(c1),
#                     f'{c2_name}': c2_count,
#                     f'{c2_name} freq': c2_count / len(c2),
#                 }
#         year_df = pd.DataFrame(year_df).transpose()
#         print(tabulate(year_df, headers='keys'))
#
#         # embed()
#
#
#         print(f'\n\nTerms distinctive for Corpus 1: {c1_name}. {len(c1)} Theses\n')
#         print(tabulate(df[headers][::-1][0:number_of_terms_to_print], headers='keys'))
#
#         print(f'\n\nTerms distinctive for Corpus 2: {c2_name}. {len(c2)} Theses\n')
#         print(tabulate(df[headers][0:number_of_terms_to_print], headers='keys'))
#
#
#     return df

def wordcloud(gender='female', relative_scaling=0.0):

    # local imports so Pillow and wordclouds are not hard requirements for running any code
    from PIL import Image
    from wordcloud import WordCloud

    def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

        color = icon.getpixel((int(position[1] + 2), int(position[0] + 2)))

        print(color)

        if color[0] == color[1] == color[2]: color = (0, 0, 0, 255)
        # if color[0] > 200: color = (220, 0, 0, 255)

        if (color[0] + color[1] + color[2]) > 230:
            print(word, color, font_size)
        return color

    if gender == 'female':
        icon_path = Path('data', 'plots', 'word_clouds', 'icon_female.png')
    else:
        icon_path = Path('data', 'plots', 'word_clouds', 'icon_male.png')

    icon = Image.open(icon_path)

    basewidth = 3000
    wpercent = (basewidth / float(icon.size[0]))
    hsize = int((float(icon.size[1]) * float(wpercent)))
    icon = icon.resize((basewidth, hsize))#, icon.ANTIALIAS)
    icon = icon.convert('RGBA')

    d = Dataset()
    c1 = d.copy().filter(author_gender='female')
    c2 = d.copy().filter(author_gender='male')
    data = divergence_analysis(d, c1, c2)

    mask = Image.new("RGBA", icon.size, (255, 255, 255))
    mask.paste(icon, icon)
    mask = np.array(mask)


    word_dict = {}
    for _, row in data.iterrows():
        dunning = row['dunning']

        if (gender == 'female' and dunning > 0):
            word_dict[row['term']] = dunning
        if gender == 'male' and dunning < 0:
            word_dict[row['term']] = -dunning

    print("Total tokens: {}".format(len(word_dict)))

    wc = WordCloud(background_color='white', max_font_size=300, mask=mask,
                   max_words=2000, relative_scaling=relative_scaling, min_font_size=4)
    wc.generate_from_frequencies(word_dict)
    wc.recolor(color_func=grey_color_func)


    wc.to_file(Path('data', 'plots', 'word_clouds', f'{gender}_{relative_scaling}.png'))





if __name__ == '__main__':


    d = JournalsDataset()

    # Create two sub-datasets, one for female authors and one for male authors
    c1 = d.copy().filter(author_gender='female')
    c2 = d.copy().filter(author_gender='male')


    # Run the divergence analysis
    div = DivergenceAnalysis(d, c1, c2, sub_corpus1_name='women', sub_corpus2_name='men')
    div.run_divergence_analysis(analysis_type='terms', sort_by='frequency_score')


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


