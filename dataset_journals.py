from dataset import Dataset

from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']

from pathlib import Path
import pandas as pd
import numpy as np

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import seaborn as sns

import matplotlib.ticker as mtick

class JournalsDataset(Dataset):

    def __init__(self):

        try:
            self.df = pd.read_csv(Path('data', 'journal_csv', 'general_journals.csv'))
        except FileNotFoundError:
            self.generate_general_journal_dataset()
            self.df = pd.read_csv(Path('data', 'journal_csv', 'general_journals.csv'))

        super(JournalsDataset, self).__init__()

        self.topics = self.load_topic_data(Path('data', 'journal_csv',
                                                'topic_words_stm_GENJ_100.csv'))


    def generate_general_journal_dataset(self):
        """
        generate a general journal dataset by merging topic weights from
        doc_with_topicvars_stm_GENJ_100full_Sherl.csv
        with the metadata info from
        general_journals_full_texts.csv

        :return:
        """

        gen_df = pd.read_csv(Path('data', 'journal_csv',
                                  'doc_with_topiccovars_stm_GENJ_100_full_Sherl.csv'))

        metadata_df = pd.read_csv(Path('data', 'journal_csv',
                                'general_journals_full_texts.csv'))
        metadata_df['pages'] = metadata_df['pages'].astype(str)

        #
        # gen_df['ID_doi'] = ''
        # gen_df['ID_jstor'] = ''
        gen_df['article_type'] = ''
        gen_df['pages'] = ''
        gen_df['title'] = ''
        gen_df['language'] = ''
        gen_df['year'] = ''
        gen_df['volume'] = ''
        gen_df['issue'] = ''
        gen_df['journal'] = ''
        gen_df['authors'] = ''

        for idx, row in gen_df.iterrows():
            print(idx)


            metadata_row = metadata_df.loc[metadata_df['ID_jstor'] == row['ID_jstor']]

            try:
                gen_df.at[idx, 'article_type'] = metadata_row.article_type.values[0]
                gen_df.at[idx, 'pages'] = metadata_row.pages.values[0]
                gen_df.at[idx, 'title'] = metadata_row.title.values[0]
                gen_df.at[idx, 'language'] = metadata_row.language.values[0]
                gen_df.at[idx, 'year'] = metadata_row.year.values[0]
                gen_df.at[idx, 'volume'] = metadata_row.volume.values[0]
                gen_df.at[idx, 'issue'] = metadata_row.issue.values[0]
                gen_df.at[idx, 'journal'] = metadata_row.journal.values[0]
                gen_df.at[idx, 'authors'] = metadata_row.authors.values[0]
            except IndexError:
                continue

        # delete rows with nan values
        # gen_df = gen_df[np.isnan(gen_df.year)]
        gen_df = gen_df[np.isnan(gen_df.ID_jstor) == False]

        # parse numbers to int
        gen_df.year = gen_df.year.astype(int)

        gen_df.to_csv(Path('data', 'journal_csv', 'general_journals.csv'))




    def plot_topics_to_weight_and_gender_graph(self):

        topic_ids = [
            # 1,  # Noise (volume / year)
            2,  # Dutch History
            # 3,  # Noise (publication)
            4,  # Migration
            5,  # World History
            6,  # Political History
            # 7,  # Noise (?letters to editor?)
            8,  # U.S. Colonial History
            # 9,  # ??Noise?? (historiography)
            10,  # ??social groups / anthropology ??
            11,  # ?? Religious History??
            12,  # French History
            13,  # Art History
            14,  # Economic History
            15,  # Ancient History
            16,  # ?? American History ??
            17,  # ?? Japanese / East Asian History ??
            18,  # Education
            19,  # Home
            20,  # Noise (dates)
            21,  # Native American History
            22,  # ?? Criminal History
            23,  # History of Slavery
            24,  # ?? Noise (German Language)
            25,  # ?? U.S. Colonial / New England ??
            26,  # Indian History
            27,  # History of Food and Consumption
            28,  # British History
            29,  # Race
            30,  # Jewish History
            31,  # Nation and Nationalism
            # 32,  # ?? Noise
            33,  # War
            34,  # ?? Trade (Asia)
            35,  # History of Music
            36,  # Colonial History
            37,  # History of Organizations
            38,  # Quantitative Social History
            39,  # History of Medicine (disease)
            40,  # ?? Noise (french language)
            41,  # ?? Annales
            42,  # ?? Spanish Colonial History
            43,  # Social History
            44,  # Political History (political parties)
            45,  # Labor
            46,  # Islam
            47,  # Family
            48,  # Science
            49,  # ?? Local History
            50,  # Religious History (Christianity)
            51,  # Italian Renaissance
            52,  # ?? Witchcraft and Freud
            53,  # ??
            54,  # Indian History (2)
            55,  # Economic History (Marxist)
            56,  # Urban
            57,  # Holocaust
            58,  # Environment
            59,  # Africa
            60,  # Soviet
            61,  # Medicine (phyisician)
            62,  # ?? Class / sport
            63,  # German History
            64,  # Slave Trade
            65,  # Latin America
            66,  # Population Statistics
            67,  # Native American History
            68,  # Intellectual History
            69,  # Childern / Childhood
            70,  # Political Theory
            71,  # French Revolution
            72,  # ?? Italy / Fascism
            73,  # ?? Oil and Cars
            74,  # ?? Feudalism
            75,  # Race (2) //unclear how it differs from topic 29
            76,  # Spain
            77,  # North America
            78,  # World War I
            79,  # Brazil
            80,  # Russia
            81,  # Governance
            82,  # ??
            83,  # ?? U.S. Politicall History
            84,  # ?? Latin America (Colonial)
            85,  # ??
            86,  # Agriculture
            # 87,  # ?? War and Economy
            88,  # ?? France (Early Modern)
            89,  # Women's History
            90,  # Mexico
            91,  # Legal History
            92,  # ??
            93,  # ?? Colonies
            94,  # China
            95,  # ?? Law Enforcement
            96,  # U.S. Civil War
            97,  # Germany (20th Century)
            98,  # Oceania
            99,  # Sexuality
        ]

        topic_names_sorted = [f'X{tid}' for tid in topic_ids]

        from divergence_analysis import divergence_analysis
        c1 = self.copy().filter(author_gender='male')
        c2 = self.copy().filter(author_gender='female')
        topic_df = divergence_analysis(self, c1, c2, topics_or_terms='topics',
                            c1_name='male', c2_name='female', sort_by='dunning',
                            number_of_terms_to_print=50)

        data = self.get_data('topics', topic_names_sorted, smoothing=1)

        median_years = {}
        for topic_id, topic_name in zip(topic_ids, topic_names_sorted):
            try:
                tdata = data[topic_name]
                topic_sum_so_far = 0
                topic_sum_to_reach = sum(tdata['freq_both']) / 2
                for i in range(len(tdata['year'])):
                    topic_sum_so_far += tdata['freq_both'][i]
                    if topic_sum_so_far >= topic_sum_to_reach:


                        median_year = tdata['year'][i]
                        median_years[topic_id] = median_year

                        # median_years[int(topic_name[1:])] = median_year
                        break
            except KeyError:
                continue


        x_coords = []
        y_coords = []
        color_codes = []
        for topic_id in topic_ids:
            row = topic_df[topic_df['index'] == topic_id - 1]
            y_coords.append(row.frequency_total.values[0])
            x_coords.append(row.frequency_score.values[0])
            color_codes.append(median_years[topic_id])

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

        norm = plt.Normalize(0.2, 0.8)
        colors = cm
        # Set the values used for colormapping
        cmap = cm.get_cmap('coolwarm')
        normalized_cmap = matplotlib.colors.Normalize(vmin=min(color_codes),
                                                      vmax=max(color_codes))

        # ax.scatter(x_coords, y_coords, s=300)
        ax.set_xlim(0, 1)


        # y axis
        # ax.set_ylabel('Topic Weight')
        # ax.label_params(labelsize=20)
        ax.set_ylim(0.002, 0.1)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(b=True, which='minor', color='lightgray', linestyle='--')
        ax.tick_params(labelsize=15)

        # ax.yaxis.set_minor_locator(MultipleLocator(5))

        ax.scatter(x_coords, y_coords, c=color_codes, s=300, cmap='coolwarm_r',
                   norm=normalized_cmap)


        for coords_id, topic_id in enumerate(topic_ids):
            x = x_coords[coords_id]
            y = y_coords[coords_id]
            topic_name = f'{topic_id}: {self.topics[topic_id]["name"]}'
            # ax.annotate(topic_name, (x_coords[coords_id], y_coords[coords_id]+0.0003))


        plt.savefig(Path('data', 'plots', f'gender_topicweight_scatterplot_nolabels.png'), dpi=300,
                    transparent=True)
        plt.show()
        embed()

if __name__ == '__main__':



    d = JournalsDataset()

    selected_topic_ids = [
        89, 27, 99, 75,
        41, 33, 68, 55
    ]

#    d.plot_topic_grid(smoothing=5)

    # d.plot_topic_grid_of_selected_topics(smoothing=5, selected_topic_ids=selected_topic_ids)

#     d.plot_londa(
# #        data_type='terms'
#         data_type='topics',
#         term_or_topic_list=['X89', 'X99', 'X31', 'X68'],
#         # terms=['women'],
#         smoothing=5
#     )

    # d.plot_topics_to_weight_and_gender_graph()

    d.plot_topic_grid_of_selected_topics([i for i in range(1, 101)])


    # d.plot_gender_development_over_time(data='terms',
    #     selected_terms_or_topics=['women', 'gender', 'female', 'woman', 'men', 'work', 'feminist'],
    #                                     smoothing=5)
    pass
