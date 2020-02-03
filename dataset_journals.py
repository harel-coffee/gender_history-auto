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
from collections import defaultdict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import seaborn as sns

import matplotlib.ticker as mtick
import re
import csv
from collections import Counter, defaultdict

class JournalsDataset(Dataset):

    def __init__(self):

        try:
            self.df = pd.read_csv(Path('data', 'journal_csv', 'general_journals.csv'))
        except FileNotFoundError:
            self.generate_general_journal_dataset()
            self.df = pd.read_csv(Path('data', 'journal_csv', 'general_journals.csv'))

        super(JournalsDataset, self).__init__()

        self.topics = self.load_topic_data(Path('data', 'journal_csv',
                                                'topic_names_jan_2020.csv'))
        self.store_aggregate_approach_and_geographical_info_in_df()

        embed()


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


    def summarize_topic(self, topic_id):
        """
        Creates a summary of a topic including:
        - docs with topic score > 0.2
        - correlated topics
        - terms prob/frex
        - most frequently mentioned centuries

        :param topic_id:
        :return:
        """



        # find the 4 most correlated topics
        topic_selector = [f'X{i}' for i in range(1, 101)]
        topic_df = self.df[topic_selector]
        correlated_topics = []
        for topic, correlation in topic_df.corr()[f'X{topic_id}'].sort_values(
                ascending=False).iteritems():
            if correlation == 1:
                continue
            if len(correlated_topics) > 3:
                break
            tid = int(topic[1:])
            correlated_topics.append({
                'topic_id': tid,
                'correlation': round(correlation, 3)
            })

        # find all docs with topic score > 0.2

        # create a sorted and filtered dataframe
        sorted_df = self.df.sort_values(by=f'X{topic_id}', ascending=False)
        # filter to only keep articles with weight 20% or more
        sorted_df = sorted_df[sorted_df[f'X{topic_id}'] > 0.2]

        docs = []
        # find number of mentions of all centuries
        centuries = {f'{i}xx': 0 for i in range(20, 9, -1)}
        for _, row in sorted_df.iterrows():
            docs.append({
                'topic_weight': round(row[f'X{topic_id}'], 3),
                'year': row.year,
                'author_gender': row.author_genders,
                'title': row.title
            })
            for hit in re.findall('[1-2][0-9][a-zA-Z0-9]{2}', row.text):
                hit = hit[:2] + 'xx'
                if hit in centuries:
                    centuries[hit] += 1

        # Store the results first as a list of lists
        out = [[""] * 10 for _ in range(500)]
        out[0][0] = 'Terms Prob'
        out[0][1] = ", ".join(self.topics[topic_id]['terms_prob'])
        out[1][0] = 'Terms Frex'
        out[1][1] = ", ".join(self.topics[topic_id]['terms_frex'])

        out[3][0] = f'{len(docs)} articles have a topic score > 0.2 for topic {topic_id}.'

        try:
            mean_publication_year = int(np.mean([d["year"] for d in docs]))
        except ValueError:
            mean_publication_year = 'n/a'
            print("no docs for ", topic_id)
        out[4][0] = f'{mean_publication_year}: Mean publication year'

        out[6][0] = 'Correlated Topics'
        out[7][0] = 'Topic ID'
        out[7][1] = 'Name'
        out[7][2] = 'Correlation'
        for idx, cor in enumerate(correlated_topics):
            out[8 + idx][0] = cor['topic_id']
            out[8 + idx][2] = cor['correlation']

        out[13][0] = 'Articles with topic score > 0.2'
        out[14][0] = 'Topic Weight'
        out[14][1] = 'Year'
        out[14][2] = 'Gender'
        out[14][3] = 'Title'
        for idx, doc in enumerate(docs):
            try:
                out[15 + idx][0] = doc['topic_weight']
                out[15 + idx][1] = doc['year']
                out[15 + idx][2] = doc['author_gender']
                out[15 + idx][3] = doc['title']
            except IndexError:
                break

        out[3][5] = 'Centuries Mentioned'
        out[4][5] = 'Century'
        out[4][6] = 'Count'
        out[5][5] = '20xx'
        out[5][6] = centuries['20xx']
        out[6][5] = '19xx'
        out[6][6] = centuries['19xx']
        out[7][5] = '18xx'
        out[7][6] = centuries['18xx']
        out[8][5] = '17xx'
        out[8][6] = centuries['17xx']
        out[9][5] = '16xx'
        out[9][6] = centuries['16xx']
        out[10][5] = '15xx'
        out[10][6] = centuries['15xx']
        out[11][5] = '10xx-14xx'
        out[11][6] = sum([centuries[f'{i}xx'] for i in range(10, 15)])

        # then store as a csv
        with open(Path('data', 'topic_summaries', f'topic_test_{topic_id}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(out)


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

        # topic_names_sorted = [f'X{tid}' for tid in topic_ids]
        topic_names_sorted = [x for x in self.df.columns if x.startswith('gen_approach_')]

        from divergence_analysis import divergence_analysis
        c1 = self.copy().filter(author_gender='male')
        c2 = self.copy().filter(author_gender='female')
        topic_df = divergence_analysis(self, c1, c2, analysis_type='gen_approach',
                            c1_name='male', c2_name='female', sort_by='dunning',
                            number_of_terms_to_print=50)

        data = self.get_data('topics', topic_names_sorted, smoothing=1)

        # median_years = {}
        # for topic_id, topic_name in zip(topic_ids, topic_names_sorted):
        #     try:
        #         tdata = data[topic_name]
        #         topic_sum_so_far = 0
        #         topic_sum_to_reach = sum(tdata['freq_both']) / 2
        #         for i in range(len(tdata['year'])):
        #             topic_sum_so_far += tdata['freq_both'][i]
        #             if topic_sum_so_far >= topic_sum_to_reach:
        #
        #
        #                 median_year = tdata['year'][i]
        #                 median_years[topic_id] = median_year
        #
        #                 # median_years[int(topic_name[1:])] = median_year
        #                 break
        #     except KeyError:
        #         continue


        x_coords = []
        y_coords = []
        color_codes = []

        # for topic_id in topic_ids:
        for topic in topic_names_sorted:
            # row = topic_df[topic_df['index'] == topic_id - 1]
            row = topic_df[topic_df.term == topic]
            y_coords.append(row.frequency_total.values[0])
            x_coords.append(row.frequency_score.values[0])
            # color_codes.append(median_years[topic_id])
            color_codes.append(row.frequency_score.values[0])


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
        ax.set_ylim(0.002, 0.2)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(b=True, which='minor', color='lightgray', linestyle='--')
        ax.tick_params(labelsize=15)

        # ax.yaxis.set_minor_locator(MultipleLocator(5))

        ax.scatter(x_coords, y_coords, c=color_codes, s=300, cmap='coolwarm_r',
                   norm=normalized_cmap)


        # for coords_id, topic_id in enumerate(topic_ids):
        for coords_id, topic in enumerate(topic_names_sorted):
            x = x_coords[coords_id]
            y = y_coords[coords_id]
            topic_name = topic
            # topic_name = f'{topic_id}: {self.topics[topic_id]["name"]}'
            ax.annotate(topic_name, (x_coords[coords_id], y_coords[coords_id]+0.0003))


        plt.savefig(Path('data', 'plots', f'general_approaches_scatterplot.png'), dpi=300,
                    transparent=False)
        plt.show()

if __name__ == '__main__':

    d = JournalsDataset()
    from divergence_analysis import divergence_analysis

    d.plot_topics_to_weight_and_gender_graph()

    d = JournalsDataset()
    for i in range(1, 101):
        d.summarize_topic(i)


    #
    # selected_topic_ids = [
    #     89, 27, 99, 75,
    #     41, 33, 68, 55
    # ]

#    d.plot_topic_grid(smoothing=5)

    # d.plot_topic_grid_of_selected_topics(smoothing=5, selected_topic_ids=selected_topic_ids)

    d.plot_londa(
#        data_type='terms'
        data_type='topics',
        term_or_topic_list=[
            'gen_approach_Political History',
            'gen_approach_Social History',
            # 'gen_approach_Cultural History',
            # 'gen_approach_Economic History',
            'gen_approach_Women’s and Gender History'
        ],
        # terms=['women'],
        smoothing=5
    )

    # d.plot_topics_to_weight_and_gender_graph()

    # d.plot_topic_grid_of_selected_topics(smoothing=0)


    # d.plot_gender_development_over_time(data='terms',
    #     selected_terms_or_topics=['women', 'gender', 'female', 'woman', 'men', 'work', 'feminist'],
    #                                     smoothing=5)
    pass


    # Noise {1, 3, 7, 82, 20, 85} 0.0103611520267198
    # Social History {2, 38, 43, 80, 49, 86, 56, 95} 0.016772475645168
    # Transnational History {77, 90, 4, 5} 0.00856549463992458
    # Economic History {34, 66, 6, 73, 45, 14, 87} 0.010486083097742295
    # Colonial History {36, 8, 42, 79, 17, 84, 25, 26, 59} 0.005613694283982997
    # Historiography {9, 92} 0.026732332490231644
    # Anthropology {10} 0.008234345022682455
    # Religious History {50, 11} 0.009378461220731436
    # Political History {97, 100, 37, 72, 12, 44, 60, 16, 81, 83, 93, 53, 24, 88, 28, 63, 31} 0.010027721118264991
    # Art History {35, 13} 0.00815322530141253
    # Classics {15} 0.007291566952386093
    # History of Education {18} 0.013391852738295653
    # History of Medicine {19, 61, 39} 0.005619493700787617
    # Indigenous History {67, 21} 0.0039057695644533824
    # Legal History {91, 22} 0.015077521181803138
    # History of Slavery {64, 23} 0.006241993858226526
    # Women’s and Gender History {89, 27, 47} 0.011440566774731934
    # History of Race {75, 29} 0.005814663743250853
    # Jewish History {30} 0.005389798226235582
    # Cultural History {32, 99, 69, 51, 52, 62} 0.012771918224215462
    # Military History {33, 78} 0.00807132538908617
    # Intellectual History {41, 68, 70, 55} 0.015824024941590912
    # Islamic History {46} 0.010842411734888804
    # History of Science {48} 0.0058515749713535705
    # Environmental History {65, 58} 0.004386051434203244
    # Medieval History {74} 0.0034007031067451766
