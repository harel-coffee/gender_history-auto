from gender_history.datasets.dataset import Dataset

from pathlib import Path
import pandas as pd
import numpy as np

from gender_history.utilities import BASE_PATH, WORD_SPLIT_REGEX

from IPython import embed

import re
import csv




class JournalsDataset(Dataset):

    def __init__(self, use_equal_samples_dataset: bool = False):
        """

        :param use_equal_samples_dataset: bool
                    If true, uses self.df_with_equal_samples_per_5_year_period
                    Else self.df
        """

        try:
            self.df = pd.read_csv(Path(BASE_PATH, 'data', 'journal_csv',
                                       'general_journals_dataset.csv'))
        except FileNotFoundError:
            print("No journals dataset found. Generating now...")
            self.generate_general_journal_dataset()
            self.df = pd.read_csv(Path(BASE_PATH, 'data', 'journal_csv',
                                       'general_journals_dataset.csv'))

        super(JournalsDataset, self).__init__()

        if use_equal_samples_dataset:
            self.df = self.create_df_1000_texts_per_5_year_interval()

        self.use_equal_samples_dataset = use_equal_samples_dataset
        self.topics = self.load_topic_data(Path(BASE_PATH, 'data', 'journal_csv',
                                                'topic_titles_and_terms.csv'))
        self.dataset_type = 'journals'
        self.store_aggregate_approach_and_geographical_info_in_df()

        self.name = f'Journals Dataset with {len(self.df)} articles.'

    def generate_general_journal_dataset(self):
        """
        generate a general journal dataset by merging topic weights from
        doc_with_topicvars_stm_GENJ_100full_Sherl.csv
        with the metadata info from
        general_journals_full_texts.csv

        :return:
        """
        from name_to_gender import GenderGuesser
        gender_guesser = GenderGuesser()

        gen_df = pd.read_csv(Path(BASE_PATH, 'data', 'journal_csv',
                                  'doc_with_topiccovars_stm_GENJ_90_final_WITH_ID.csv'))

        metadata_df = pd.read_csv(Path(BASE_PATH, 'data', 'journal_csv',
                                'general_journals_full_texts_raw.csv'))
        metadata_df['pages'] = metadata_df['pages'].astype(str)

        # put id_doi_jstor column into the same format as the results csv
        metadata_df['id_doi_jstor'] = metadata_df['ID_jstor'].fillna(0).astype(int).astype(str)
        metadata_df['id_doi_jstor'] = np.where(metadata_df['id_doi_jstor'] == '0',
                                               metadata_df['ID_doi'] + 'NA',
                                               metadata_df['id_doi_jstor'])

        gen_df['m_ID_doi'] = ''
        gen_df['m_ID_jstor'] = ''
        gen_df['m_article_type'] = ''
        gen_df['m_pages'] = ''
        gen_df['m_title'] = ''
        gen_df['m_language'] = ''
        gen_df['m_year'] = ''
        gen_df['m_volume'] = ''
        gen_df['m_issue'] = ''
        gen_df['m_journal'] = ''
        gen_df['m_authors'] = ''
        gen_df['m_text_len'] = 0

        for idx, row in gen_df.iterrows():

            # find metadata row matching the DOI.JSTOR.ID
            metadata_row_id = np.where(metadata_df['id_doi_jstor'] == row['DOI.JSTOR.ID'])[0]
            if len(metadata_row_id) != 1:
                raise ValueError(f"Exactly one row should match the id_doi_jstor "
                                 f"{row['DOI.JSTOR.ID']}")
            metadata_row = metadata_df.iloc[metadata_row_id[0]]

            try:
                gen_df.at[idx, 'm_article_type'] = metadata_row.article_type
                gen_df.at[idx, 'm_pages'] = metadata_row.pages
                gen_df.at[idx, 'm_title'] = metadata_row.title
                gen_df.at[idx, 'm_language'] = metadata_row.language
                gen_df.at[idx, 'm_year'] = metadata_row.year
                gen_df.at[idx, 'm_volume'] = metadata_row.volume
                gen_df.at[idx, 'm_issue'] = metadata_row.issue
                gen_df.at[idx, 'm_journal'] = metadata_row.journal
                gen_df.at[idx, 'm_authors'] = metadata_row.authors

                # get author gender from google sheet data
                author_genders = gender_guesser.get_gender_of_journal_authors(metadata_row.authors)
                if author_genders == 'undetermined':
                    print(metadata_row.authors, metadata_row.author_genders, author_genders)

                gen_df.at[idx, 'm_author_genders'] = author_genders
                gen_df.at[idx, 'm_text'] = metadata_row.text
                gen_df.at[idx, 'm_text_len'] = len(re.findall(WORD_SPLIT_REGEX, metadata_row.text))

            except IndexError:
                continue

        # parse numbers to int
        gen_df['m_year'] = gen_df['m_year'].astype(int)
        gen_df.to_csv(Path(BASE_PATH, 'data', 'journal_csv', 'general_journals_dataset.csv'))


    def summarize_topic(self, topic_id):
        """
        Creates a summary of a topic including:
        - docs with topic score > 0.2
        - correlated topics
        - terms prob/frex
        - most frequently mentioned centuries

        Used to create google sheet to label topics.

        :param topic_id:
        :return:
        """

        # find the 4 most correlated topics
        topic_selector = [f'topic.{i}' for i in range(1, 91)]
        topic_df = self.df[topic_selector]
        correlated_topics = []
        for topic, correlation in topic_df.corr()[f'topic.{topic_id}'].sort_values(
                ascending=False).iteritems():
            if correlation == 1:
                continue
            if len(correlated_topics) > 3:
                break
            tid = int(topic[6:])
            correlated_topics.append({
                'topic_id': tid,
                'correlation': round(correlation, 3)
            })

        # find all docs with topic score > 0.2

        # create a sorted and filtered dataframe
        sorted_df = self.df.sort_values(by=f'topic.{topic_id}', ascending=False)
        # filter to only keep articles with weight 20% or more
        sorted_df = sorted_df[sorted_df[f'topic.{topic_id}'] > 0.2]

        docs = []
        # find number of mentions of all centuries
        centuries = {f'{i}xx': 0 for i in range(20, 9, -1)}
        for _, row in sorted_df.iterrows():
            docs.append({
                'topic_weight': round(row[f'topic.{topic_id}'], 3),
                'year': row.m_year,
                'author_gender': row.m_author_genders,
                'title': row.m_title
            })

            if isinstance(row.m_text, str):
                for hit in re.findall('[1-2][0-9][a-zA-Z0-9]{2}', row.m_text):
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


    # def plot_topics_to_weight_and_gender_graph(self):
    #
    #     topic_ids = [
    #         # 1,  # Noise (volume / year)
    #         2,  # Dutch History
    #         # 3,  # Noise (publication)
    #         4,  # Migration
    #         5,  # World History
    #         6,  # Political History
    #         # 7,  # Noise (?letters to editor?)
    #         8,  # U.S. Colonial History
    #         # 9,  # ??Noise?? (historiography)
    #         10,  # ??social groups / anthropology ??
    #         11,  # ?? Religious History??
    #         12,  # French History
    #         13,  # Art History
    #         14,  # Economic History
    #         15,  # Ancient History
    #         16,  # ?? American History ??
    #         17,  # ?? Japanese / East Asian History ??
    #         18,  # Education
    #         19,  # Home
    #         20,  # Noise (dates)
    #         21,  # Native American History
    #         22,  # ?? Criminal History
    #         23,  # History of Slavery
    #         24,  # ?? Noise (German Language)
    #         25,  # ?? U.S. Colonial / New England ??
    #         26,  # Indian History
    #         27,  # History of Food and Consumption
    #         28,  # British History
    #         29,  # Race
    #         30,  # Jewish History
    #         31,  # Nation and Nationalism
    #         # 32,  # ?? Noise
    #         33,  # War
    #         34,  # ?? Trade (Asia)
    #         35,  # History of Music
    #         36,  # Colonial History
    #         37,  # History of Organizations
    #         38,  # Quantitative Social History
    #         39,  # History of Medicine (disease)
    #         40,  # ?? Noise (french language)
    #         41,  # ?? Annales
    #         42,  # ?? Spanish Colonial History
    #         43,  # Social History
    #         44,  # Political History (political parties)
    #         45,  # Labor
    #         46,  # Islam
    #         47,  # Family
    #         48,  # Science
    #         49,  # ?? Local History
    #         50,  # Religious History (Christianity)
    #         51,  # Italian Renaissance
    #         52,  # ?? Witchcraft and Freud
    #         53,  # ??
    #         54,  # Indian History (2)
    #         55,  # Economic History (Marxist)
    #         56,  # Urban
    #         57,  # Holocaust
    #         58,  # Environment
    #         59,  # Africa
    #         60,  # Soviet
    #         61,  # Medicine (phyisician)
    #         62,  # ?? Class / sport
    #         63,  # German History
    #         64,  # Slave Trade
    #         65,  # Latin America
    #         66,  # Population Statistics
    #         67,  # Native American History
    #         68,  # Intellectual History
    #         69,  # Childern / Childhood
    #         70,  # Political Theory
    #         71,  # French Revolution
    #         72,  # ?? Italy / Fascism
    #         73,  # ?? Oil and Cars
    #         74,  # ?? Feudalism
    #         75,  # Race (2) //unclear how it differs from topic 29
    #         76,  # Spain
    #         77,  # North America
    #         78,  # World War I
    #         79,  # Brazil
    #         80,  # Russia
    #         81,  # Governance
    #         82,  # ??
    #         83,  # ?? U.S. Politicall History
    #         84,  # ?? Latin America (Colonial)
    #         85,  # ??
    #         86,  # Agriculture
    #         # 87,  # ?? War and Economy
    #         88,  # ?? France (Early Modern)
    #         89,  # Women's History
    #         90,  # Mexico
    #         91,  # Legal History
    #         92,  # ??
    #         93,  # ?? Colonies
    #         94,  # China
    #         95,  # ?? Law Enforcement
    #         96,  # U.S. Civil War
    #         97,  # Germany (20th Century)
    #         98,  # Oceania
    #         99,  # Sexuality
    #     ]
    #
    #     # topic_names_sorted = [f'X{tid}' for tid in topic_ids]
    #     topic_names_sorted = [x for x in self.df.columns if x.startswith('gen_approach_')]
    #
    #     from divergence_analysis import divergence_analysis
    #     c1 = self.copy().filter(author_gender='male')
    #     c2 = self.copy().filter(author_gender='female')
    #     topic_df = divergence_analysis(self, c1, c2, analysis_type='gen_approach',
    #                         c1_name='male', c2_name='female', sort_by='dunning',
    #                         number_of_terms_to_print=50)
    #
    #     data = self.get_data('topics', topic_names_sorted, smoothing=1)
    #
    #     # median_years = {}
    #     # for topic_id, topic_name in zip(topic_ids, topic_names_sorted):
    #     #     try:
    #     #         tdata = data[topic_name]
    #     #         topic_sum_so_far = 0
    #     #         topic_sum_to_reach = sum(tdata['freq_both']) / 2
    #     #         for i in range(len(tdata['year'])):
    #     #             topic_sum_so_far += tdata['freq_both'][i]
    #     #             if topic_sum_so_far >= topic_sum_to_reach:
    #     #
    #     #
    #     #                 median_year = tdata['year'][i]
    #     #                 median_years[topic_id] = median_year
    #     #
    #     #                 # median_years[int(topic_name[1:])] = median_year
    #     #                 break
    #     #     except KeyError:
    #     #         continue
    #
    #
    #     x_coords = []
    #     y_coords = []
    #     color_codes = []
    #
    #     # for topic_id in topic_ids:
    #     for topic in topic_names_sorted:
    #         # row = topic_df[topic_df['index'] == topic_id - 1]
    #         row = topic_df[topic_df.term == topic]
    #         y_coords.append(row.frequency_total.values[0])
    #         x_coords.append(row.frequency_score.values[0])
    #         # color_codes.append(median_years[topic_id])
    #         color_codes.append(row.frequency_score.values[0])
    #
    #
    #     fig = plt.figure(figsize=(12, 12))
    #     gs = gridspec.GridSpec(nrows=1,
    #                            ncols=1,
    #                            figure=fig,
    #                            width_ratios=[1],
    #                            height_ratios=[1],
    #                            wspace=0.2, hspace=0.05
    #                            )
    #
    #     ax = fig.add_subplot(gs[0, 0])
    #     ax.set_ylim(0, 1)
    #     ax.set_xlim(self.start_year + 2, self.end_year - 2)
    #     ax.set_axisbelow(True)
    #     ax.grid(which='major', axis='both')
    #
    #     norm = plt.Normalize(0.2, 0.8)
    #     colors = cm
    #     # Set the values used for colormapping
    #     cmap = cm.get_cmap('coolwarm')
    #     normalized_cmap = matplotlib.colors.Normalize(vmin=min(color_codes),
    #                                                   vmax=max(color_codes))
    #
    #     # ax.scatter(x_coords, y_coords, s=300)
    #     ax.set_xlim(0, 1)
    #
    #
    #     # y axis
    #     # ax.set_ylabel('Topic Weight')
    #     # ax.label_params(labelsize=20)
    #     ax.set_ylim(0.002, 0.2)
    #     ax.set_yscale('log')
    #     ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    #     ax.grid(b=True, which='minor', color='lightgray', linestyle='--')
    #     ax.tick_params(labelsize=15)
    #
    #     # ax.yaxis.set_minor_locator(MultipleLocator(5))
    #
    #     ax.scatter(x_coords, y_coords, c=color_codes, s=300, cmap='coolwarm_r',
    #                norm=normalized_cmap)
    #
    #
    #     # for coords_id, topic_id in enumerate(topic_ids):
    #     for coords_id, topic in enumerate(topic_names_sorted):
    #         x = x_coords[coords_id]
    #         y = y_coords[coords_id]
    #         topic_name = topic
    #         # topic_name = f'{topic_id}: {self.topics[topic_id]["name"]}'
    #         ax.annotate(topic_name, (x_coords[coords_id], y_coords[coords_id]+0.0003))
    #
    #
    #     plt.savefig(Path('data', 'plots', f'general_approaches_scatterplot.png'), dpi=300,
    #                 transparent=False)
    #     plt.show()

if __name__ == '__main__':

    d = JournalsDataset()
#    x = d.df_with_equal_samples_per_5_year_period
    embed()

