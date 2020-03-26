from gender_history.datasets.dataset import Dataset

from pathlib import Path
import pandas as pd
from IPython import embed

import html
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import numpy as np

class DissertationDataset(Dataset):

    def __init__(self):


        try:
            self.df = pd.read_csv(Path('data', 'dissertations',
                                       'cleaned_history_dissertations_dataset.csv'),
                                  encoding='utf-8')
        except FileNotFoundError:
            self.create_merged_and_cleaned_dataset()
            self.df = pd.read_csv(Path('data', 'dissertations',
                                       'cleaned_history_dissertations_dataset.csv'),
                                  encoding='utf-8')

        self.topics = self.load_topic_data(Path('data', 'dissertations', 'topic_terms.csv'))


        super(Dataset, self).__init__()


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


    def create_merged_and_cleaned_dataset(self):

        self.df = pd.read_csv(Path('data', 'dissertations', 'doc_with_outcome_and_abstract_stm.csv'),
                              encoding='windows-1252')

        print("creating and storing a merged, cleaned dataset at "
              "cleaned_history_dissertations_dataset.csv")

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
        fields_df = pd.read_csv(Path('data', 'dissertations', 'all_data.csv'), encoding='ISO-8859-1')

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

        embed()

        print("storing non-history dataset at non_history_dissertations_in_dataset")
        non_history_df = self.df[self.df['is_history'] == False]
        non_history_df.reset_index(inplace=True)
        non_history_df.to_csv(Path('data', 'dissertations',
                                           'non_history_dissertations_in_dataset.csv'),
                              encoding='utf8')

        print("Eliminating non-history dissertations from the dataset")

        self.df = self.df[self.df['is_history'] == True]
        self.df.reset_index(inplace=True)

        self.df.to_csv(Path('data', 'dissertations',
                                    'cleaned_history_dissertations_dataset.csv'), encoding='utf8')


    # def gender(self):
    #
    #     from nameparser import HumanName
    #     from name_to_gender import GenderGuesser
    #     gg = GenderGuesser()
    #     count = 0
    #
    #     ambiguous_names = []
    #     for _, row in self.df.iterrows():
    #
    #
    #         name = row['AdviseeID'][:-2].replace('_', ' ')
    #         name = " ".join([n.capitalize() for n in name.split()])
    #         hn = HumanName(name)
    #         if row.AdviseeID == 'afroz,_sultana:0':
    #             embed()
    #
    #         guessed_gender_gg = gg.guess_gender_of_human_name(hn, mode='gender_guesser')
    #         guessed_gender_census = gg.guess_gender_of_human_name(hn, mode='census')
    #
    #         if (guessed_gender_census != guessed_gender_gg or
    #             guessed_gender_census != row.AdviseeGender):
    #
    #             if (row.AdviseeGender == 'male' and guessed_gender_census in {'male', 'mostly_male'}
    #                     and guessed_gender_gg in {'male', 'mostly_male'}):
    #                 continue
    #             if (row.AdviseeGender == 'female' and
    #                     guessed_gender_census in {'female', 'mostly_female'} and
    #                     guessed_gender_gg in {'female', 'mostly_female'}):
    #                 continue
    #
    #             name_str = hn.first
    #             if hn.middle:
    #                 name_str += f' {hn.middle}'
    #             name_str += f' {hn.last}'
    #
    #             ambiguous_names.append({
    #                 'name': name_str,
    #                 'original': row.AdviseeGender,
    #                 'gender_guesser': guessed_gender_gg,
    #                 'census': guessed_gender_census
    #             })
    #
    #             print(name, guessed_gender_gg, guessed_gender_census, row.AdviseeGender)
    #             count += 1
    #     df = pd.DataFrame(ambiguous_names)
    #     df.to_csv(Path('data', 'gender_inference', 'ambiguous_dissertation_names.csv'))
    #     print(count)

if __name__ == '__main__':
    d = DissertationDataset()

    d.gender()
    embed()