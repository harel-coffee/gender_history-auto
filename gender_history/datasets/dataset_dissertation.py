from gender_history.datasets.dataset import Dataset

from pathlib import Path
import pandas as pd
from IPython import embed

import html
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import numpy as np

from gender_history.utilities import BASE_PATH, WORD_SPLIT_REGEX

from nameparser import HumanName
import re

class DissertationDataset(Dataset):

    def __init__(self):

        try:
            self.df = pd.read_csv(Path(BASE_PATH, 'data', 'dissertations',
                                       'cleaned_history_dissertations_dataset.csv'),
                                  encoding='utf-8')
        except FileNotFoundError:
            self.create_merged_and_cleaned_dataset()
            self.df = pd.read_csv(Path(BASE_PATH, 'data', 'dissertations',
                                       'cleaned_history_dissertations_dataset.csv'),
                                  encoding='utf-8')

        super(DissertationDataset, self).__init__()
        self.dataset_type = 'dissertations'



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

    @staticmethod
    def proquest_name_parser(name):
        """
        Parses those strange proquest names like
        'larocque,_brendan_p.:0'

        :param name:
        :return:
        """

        name = name[:-2].replace('_', ' ')
        name = " ".join([n.capitalize() for n in name.split()])
        hn = HumanName(name)

        name_str = hn.first
        if hn.middle:
            name_str += f' {hn.middle}'
        name_str += f' {hn.last}'

        return name_str


    def create_merged_and_cleaned_dataset(self):


        print("creating and storing a merged, cleaned dataset at "
              "cleaned_history_dissertations_dataset.csv")

        theses = []
        raw_df = pd.read_csv(Path(BASE_PATH, 'data', 'dissertations', 'proquest_raw_dataset.csv'),
                             encoding='windows-1252')
        raw_df['ProQuest.Thesis.ID'] = raw_df['ProQuest.Thesis.ID'].astype('str')
        raw_df['AdvisorID'].fillna('unknown', inplace=True)

        weights_df = pd.read_csv(
            Path(BASE_PATH, 'data', 'dissertations', 'dissertation_topic_weights.csv')
        )
        gender_df = pd.read_csv(
            Path(BASE_PATH, 'data', 'dissertations', 'author_genders_dissertations.csv')
        )
        gender_df['assigned'].fillna(value='unknown', inplace=True)

        name_to_gender = {}
        for _, row in gender_df.iterrows():
            name_to_gender[row['name'].lower()] = row['assigned']

        count_found_in_name_to_gender = 0

        for _, pid in weights_df.ProQid.iteritems():

            pid = str(pid)

            raw_row = raw_df[raw_df['ProQuest.Thesis.ID'] == pid]
            if not len(raw_row) == 1:
                print("not 1 row")
                embed()

            raw_row = raw_row.iloc[0]

            thesis = {'m_pid' : pid}
            thesis['m_year'] = int(raw_row['ThesisYear'])
            thesis['m_descendants'] = int(raw_row['NumDirectDescendants'])

            thesis['m_title'] = raw_row['ThesisTitle']
            thesis['m_keywords'] = raw_row['ThesisKeywords']
            thesis['m_institution'] = raw_row['ThesisInstitution']
            thesis['m_text'] = raw_row['Abstract']
            thesis['m_text_len'] = len(re.findall(WORD_SPLIT_REGEX, thesis['m_text']))

            # Advisee name and gender
            try:
                thesis['m_authors'] = self.proquest_name_parser(raw_row['AdviseeID'])
            except ValueError:
                print('author embed')
                embed()

            assert raw_row['AdviseeGender'] == raw_row['AdviseeGender.1']

            thesis['m_author_genders'] = raw_row['AdviseeGender']
            if thesis['m_authors'].lower() in name_to_gender:
                thesis['m_author_genders'] = name_to_gender[thesis['m_authors'].lower()]
                count_found_in_name_to_gender += 1

            # Advisor name and gender
            if raw_row['AdvisorID'] == 'unknown':
                thesis['m_advisor'] = 'unknown'
            else:
                try:
                    thesis['m_advisor'] = self.proquest_name_parser(raw_row['AdvisorID'])
                except:
                    print("advisor embed")
                    embed()

            assert raw_row['AdvisorGender'] == raw_row['AdvisorGender.1']
            thesis['m_advisor_gender'] = raw_row['AdvisorGender']
            if thesis['m_advisor'].lower() in name_to_gender:
                thesis['m_advisor_gender'] = name_to_gender[thesis['m_advisor'].lower()]

            theses.append(thesis)

            weights_row = weights_df[weights_df['ProQid'] == pid]
            if not len(weights_row) == 1:
                print("weights row not len 1")
                embed()

            weights_row = weights_row.iloc[0]
            for i in range(1, 91):
                thesis[f'topic.{i}'] = weights_row[f'X{i}']

        dissertations_df = pd.DataFrame(theses)
        dissertations_df.to_csv(Path(BASE_PATH, 'data', 'dissertations',
                                     'cleaned_history_dissertations_dataset.csv'),
                                encoding='utf-8')

        return

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
