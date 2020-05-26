

from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.utilities import BASE_PATH

from pathlib import Path

import pandas as pd

from IPython import embed

from collections import defaultdict


def get_individual_topic_viz_data(terms, topic_name, d: JournalsDataset=None):

    if not d:
        d = JournalsDataset()
    d.get_vocabulary_and_document_term_matrix(vocabulary=terms, use_frequencies=True,
                                              store_in_df=True)
    men = d.copy().filter(author_gender='male')
    women = d.copy().filter(author_gender='female')

    data = defaultdict(list)

    for year in range(d.start_year, d.end_year + 1):


        men_year = men.copy().filter(start_year=year, end_year=year)
        women_year = women.copy().filter(start_year=year, end_year=year)
        all_year = d.copy().filter(start_year=year, end_year=year)


        print(year, len(all_year))

        for term in terms:
            data[f'men_{term}'].append(men_year.df[term].mean())
            data[f'women_{term}'].append(women_year.df[term].mean())
            data[f'all_{term}'].append(all_year.df[term].mean())

    avg_data = {}
    for term in terms:
        # avg_data[f'men_{term}'] = pd.DataFrame(data[f'men_{term}']).rolling(center=True,
        #                                                         window=5).mean()[0].tolist()[2:-5]
        # avg_data[f'women_{term}'] = pd.DataFrame(data[f'women_{term}']).rolling(center=True,
        #                                                         window=5).mean()[0].tolist()[2:-5]
        avg_data[f'all_{term}'] = pd.DataFrame(data[f'all_{term}']).rolling(center=True,
                                                                window=5).mean()[0].tolist()[2:-5]

    avg_data['year'] = [int(i) for i in range(d.start_year, d.end_year + 1)][2:-5]


    df = pd.DataFrame.from_dict(avg_data, orient='index').transpose()
    df['year'] = df['year'].astype(int)

    df.to_csv(Path(BASE_PATH, 'visualizations', 'plotly_data', f'{topic_name}_topic_data.csv'))


def get_all_viz_data_for_sexuality():

    get_individual_topic_viz_data(topic_name='sexuality',
                                  terms=['sex', 'sexuality', 'freud', 'emotions', 'love',
                                         'gay', 'masculinity'])

def get_viz_data_for_gender_broadening():

    d = JournalsDataset()
    d.topic_score_filter(topic_id=61, min_percentile_score=90)
    get_individual_topic_viz_data(d=d, topic_name='gender_top_decile',
                                  terms=['work', 'family', 'percent', 'medical', 'age', 'table',
                                         'married',
                                         'gender', 'white', 'black', 'race',
                                         'war', 'nation', 'colonial',
                                         'african', 'british'
                                         ])
    # get_individual_topic_viz_data(d=d, topic_name='gender_top_decile_late',
    #                               terms=['gender', 'white', 'african', 'race', 'black',
    #                                      'war', 'racial', 'nation', 'colonial'])



if __name__ == '__main__':
    get_viz_data_for_gender_broadening()