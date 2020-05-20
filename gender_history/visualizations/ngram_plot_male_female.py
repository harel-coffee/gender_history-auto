


from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.datasets.dataset import Dataset
from gender_history.utilities import BASE_PATH

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors

import csv

import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd



def ngram_plot(token):

    d = JournalsDataset()



    male = d.copy().filter(author_gender='male')
    female = d.copy().filter(author_gender='female')

    if not token.startswith('topic.'):
        male.get_vocabulary_and_document_term_matrix(vocabulary=[token], store_in_df=True,
                                                 use_frequencies=True)
        female.get_vocabulary_and_document_term_matrix(vocabulary=[token], store_in_df=True,
                                                 use_frequencies=True)

    male_data = []
    female_data = []

    for year in range(d.start_year, d.end_year + 1):
        male_data.append(male.df[male.df.m_year == year][token].mean())
        female_data.append(female.df[female.df.m_year == year][token].mean())


    rolling_male = pd.DataFrame(male_data).rolling(center=True, window=7).mean()[0].tolist()[2:-5]
    rolling_female = pd.DataFrame(female_data).rolling(center=True, window=7).mean()[0].tolist()[2:-5]
    x = [i for i in range(d.start_year, d.end_year + 1)][2:-5]

    plt.figure(figsize=(6, 6))
    plt.plot(x, rolling_male, color='blue')
    plt.plot(x, rolling_female, color='red')

    plt.title(f'{token} in articles by men (blue) and women (red)')

    # plt.savefig(Path(BASE_PATH, 'visualizations', 'dataset_summaries', 'male_female_articles.png'))
    plt.show()

    return rolling_male, rolling_female, x

def get_data():

    d = JournalsDataset()
    d.get_vocabulary_and_document_term_matrix(vocabulary=['women', 'gender'], use_frequencies=True,
                                              store_in_df=True)


    from gender_history.visualizations.bechdel_plot import plot_bechdel

    b_women_male, b_women_female, x1 = plot_bechdel(term='women', dataset=d)
    b_gender_male, b_gender_female, _ = plot_bechdel(term='gender', dataset=d)


    f_gender_male, f_gender_female, x = ngram_plot('gender')

    f_women_male, f_women_female, _ = ngram_plot('women')
    f_topic_male, f_topic_female, _ = ngram_plot('topic.61')


    with open(Path(BASE_PATH, 'visualizations', 'gender_women.csv'), 'w') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['topic_male', 'topic_female', 'women_male', 'women_female',
                             'gender_male', 'gender_female',
                             'bechdel_women_male', 'bechdel_women_female',
                             'bechdel_gender_male', 'bechdel_gender_female',
                             'x',
                             ])
        for i in range(len(f_gender_male)):
            csv_writer.writerow([
                f_topic_male[i], f_topic_female[i], f_women_male[i], f_women_female[i],
                f_gender_male[i], f_gender_female[i],
                b_women_male[i], b_women_female[i], b_gender_male[i], b_gender_female[i],
                x[i]
            ])

if __name__ == '__main__':
    # ngram_plot('topic.61')
    # ngram_plot('women')
    # ngram_plot('gender')
    get_data()