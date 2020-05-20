from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.utilities import BASE_PATH

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors

from collections import Counter

import pandas as pd


from collections import defaultdict
import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes._subplots import Subplot

import pickle
from gender_history.visualizations.ngram_plot import create_ngram_plot, load_master_viz_data


def show_male_female_publications_over_time(dataset='journals'):
    """
    Quick visualization of number of articles by men and women

    :return:
    """

    if dataset == 'journals':
        d = JournalsDataset()
    else:
        d = DissertationDataset()
        d.filter(start_year=1980)
    male_counter = Counter()
    female_counter = Counter()

    for _, row in d.df.iterrows():

        if row.m_author_genders == 'male':
            male_counter[row.m_year] += 1
        if row.m_author_genders == 'female':
            female_counter[row.m_year] += 1

    male_arr = []
    female_arr = []
    for year in range(d.start_year, d.end_year + 1):
        male_arr.append(male_counter[year])
        female_arr.append(female_counter[year])

    rolling_female = np.array(pd.DataFrame(female_arr).rolling(center=True, window=5).mean()[0].tolist()[2:-5])
    rolling_male = np.array(pd.DataFrame(male_arr).rolling(center=True, window=5).mean()[0].tolist()[2:-5])

    x = [i for i in range(d.start_year, d.end_year + 1)][2:-5]

    plt.figure(figsize=(6, 6))
    plt.plot(x, rolling_female / (rolling_female + rolling_male), color='blue')
    # plt.plot(x, rolling_male, color='red')

    plt.title('Articles by men (blue) and women (red)')

    plt.savefig(Path(BASE_PATH, 'visualizations', 'dataset_summaries', 'male_female_articles.png'))
    plt.show()

    return rolling_male, rolling_female, x


def get_data_and_store_as_csv():


    j_male, j_female, j_x = show_male_female_publications_over_time(dataset='journals')
    d_male, d_female, d_x = show_male_female_publications_over_time(dataset='dissertations')

    data = defaultdict(dict)
    for idx, year in enumerate(j_x):
        data[year]['journal_male'] = j_male[idx]
        data[year]['journal_female'] = j_female[idx]
        data[year]['journal_ratio'] = j_female[idx] / (j_male[idx] + j_female[idx])

    for idx, year in enumerate(d_x):
        data[year]['diss_male'] = d_male[idx]
        data[year]['diss_female'] = d_female[idx]
        data[year]['diss_ratio'] = d_female[idx] / (d_male[idx] + d_female[idx])

    df = pd.DataFrame(data).transpose()
    df.to_csv(Path(BASE_PATH, 'visualizations', 'gender_ratios.csv'))
    embed()

if __name__ == '__main__':
    get_data_and_store_as_csv()