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


from collections import defaultdict
import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes._subplots import Subplot

import pickle
from gender_history.visualizations.ngram_plot import create_ngram_plot, load_master_viz_data


def show_male_female_publications_over_time():
    """
    Quick visualization of number of articles by men and women

    :return:
    """


    d = JournalsDataset()
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

    x = [i for i in range(d.start_year, d.end_year + 1)]

    plt.figure(figsize=(6, 6))
    plt.plot(x, male_arr, color='blue')
    plt.plot(x, female_arr, color='red')

    plt.title('Articles by men (blue) and women (red)')

    plt.savefig(Path(BASE_PATH, 'visualizations', 'dataset_summaries', 'male_female_articles.png'))
    plt.show()



if __name__ == '__main__':
    show_male_female_publications_over_time()
