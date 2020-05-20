from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.utilities import BASE_PATH

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mtick

import pandas as pd



from collections import defaultdict
import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes._subplots import Subplot


def plot_bechdel(term='she', dataset=None):

    if not dataset:
        dataset = JournalsDataset()
        dataset.get_vocabulary_and_document_term_matrix(vocabulary=[term], use_frequencies=True,
                                                    store_in_df=True)
    df = dataset.df
    male_data = []
    female_data = []

    for year in range(dataset.start_year, dataset.end_year + 1):
        print(year)
        male_articles_in_year = df[(df.m_year == year) & (df.m_author_genders == 'male')]
        female_articles_in_year = df[(df.m_year == year) & (df.m_author_genders == 'female')]
        count_male_term = len(male_articles_in_year[male_articles_in_year[term] > 0])
        count_female_term = len(female_articles_in_year[female_articles_in_year[term] > 0])
        male_data.append(count_male_term / len(male_articles_in_year) + 0.0000001)
        female_data.append(count_female_term / len(female_articles_in_year) + 0.0000001)


    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(nrows=1, ncols= 1, figure=fig)
    ax = fig.add_subplot(gs[0,0])

    x = [i for i in range(dataset.start_year + 2, dataset.end_year - 4 )]
    print(x)
    rolling_mean_male = pd.DataFrame(male_data).rolling(center=True, window=5).mean()[0].tolist()[2:-5]
    rolling_mean_female = pd.DataFrame(female_data).rolling(center=True, window=5).mean()[0].tolist()[2:-5]

    ax.plot(x, rolling_mean_male, color='blue')
    ax.plot(x, rolling_mean_female, color='red')

    ax.set(ylim=(0, 1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.title(f'Percentage of articles using the word "{term}" at least once')
    plt.savefig(Path(BASE_PATH, 'visualizations', 'bechdel', f'bechdel_{term}.png'))
    plt.show()

    return rolling_mean_male, rolling_mean_female, x




if __name__ == '__main__':

    dataset = JournalsDataset()
    dataset.get_vocabulary_and_document_term_matrix(max_features=10000, use_frequencies=True,
                                                    store_in_df=True)

    plot_bechdel(term='she', dataset=dataset)
    plot_bechdel(term='women', dataset=dataset)
    plot_bechdel(term='gender', dataset=dataset)
    # plot_bechdel(term='sex', dataset=dataset)
    # plot_bechdel(term='sexuality', dataset=dataset)


