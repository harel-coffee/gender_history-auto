import numpy as np
import pandas as pd
import seaborn as sns
import math

from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.datasets.dataset import Dataset
from gender_history.utilities import BASE_PATH

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mtick



from collections import defaultdict
import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes._subplots import Subplot


def topic_percentile_plot(
        dataset: Dataset,

        selection_column: str,
        selection_name: str,
        show_plot: bool=True,
        filename: str=None
):
    """


    NOTE: for topics with many undetermined authors, e.g. topic 16, there will only be about 5
    articles overall in the top 1 percentile because most are unknown.

    :param dataset:
    :param selection_column:
    :param selection_name:
    :param show_plot:
    :param filename:
    :return:
    """

    percentile_ranges_to_display =[
        (0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70),
         (70, 80), (80, 90), (90, 100), (95, 100), (99, 100)
    ]

    df = dataset.df

    # Generate decile data
    df['m_percentile'] = df[selection_column].rank(pct=True) * 100
    df['m_percentile'] = df['m_percentile'].astype(int)
    # replace 100th percentile with 90th
    df['m_percentile'].replace(to_replace=100, value=99, inplace=True)

    # get percentage of male/female articles in each decile
    male_docs = df[df.m_author_genders == 'male']
    female_docs = df[df.m_author_genders == 'female']
    percentile_freqs_male = male_docs.m_percentile.value_counts(normalize=True)\
        .reset_index().sort_values(by='index')
    percentile_freqs_female = female_docs.m_percentile.value_counts(normalize=True)\
        .reset_index().sort_values(by='index')

    freq_male = []
    freq_female = []
    for start, end in percentile_ranges_to_display:
        freq_male.append(percentile_freqs_male[start: end].m_percentile.sum() * 100)
        freq_female.append(percentile_freqs_female[start: end].m_percentile.sum() * 100)

    # freq_male = ((male_docs.m_percentile.value_counts(normalize=True).reset_index().sort_values(by='index').m_percentile) * 100).tolist()
    # freq_female = ((female_docs.m_percentile.value_counts(normalize=True).reset_index().sort_values(by='index').m_percentile) * 100).tolist()
    # embed()

   # Format axes
    def ax_settings(ax, var_name, x_min, x_max):
        ax.set_xlim(x_min, x_max)
        ax.set_yticks([])

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.spines['bottom'].set_edgecolor('#444444')
        ax.spines['bottom'].set_linewidth(2)

        # settings for x, y, varname
        ax.text(0.00, 0.1, var_name, fontsize=15, transform=ax.transAxes)
#        ax.text(0.02, 0.05, var_name, fontsize=17, fontweight="bold", transform=ax.transAxes)
        return None


    number_gp = len(percentile_ranges_to_display)
    # Manipulate each axes object in the left. Try to tune some parameters and you'll know how each command works.
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(nrows=number_gp,
                           ncols=2,
                           figure=fig,
                           width_ratios=[3, 1],
                           height_ratios=[1] * number_gp,
                           wspace=0.2, hspace=0.05
                           )
    ax = [None] * (number_gp + 1)
    # features = list(range(0, 100, 10))

    # Create a figure, partition the figure into 7*2 boxes, set up an ax array to store axes objects, and create a list of age group names.
    for idx, (start, end) in enumerate(percentile_ranges_to_display):
        ax[idx] = fig.add_subplot(gs[idx, 0])

        ax_settings(ax[idx], f'Percentile: {start}-{end}', 1951, 2015)

        clip = (1951, 2015)

        sns.kdeplot(data=df[
            (df.m_author_genders == 'male') & (df.m_percentile >= start) & (df.m_percentile < end)
            ].m_year, ax=ax[idx], shade=True, color="blue", legend=False, bw=.1, clip=clip)

#        ax[i].set(ylim=(0, 0.1))
        sns.kdeplot(data=df[
            (df.m_author_genders == 'female') & (df.m_percentile >= start) & (df.m_percentile < end)
            ].m_year,
                    ax=ax[idx], shade=True, color="red", legend=False, bw=.1, clip=clip)

        # scale plots
        # kdeplots by default take up the maximum range, i.e. male and female plots always show the
        # same max value
        scale_male = freq_male[idx] / (freq_male[idx] + freq_female[idx])
        scale_female = freq_female[idx] / (freq_male[idx] + freq_female[idx])
        print(idx, scale_male, scale_female)

        # line 1 -> male, line 2 -> female

        if freq_female[idx] > 0:
            ax[idx].lines[1].set_ydata(ax[idx].lines[1].get_ydata() * scale_female)
            female_collection = ax[idx].collections[1].get_paths()
            female_collection[0].vertices[:, 1] *= scale_female
        if freq_male[idx] > 0:
            ax[idx].lines[0].set_ydata(ax[idx].lines[0].get_ydata() * scale_male)
            # Collections, i.e. the patches under the lines
            male_collection = ax[idx].collections[0].get_paths()
            male_collection[0].vertices[:, 1] *= scale_male

        if idx < (number_gp - 1):
            ax[idx].set_xticks([])


    ax[0].legend(['Male', 'Female'], facecolor='w')

    # adding legends on the top axes object
    bar_height = 3
    for idx in range(len((percentile_ranges_to_display))):
        ax = fig.add_subplot(gs[idx, 1])

        ax.barh(y=0, width=freq_male[idx], color='#004c99', height=bar_height)
        ax.barh(y=3, width=freq_female[idx], color='red', height=bar_height)
        ax.set_xlim(0, max(20, math.ceil(max(freq_male + freq_female)) + 1))
        ax.invert_yaxis()
        ax.get_yaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='y', labelsize=14)

        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

        if idx + 1 < len(percentile_ranges_to_display):
            ax.tick_params(labelbottom=False)


    plt.suptitle(f'Decile Plots for {selection_name}',
                    weight='bold', fontsize=18)

    if filename:
        plt.savefig(Path(BASE_PATH, 'visualizations', 'topic_percentile_plots', filename),
                    dpi=100)
    if show_plot:
        plt.show()

    return ax


def plot_all_topics_and_general_approaches():



    dataset = JournalsDataset()

    for column in dataset.df.columns:

        print(column)

        if column.startswith('gen_approach'):
            topic_percentile_plot(dataset,
                                  selection_column=column,
                                  selection_name=f'General Approach {column[13:]}',
                                  filename=f'{column.replace(" ", "_")}.png',
                                  show_plot=False)
        if column.startswith('topic.'):
            topic_no = int(column[6:])
            topic_name = dataset.topics[topic_no]['name']
            topic_percentile_plot(dataset,
                                  selection_column=column,
                                  selection_name=f'Topic {topic_name}',
                                  filename=f'{topic_no}_{topic_name.replace(" ", "_")}.png',
                                  show_plot=False)


if __name__ == '__main__':

    # dataset = JournalsDataset()
    # topic_percentile_plot(dataset=dataset, selection_column="topic.16",
    #                       selection_name='Gender and Feminism')

    plot_all_topics_and_general_approaches()
