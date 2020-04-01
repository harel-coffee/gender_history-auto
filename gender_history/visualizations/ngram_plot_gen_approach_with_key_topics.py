from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
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

import pickle
from gender_history.visualizations.ngram_plot import create_ngram_plot, load_master_viz_data




def plot_gen_approach_with_key_topics(gen_approach, topic_list, store_plot=True):
    """
    Plots the topic weight over time with the frequency charts of 6 key terms
    The term_list can be selected from term prob, frex, and divergence analysis.

    :param topic_id: int
    :param term_list:
    :param store_plot:
    :return:
    """

    dataset = JournalsDataset()

    if len(topic_list) == 4:
        cols = 4

    fig = plt.figure(figsize=(cols * 12, 2 * 12))
    gs = gridspec.GridSpec(nrows=2,
                           ncols=cols + 1,  # + 1 for color bar
                           figure=fig,
                           wspace=0.2, hspace=0.2,
                           # final 0.5 is to draw the colorbar into
                           width_ratios=[5] * cols + [0.5]
                           )

    # draw the topic weight plot into the first 2x2 chart section
    ax_topic = fig.add_subplot(gs[0:2, 0:2])
    create_ngram_plot(subplot_ax=ax_topic,
                      term_or_topic_list=[f'gen_approach_{gen_approach}'],
                      plot_title="General Approach Weight",
                      scale_factor=2)
    # slightly reduce title font size and padding and add y axis label
    ax_topic.set_ylabel('Mean topic weight', fontsize=28)
    ax_topic.set_title(label='Overall Approach Weight', weight='bold', fontsize=32, pad=30)

    # add the six terms
    for idx, topic_id in enumerate(topic_list):
        row = idx // (cols - 2)
        col = idx % (cols - 2) + 2
        print(row, col, topic_id)
        ax = fig.add_subplot(gs[row, col])
        create_ngram_plot(
            subplot_ax=ax,
            term_or_topic_list=[f'topic.{topic_id}'],
            plot_title=f'Topic: {dataset.topics[topic_id]["name"]}'
        )

    # Draw colorbar
    lc = LineCollection([], cmap='coolwarm', norm=plt.Normalize(0.0, 1.0))
    cbar_ax = fig.add_subplot(gs[:, cols])
    cbar = fig.colorbar(lc,
                        cax=cbar_ax,
                        ticks = [0.025,  0.975],
                        fraction=0.03)
    cbar.ax.set_yticklabels(['Only men \nwrite on a topic',
                             'Only women \nwrite on a topic'])
    cbar.ax.tick_params(labelsize=28)

    # Draw title
    title = f'{gen_approach} (General Approach): Overall Weight and Key Terms'
    fig.suptitle(title, fontsize=60, weight='bold')

    # Add y axis labels for the first term plots
    fig.get_axes()[1].set_ylabel('Mean term frequency', fontsize=14)
    fig.get_axes()[4].set_ylabel('Mean term frequency', fontsize=14)

    if store_plot:
        filename = f'{gen_approach}.png'
        plt.savefig(Path(BASE_PATH, 'visualizations', 'topic_frequency_plots', filename))


    plt.show()

'''
defaultdict(set,
            {'Indigenous History': {1, 36, 38, 65, 72, 83},
             'History of Race and Racism': {2, 28, 55, 68},
             'Social History': {3, 10, 15, 27, 30, 42, 46, 57, 63, 73, 75, 89},
             'Historiography': {4, 6, 11, 21, 25, 45, 73, 81, 87},
             'Political History': {4, 17, 22, 26, 29, 33, 42, 47, 51, 55, 56, 58, 59, 64,
              66, 67, 70, 74, 86, 88},
             'Transnational History': {6, 8, 14},
             'Economic History': {9, 20, 34, 50, 59, 60, 63, 75},
             'Environmental History': {9, 10},
             'Jewish History': {12, 28},
             'Noise': {13, 16, 18, 24, 84},
             'Colonies and Empires': {14, 22, 48, 49, 83},
             'Religious History': {23, 54},
             'Intellectual History': {26, 62, 82},
             'Military History': {31},
             'History of Medicine and Public Health': {32, 52},
             'Art History': {39},
             'Colonies & Empires': {40, 44, 53, 78},
             'Islamic History': {43},
             'Cultural History': {45, 67, 71, 76, 89, 90},
             "Women's and Gender History": {46, 61, 71, 76},
             'Medieval History': {69},
             'Classics': {77},
             'Legal History': {79}})


'''

def plot_general_approach_graphs():

    names_and_topic_lists = [
        ("Women's and Gender History",      [46, 61, 71, 76]),
        ('Social History',                  [57, 73, 46, 75]),
        ('Historiography',                  [6, 25, 87, 45])
    ]
    pass

if __name__ == '__main__':

    plot_gen_approach_with_key_topics(
        gen_approach='Historiography',
        topic_list=[6, 25, 87, 45]
    )

