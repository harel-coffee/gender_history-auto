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


def plot_multiple_terms(term_list, title=None):

    dataset = JournalsDataset()
    if term_list[0].startswith('topic.'):
        master_viz_data = load_master_viz_data(mode='topics')
    else:
        master_viz_data = load_master_viz_data(mode='terms')

    if len(term_list) % 5 == 0:
        cols = 5
        rows = len(term_list) // 5
    elif len(term_list) % 4 == 0:
        cols = 4
        rows = len(term_list) // 4
    elif len(term_list) % 3 == 0:
        cols = 3
        rows = len(term_list) // 3
    else:
        raise ValueError

    fig = plt.figure(figsize=(cols * 12, rows * 12))
    gs = gridspec.GridSpec(nrows=rows,
                           ncols=cols,
                           figure=fig,
                           wspace=0.2, hspace=0.2
                           # left=0.1

                           # width_ratios=[1],
                           # height_ratios=[1],
                           # wspace=0.1, hspace=0.05,
                           # left=0.1
                           )
    for idx, term in enumerate(term_list):
        row = idx // cols
        col = idx % cols

        title = term.capitalize()
        if term.startswith('topic.'):
            title = f"({term}) {dataset.topics[int(term[6:])]['name']}"

        ax = fig.add_subplot(gs[row, col])
        create_ngram_plot(
            subplot_ax=ax,
            term_or_topic_list=[term],
            plot_title=title,
            master_viz_data=master_viz_data
        )

    plt.savefig('test.png')
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

if __name__ == '__main__':
    term_list = [
        4, 17, 22, 26, 29,
        33, 42, 47, 51, 55,
        56, 58, 59, 64, 66,
        67, 70, 74, 86, 88
    ]
    term_list = [f'topic.{i}' for i in term_list]

    plot_multiple_terms(term_list=['race', 'racism', 'gay'],
                        )
