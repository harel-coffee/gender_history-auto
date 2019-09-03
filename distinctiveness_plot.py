from dataset import Dataset
import numpy as np
#from configuration import TOPIC_IDS_TO_NAME
from topics import TOPICS
from scipy.interpolate import make_interp_spline, BSpline

from scipy.interpolate import interp1d
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

import matplotlib.patches as mpatches
from IPython import embed
import re
import seaborn as sns
import math

from divergence_analysis import divergence_analysis


def distinctiveness_scatter_plot(dataset, terms=[], x_axis='mwr'):

    d = Dataset()

    male = d.copy().filter(author_gender='male')
    female = d.copy().filter(author_gender='female')
    divergence = divergence_analysis(dataset, female, male,
                                     topics_or_terms='topics',
                                     print_results=True,
                                     sort_by='dunning',
                                     number_of_terms_to_print=50)


    # 2: Set up plot
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(nrows=1,
                           ncols=1,
                           figure=fig,
                           width_ratios=[1],
                           height_ratios=[1],
                           wspace=0.2, hspace=0.05
                           )

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both')

    max_y = 0
    dunning_min = math.inf
    dunning_max = -math.inf
    dunning_max = 0
    for term in terms:
        topic_id = int(term[6:])
        freq_score = float(divergence[divergence['index'] == topic_id - 1]['frequency_score'])
        dunning = float(divergence[divergence['index'] == topic_id - 1]['dunning'])
        mwr = float(divergence[divergence['index'] == topic_id - 1]['mwr'])
        freq = float(divergence[divergence['index'] == topic_id - 1]['frequency_total'])

        if x_axis == 'mwr':
            x = mwr
            if abs(x-0.5) < 0.1:
                continue
        elif x_axis == 'frequency_score':
            x = freq_score
            if abs(x-0.5) < 0.05:
                continue
        y = freq



        ax.scatter(x=x, y=y)
        ax.annotate(term, (x*1.01,y*1.01))

        if freq > max_y:
            max_y = freq
        if abs(dunning) > dunning_max:
            dunning_max = abs(dunning)

    ax.set_ylim(0, max_y+0.002)

    if x_axis == 'mwr' or x_axis == 'frequency_score':
        ax.set_xlim(0, 1)
    elif x_axis == 'dunning':
        ax.set_xlim(-dunning_max*10, dunning_max*10)
        ax.set_xscale('symlog')

    from stats import StatisticalAnalysis
    vocabulary = [f'topic.{i}' for i in range(1, 71)]
    d_dtm = d.get_document_topic_matrix()
    c1_dtm = female.get_document_topic_matrix() * 300
    c2_dtm = male.get_document_topic_matrix() * 300
    s = StatisticalAnalysis(d_dtm, c1_dtm, c2_dtm, vocabulary)
    correlated_terms = s.correlation_coefficient(return_correlation_matrix=True)




#    plt.show()
    embed()


def pca_scatter():

    from stats import StatisticalAnalysis
    from sklearn.decomposition import PCA
    d = Dataset()
    vocabulary = [f'topic.{i}' for i in range(1, 71)]
    d_dtm = d.get_document_topic_matrix()
    male = d.copy().filter(author_gender='male')
    female = d.copy().filter(author_gender='female')
    c1_dtm = female.get_document_topic_matrix() * 300
    c2_dtm = male.get_document_topic_matrix() * 300
    s = StatisticalAnalysis(d_dtm, c1_dtm, c2_dtm, vocabulary)
    correlated_terms = s.correlation_coefficient(return_correlation_matrix=True)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(correlated_terms)

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(nrows=1,
                           ncols=1,
                           figure=fig,
                           width_ratios=[1],
                           height_ratios=[1],
                           wspace=0.2, hspace=0.05
                           )

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both')

    for i in range(70):
        topic_id = i + 1
        topic_name = TOPICS[topic_id]['name']
        x = pca_data[i, 0]
        y = pca_data[i, 1]
        ax.scatter(x=x, y=y)
        ax.annotate(topic_name, (x*1.01,y*1.01))

    plt.show()





if __name__ == '__main__':
    d = Dataset()
    terms = [f'topic.{i}' for i in range(1, 71)]
    termso=[
        'topic.28', 'topic.48', 'topic.35', 'topic.36', 'topic.22',
        'topic.55', 'topic.6' , 'topic.44', 'topic.67', 'topic.40'
    ]
#    distinctiveness_scatter_plot(d, terms, x_axis='frequency_score')
    pca_scatter()