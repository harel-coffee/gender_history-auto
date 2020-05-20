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
import csv


from collections import defaultdict
import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes._subplots import Subplot

import pickle
from gender_history.visualizations.ngram_plot import create_ngram_plot, load_master_viz_data

import matplotlib.pyplot as plt

def get_co_use_data(topic_id=61):

    percentile_cutoff = 90

    d_all = JournalsDataset()
    d_all.topic_score_filter(topic_id=topic_id, min_percentile_score=percentile_cutoff)


    output_data = {d_all.topics[i]['name']: {} for i in range(1, 91)
                   if not d_all.topics[i]['name'].startswith('Noise')}

    print(len(output_data))


    for start_year in [1960, 1970, 1980, 1990, 2000]:

        print('\n\n', start_year)


        d = JournalsDataset()
        d.filter(start_year=start_year, end_year=start_year + 9)

        doc_topic_matrix = d.get_document_topic_matrix()

        co_use_matrix = doc_topic_matrix.T * doc_topic_matrix

        topic_selector = [f'topic.{i}' for i in range(1, 91)]
        topic_df = d.df[topic_selector]


        for idx, (topic, correlation) in enumerate(topic_df.corr()[f'topic.{topic_id}'].sort_values(
                ascending=False).iteritems()):

            if idx == 0:
                continue

            if correlation > 0.03:
                tid = int(topic[6:])
                # print(tid, d.topics[tid]['name'], correlation)


        print("\nco use")
        co_use_vector = np.array(co_use_matrix[:, topic_id - 1].todense()).flatten()
        for co_use_id in co_use_vector.argsort()[::-1][:10]:
            co_use_topic_id = co_use_id + 1
            # print(co_use_id, d.topics[co_use_topic_id]['name'],
            #       co_use_matrix[topic_id - 1, co_use_id] / sum(co_use_vector))

        from scipy.stats import entropy
        # co_use_vector[topic_id - 1] = 0
        print("\nentropy", entropy(co_use_vector / sum(co_use_vector)))

        number_of_journals_in_decade = len(JournalsDataset().filter(start_year=start_year,
                                                                    end_year=start_year + 9))
        d_all_in_years = d_all.copy().filter(start_year=start_year, end_year=start_year + 9)
        intersections = {}
        for i in range(1, 91):
            overlapping_articles = len(d_all_in_years.copy().topic_score_filter(topic_id=i,
                                                                        min_percentile_score=90))
            intersections[i] = overlapping_articles / number_of_journals_in_decade

        # print("\noverlap:")
        # for topic_id, overlap in sorted(intersections.items(), key=lambda x:x[1], reverse=True)[:10]:
        #     print(topic_id, d.topics[topic_id]['name'], overlap)


        print("\nslice")
        topic_weights = {}
        for topic_id in range(1, 91):
            if d.topics[topic_id]['name'].startswith('Noise'):
                continue
            topic_weights[topic_id] = d_all_in_years.df[f'topic.{topic_id}'].mean()

        for topic_id, slice in sorted(topic_weights.items(), key=lambda x: x[1], reverse=True):
            topic_name = d.topics[topic_id]['name']
            output_data[topic_name][start_year] = slice / sum(topic_weights.values())
            # print(topic_id, topic_name, slice / sum(topic_weights.values()))



        import matplotlib.pyplot
        s = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)
        labels = [i[0] for i in s]
        sizes = [i[1]/sum(topic_weights.values()) for i in s]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'{start_year}, {percentile_cutoff}th percentile')
        plt.show()

    sorted_output_data = sorted(output_data.items(), key=lambda x: x[1][1990], reverse=True)
    df = pd.DataFrame()
    df['labels'] = [i[0] for i in sorted_output_data]
    for year in [1960, 1970, 1980, 1990, 2000]:
        df[year] = [i[1][year] for i in sorted_output_data]

    df.to_csv()

    embed()

def gini(dataset, topic_id):

    doc_topic_matrix = dataset.get_document_topic_matrix()
    co_use_matrix = doc_topic_matrix.T * doc_topic_matrix
    co_use_vector = np.array(co_use_matrix[:, topic_id - 1].todense()).flatten()
    arr = co_use_vector

    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

def ginis():

    d_all = JournalsDataset()
    # d_all.topic_score_filter(topic_id=61, min_percentile_score=90)

    ginis_gender = []
    ginis_all_topics = []
    gender_topics = []
    for start_year in range(d_all.start_year, d_all.end_year + 1):
        year_d = d_all.copy().filter(start_year=start_year, end_year=start_year)

        ginis_gender.append(gini(year_d, topic_id=61))
        gender_topics.append(year_d.df['topic.61'].mean())

        ginis_all_topics_year = []
        for i in range(1, 91):
            if d_all.topics[i]['name'].startswith('Noise'):
                continue
            ginis_all_topics_year.append(gini(year_d, topic_id=i))

        ginis_all_topics.append(np.array(ginis_all_topics_year).mean())

    gini_gender_rolling = pd.DataFrame(ginis_gender).rolling(center=True, window=7).mean()[0].tolist()[3:-5]
    gini_all_topics_rolling = pd.DataFrame(ginis_all_topics).rolling(center=True, window=7).mean()[0].tolist()[3:-5]
    gender_rolling = pd.DataFrame(gender_topics).rolling(center=True, window=7).mean()[0].tolist()[3:-5]
    years = [i for i in range(d_all.start_year, d_all.end_year + 1)][3:-5]

    plt.plot(years, gini_all_topics_rolling)
    plt.plot(years, gini_gender_rolling)
    plt.title("Gini values for women and gender in all articles")
    plt.show()

    df = pd.DataFrame()
    df['years'] = years
    df['gini_gender'] = gini_gender_rolling
    df['gini_all_topics'] = gini_all_topics_rolling
    df['gender_rolling'] = gender_rolling
    df.to_csv(Path(BASE_PATH, 'visualizations', 'plotly_data', 'gini.csv'))



def get_lorenz_data(dataset):

    doc_topic_matrix = dataset.get_document_topic_matrix()
    co_use_matrix = doc_topic_matrix.T * doc_topic_matrix
    co_use_vector = np.array(co_use_matrix[:, 61 - 1].todense()).flatten()
    x_all = co_use_vector

    x_all.sort()
    X_lorenz = x_all.cumsum() / x_all.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)

    return X_lorenz


def generate_lorenz_curves():

    d = JournalsDataset()
    d_topic = d.copy().topic_score_filter(topic_id=61, min_percentile_score=90)

    for start_year in [1960, 1970, 1980, 1990, 2000]:
        year_d = d.copy().filter(start_year=start_year, end_year=start_year + 9)
        year_topic_d = d_topic.copy().filter(start_year=start_year, end_year=start_year + 9)

        lorenz_all = get_lorenz_data(year_d)
        gini_all = np.round(gini(year_d), 3)
        lorenz_topic = get_lorenz_data(year_topic_d)
        gini_topic = np.round(gini(year_topic_d), 3)

        fig, ax = plt.subplots(figsize=[6, 6])
        ## scatter plot of Lorenz curve
        ax.scatter(np.arange(lorenz_all.size) / (lorenz_all.size - 1), lorenz_all,
                   marker='x', color='darkgreen', s=100)
        ax.scatter(np.arange(lorenz_topic.size) / (lorenz_topic.size - 1), lorenz_topic,
                   marker='+', color='blue', s=100)
        ## line plot of equality
        ax.plot([0, 1], [0, 1], color='k')
        ax.set_title(f'Lorenz Curve for the Women and Gender Topic in the {start_year}s\n'
                     f'Green: all articles. Blue: top decile for women and gender.\n'
                     f'Gini coefficient.  all articles = {gini_all}. top decile = {gini_topic}')
        plt.show()


def pie_plot():

    data = pd.read_csv(Path(BASE_PATH, 'visualizations', 'plotly_data', 'broadening.csv'))

    labels = list(data['labels'])
    sizes = list(data['1990'])

    colors = ['#9f9f9f', '#4e4e4e'] * 50


    import matplotlib.pyplot
    from matplotlib.cm import get_cmap
    from matplotlib.patches import Wedge

    cmap = get_cmap('Set3').colors

    for color_idx, topic_name in enumerate([
        'Women and Gender',
        'History, Memory, and Fiction',
        'Cultural Turn',
        'Family',
        'General Historiography',
        'Legal History',
        '20th Century U.S. Historiography',
        '20th Century Labor History'
    ]):
        topic_idx = labels.index(topic_name)

        colors[topic_idx] = cmap[color_idx]

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    start_degree = 90
    for topic_idx, size in enumerate(sizes):
        degrees_of_topic = size / sum(sizes) * 360

        if topic_idx < 4:
            r = 0.45
            width=0.77777777 * r
        elif topic_idx < 15:
            r = 0.425
            width = 0.7647 * r
        else:
            r = 0.4
            width = 0.75 * r

        w = Wedge(center=0.5, r=r, width=width,
                  theta1=start_degree, theta2=start_degree + degrees_of_topic,
                  facecolor=colors[topic_idx],
                  edgecolor="#333333", linewidth=1)
        ax.add_patch(w)
        start_degree += degrees_of_topic

    # ax.pie(sizes, labels=labels, autopct='%1.1f%%',
    #         shadow=False, startangle=90, colors=colors, explode=explodes,
    #        radius=1,
    #        wedgeprops=dict(width=0.7, edgecolor='w'))
    # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'{1970}, {90}th percentile')



    plt.show()

    embed()



if __name__ == '__main__':

    j = JournalsDataset()
    embed()

    # get_co_use_data()
    # pie_plot()
    # ginis()
    ginis()