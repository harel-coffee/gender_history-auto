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

import pickle

def create_ngram_plot(
        term_or_topic_list: list,
        use_log_scale: bool=False
    ):

    if term_or_topic_list[0].startswith('topic.'):
        mode = 'topics'
    else:
        mode = 'terms'

    # 1: Init and get data
    dataset = JournalsDataset()
    # data = get_viz_data(dataset, term_or_topic_list, smoothing=5)
    # with open('testdata.pickle', 'wb') as out:
    #     pickle.dump(data, out)
    with open('testdata.pickle', 'rb') as infile:
        data = pickle.load(infile)




    # 2: Set up the plot
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(nrows=1,
                           ncols=1,
                           figure=fig,
                           width_ratios=[1],
                           height_ratios=[1],
                           wspace=0.2, hspace=0.05
                           )

    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(0, 1)
    ax.set_xlim(dataset.start_year + 2, dataset.end_year - 2)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both')

    for t in term_or_topic_list:
        x = data[t]['year']
        y = data[t]['freq']
        freq_scores = data[t]['freq_score']
        x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
        spl = make_interp_spline(x, y, k=2)
        y_lin = spl(x_lin)
        spl_freq_score = make_interp_spline(x, freq_scores, k=1)
        freq_score_lin = spl_freq_score(x_lin)

        # points[0] = (x[0], y[0]
        points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
        # segments[0] = 2x2 matrix. segments[0][0] = points[0]; segments[0][1] = points[1]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0.0, 1.0)

        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        # Set the values used for colormapping
        lc.set_array(freq_score_lin)
        lc.set_linewidth(4)
        line = ax.add_collection(lc)

        ax.text(x=max(x)+1, y=y[-1], s=t, fontsize=14)

    fig.colorbar(line, ax=ax, orientation='horizontal', fraction=0.1)
    ax.set_xlim(x_lin.min(), x_lin.max())

    # ax.set_ylim(0, y_lin.max() * 1.1)

    if mode == 'terms':
        y_max = max([max(data[t]['freq']) for t in term_or_topic_list]) * 1.2
    else:
        y_max = 0.4
    ax.set_ylim(0.00000001, y_max)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    if use_log_scale:
        y_min = min([min(data[t]['freq']) for t in term_or_topic_list]) * 1.2
        ax.set_yscale('log')

    plt.savefig(Path(BASE_PATH, 'visualizations', 'ngram_plots', 'test.png'))
    plt.show()
    embed()

def get_viz_data(dataset, token_list, smoothing=5):
    """
    For every term or topic in the token list, this function returns a dict consisting of:
    - year          (list of years in the dataset)
    - freq_score    (list of yearly frequency scores)
    - freq          (list of yearly frequency of the term)
    - mean_freq_score, mean_freq, freq_score_range (floats)

    :param dataset:
    :param token_list:
    :param smoothing:
    :return:
    """


    # load text info and turn it into term frequencies
    dataset.get_document_term_matrix(vocabulary=token_list, store_in_df=True)
    for idx, row in dataset.df.iterrows():
        text_len = len(row.text.split())
        dataset.df.at[idx, 'text_len'] = text_len
    for t in token_list:
        dataset.df[t] = dataset.df[t] / dataset.df['text_len']

    data = {}
    for t in token_list:
        data[t] = defaultdict(list)

    df = dataset.df

    # create time slices for every year
    for idx, year in enumerate(range(dataset.start_year, dataset.end_year)):
        time_slice = df[(df.year >= year - smoothing) & (df.year <= year + smoothing)]
        time_slice_female = time_slice[time_slice.author_genders == 'female']
        time_slice_male = time_slice[time_slice.author_genders == 'male']

        for t in token_list:
            freq_both = time_slice[t].mean()
            freq_female = time_slice_female[t].mean()
            freq_male = time_slice_male[t].mean()

            # if a term doesn't appear, it is neutral
            if (freq_male + freq_female) == 0:
                freq_score = 0.5
            else:
                freq_score = freq_female / (freq_female + freq_male)

            data[t]['year'].append(year)
            data[t]['freq_score'].append(freq_score)
            data[t]['freq'].append(freq_both)

    for t in token_list:
        data[t]['mean_freq_score'] = np.mean(data[t]['freq_score'])
        data[t]['mean_freq'] = np.mean(data[t]['freq_both'])
        data[t]['freq_score_range'] = max(data[t]['freq_score']) - min(data[t]['freq_score'])

    return data


if __name__ == '__main__':
    create_ngram_plot(term_or_topic_list=['gender', 'women'],
                      use_log_scale=True)

