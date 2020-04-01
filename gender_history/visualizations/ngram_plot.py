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

def create_ngram_plot(
        term_or_topic_list: list,
        use_log_scale: bool=False,
        subplot_ax: Subplot=None,
        plot_title: str=None,
        plot_filename: str=None,
        show_plot: bool=False,
        show_male_female_bar: bool=None,
        scale_factor:int = 1,
        master_viz_data: dict=None
    ):

    print(f'Drawing ngram plot for {term_or_topic_list}')

    if subplot_ax and plot_filename:
        raise ValueError("Cannot save a plot that will be stored in a subplot.")
    if show_male_female_bar not in [None, 'horizontal', 'vertical']:
        raise ValueError("show_male_female_bar has to be None, horizontal, or vertical.")

    if term_or_topic_list[0].startswith('topic.') or term_or_topic_list[0].startswith('gen_approach_'):
        mode = 'topics'
    else:
        mode = 'terms'

    if not master_viz_data:
        master_viz_data = load_master_viz_data(mode)

    data = {term: master_viz_data[term] for term in term_or_topic_list}


    start_year = next(iter(data.values()))['year'][0]
    end_year = next(iter(data.values()))['year'][0]

    # 2: Set up the plot
    # if an ax is passed, we're creating a subplot for that axes. otherwise, we're creating a
    # complete plot here.
    if subplot_ax is None:
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(nrows=1,
                               ncols=1,
                               figure=fig,
                               width_ratios=[1],
                               height_ratios=[1],
                               wspace=0.2, hspace=0.05
                                )
        ax = fig.add_subplot(gs[0, 0])
    else:
        ax = subplot_ax

    ax.set_ylim(0, 1)
    ax.set_xlim(start_year + 2, end_year - 2)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both')

    # add all passed terms to chart
    for t in term_or_topic_list:
        x = data[t]['year']
        y = data[t]['freq']

        freq_scores = data[t]['freq_score']
        x_lin = np.linspace(min(data[t]['year']), max(data[t]['year']), 1000)
        spl = make_interp_spline(x, y, k=2)
        y_lin = spl(x_lin)
        spl_freq_score = make_interp_spline(x, freq_scores, k=1)
        freq_score_lin = spl_freq_score(x_lin)

        points = np.array([x_lin, y_lin]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0.0, 1.0)

        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        # Set the values used for colormapping
        lc.set_array(freq_score_lin)
        lc.set_linewidth(4 * scale_factor)
        line = ax.add_collection(lc)

        # Line below can be used to add a label to the right of the frequency line
        # ax.text(x=max(x)+1, y=y[-1], s=t, fontsize=14)

    # Set x axis (years)
    ax.set_xlim(x_lin.min(), x_lin.max())
    ax.tick_params(axis='x', labelsize=14 * scale_factor)

    # Set y axis (frequencies or topic weights)
    y_max = max([max(data[t]['freq']) for t in term_or_topic_list]) * 1.2
    ax.set_ylim(0, y_max)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    if use_log_scale:
        y_min = min([min(data[t]['freq']) for t in term_or_topic_list]) * 1.2
        ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=14 * scale_factor)


    # Set title
    if plot_title:
        ax.set_title(label=plot_title, weight='bold', fontsize=24 * scale_factor,
                     pad=20 * scale_factor)

    # Set color bar with male female indicator
    if show_male_female_bar:
        fig.colorbar(line, ax=ax, orientation=show_male_female_bar, fraction=0.1)

    if plot_filename:
        plt.savefig(Path(BASE_PATH, 'visualizations', 'ngram_plots', plot_filename))
    if show_plot:
        plt.show()
    if subplot_ax:
        return ax


def load_master_viz_data(mode, smoothing=5):
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
    if not mode in {'terms', 'topics'}:
        raise ValueError('mode has to be either "terms" or "topics".')

    master_viz_path = Path(BASE_PATH, 'data', 'dtms', f'viz_data_{mode}.pickle')

    if master_viz_path.exists():
        with open(master_viz_path, 'rb') as infile:
            return pickle.load(infile)

    print(f"Creating new master viz dataset for {mode}.")
    dataset = JournalsDataset()
    if mode == 'terms':
        _, master_vocabulary = dataset.get_vocabulary_and_document_term_matrix(max_features=100000)
        # load text info and turn it into term frequencies
        dataset.get_vocabulary_and_document_term_matrix(vocabulary=master_vocabulary,
                                                        use_frequencies=True, store_in_df=True)
    else:
        master_vocabulary = [f'topic.{i}' for i in range(1, 91)]
        for column in dataset.df.columns:
            if column.startswith('gen_approach_'):
                master_vocabulary.append(column)

    data = {}
    for t in master_vocabulary:
        data[t] = defaultdict(list)

    df = dataset.df

    # create time slices for every year
    for idx, year in enumerate(range(dataset.start_year, dataset.end_year + 1)):
        print(year)
        time_slice = df[(df.m_year >= year - smoothing) & (df.m_year <= year + smoothing)]
        time_slice_female = time_slice[time_slice.m_author_genders == 'female']
        time_slice_male = time_slice[time_slice.m_author_genders == 'male']

        for t in master_vocabulary:
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
            data[t]['freq_male'].append(freq_male)
            data[t]['freq_female'].append(freq_female)

    for t in master_vocabulary:
        data[t]['mean_freq_score'] = np.mean(data[t]['freq_score'])
        data[t]['mean_freq'] = np.mean(data[t]['freq'])
        data[t]['freq_score_range'] = max(data[t]['freq_score']) - min(data[t]['freq_score'])

    with open(master_viz_path, 'wb') as outfile:
        pickle.dump(data, outfile)
    return data



if __name__ == '__main__':
    # create_ngram_plot(term_or_topic_list=['women\'s history'],
    #                   use_log_scale=False, show_plot=True,
    #                   plot_title='Women\'s History')
    # create_ngram_plot(term_or_topic_list=['gen_approach_Women\'s and Gender History'],
    #                   show_plot=True, plot_title='gen approach')
    # create_ngram_plot(term_or_topic_list=['topic.61'],
    #                   show_plot=True, plot_title='topic')
    create_ngram_plot(term_or_topic_list=['native', 'indian'],
                      plot_title='"indian" (top) and "native" (bottom)',
                      show_plot=True,
                      plot_filename='native_and_indian.png')

