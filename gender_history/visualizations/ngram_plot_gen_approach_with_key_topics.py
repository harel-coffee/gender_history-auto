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




def plot_topic_frequency_with_6_terms(topic_id, term_list, store_plot=True):
    """
    Plots the topic weight over time with the frequency charts of 6 key terms
    The term_list can be selected from term prob, frex, and divergence analysis.

    :param topic_id: int
    :param term_list:
    :param store_plot:
    :return:
    """

    dataset = JournalsDataset()
    master_viz_data = load_master_viz_data(mode='terms')

    fig = plt.figure(figsize=(5 * 12, 2 * 12))
    gs = gridspec.GridSpec(nrows=2,
                           ncols=6,
                           figure=fig,
                           wspace=0.2, hspace=0.2,
                           # final 0.5 is to draw the colorbar into
                           width_ratios=[5, 5, 5, 5, 5, 0.5]
                           )

    # draw the topic weight plot into the first 2x2 chart section
    ax_topic = fig.add_subplot(gs[0:2, 0:2])
    create_ngram_plot(subplot_ax=ax_topic,
                      term_or_topic_list=[f'topic.{topic_id}'],
                      plot_title="Overall Topic Weight",
                      scale_factor=2)
    # slightly reduce title font size and padding and add y axis label
    ax_topic.set_ylabel('Mean topic weight', fontsize=28)
    ax_topic.set_title(label='Overall Topic Weight', weight='bold', fontsize=32, pad=30)

    # add the six terms
    for idx, term in enumerate(term_list):
        row = idx // 3
        col = idx % 3 + 2
        print(row, col, term)
        ax = fig.add_subplot(gs[row, col])
        create_ngram_plot(
            subplot_ax=ax,
            term_or_topic_list=[term],
            plot_title=term.capitalize(),
            master_viz_data=master_viz_data
        )

    # Draw colorbar
    lc = LineCollection([], cmap='coolwarm', norm=plt.Normalize(0.0, 1.0))
    cbar_ax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(lc,
                        cax=cbar_ax,
                        ticks = [0.025,  0.975],
                        fraction=0.03)
    cbar.ax.set_yticklabels(['Only men \nuse a term',
                             'Only women \nuse a term'])
    cbar.ax.tick_params(labelsize=28)

    # Draw title
    title = f'{dataset.topics[topic_id]["name"]}: Overall Weight and Key Terms'
    fig.suptitle(title, fontsize=60, weight='bold')

    # Add y axis labels for the first term plots
    fig.get_axes()[1].set_ylabel('Mean term frequency', fontsize=14)
    fig.get_axes()[4].set_ylabel('Mean term frequency', fontsize=14)

    if store_plot:
        filename = f'{topic_id}_{dataset.topics[topic_id]["name"]}.png'
        plt.savefig(Path(BASE_PATH, 'visualizations', 'topic_frequency_plots', filename))


    plt.show()

def plot_topic_graphs():

    topic_ids_to_terms = {
        46: ['family', 'children', 'women', 'marriage', 'household', 'parents'],
        61: ['women', 'female', 'gender', 'sexual', 'male', 'feminist'],
        71: ['sexual', 'women', 'sexuality', 'love', 'freud', 'emotional'],
        76: ['consumer', 'consumption', 'home', 'food', 'women', 'culture']
    }
    for topic_id, terms in topic_ids_to_terms.items():
        plot_topic_frequency_with_6_terms(topic_id=topic_id, term_list=terms)

if __name__ == '__main__':

    plot_topic_graphs()
