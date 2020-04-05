from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.datasets.dataset import Dataset
from gender_history.utilities import BASE_PATH

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors


import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def draw_gender_frequency_scatterplot(
        dataset: Dataset,
        figsize: int,
        filename: str,
        show_labels: bool=True,
        transparent_image: bool=False,
        dynamic_y_coords: bool=False
):
    """

    dynamic_y_coords: default (False) uses 0.001 to 0.1. With dynamic y_coords, they are adjusted
    by local min/max

    :param figsize:
    :param filename:
    :param show_labels:
    :param transparent_image:
    :return:
    """

    c1 = dataset.copy().filter(author_gender='female')
    c2 = dataset.copy().filter(author_gender='male')
    div = DivergenceAnalysis(dataset, c1, c2)
    divergence_df = div.run_divergence_analysis('topics')

    df_sorted_by_years = dataset.df.sort_values(by='m_year')


    fig = plt.figure(figsize=(figsize, figsize))
    gs = gridspec.GridSpec(nrows=1,
                           ncols=2,
                           figure=fig,
                           width_ratios=[1, 0.05],
                           height_ratios=[1],
                           wspace=0.2, hspace=0.05
                           )

    ax = fig.add_subplot(gs[0, 0])

    x_coords = []
    y_coords = []
    color_codes = []

    for topic_id in range(1, 91):

        gen_approach = dataset.topics[topic_id]['gen_approach']
        if isinstance(gen_approach, str) and gen_approach.find('Noise') > -1:
            continue

        x = divergence_df[divergence_df['index'] == topic_id - 1]['frequency_score'].values[0]
        x_coords.append(x)
        y = dataset.df[f'topic.{topic_id}'].mean()
        y_coords.append(y)

        topic_array_sorted_by_year = np.array(df_sorted_by_years[f'topic.{topic_id}'])
        total_topic_weight = topic_array_sorted_by_year.sum()
        topic_weight_so_far = 0
        median_year = None
        for idx, article_topic_weight in enumerate(topic_array_sorted_by_year):
            if topic_weight_so_far > total_topic_weight / 2:
                median_year = df_sorted_by_years.iloc[idx]['m_year']
                break
            topic_weight_so_far += article_topic_weight

        color_codes.append(median_year)

        topic_name = dataset.topics[topic_id]['name']

        if show_labels:
            ax.annotate(topic_name, (x, y + 0.000))



    ax.set_ylim(0, 1)
    ax.set_xlim(dataset.start_year + 2, dataset.end_year - 2)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both')

    # Set the values used for colormapping
    normalized_cmap = matplotlib.colors.Normalize(vmin=min(color_codes),
                                                  vmax=max(color_codes))

    # ax.scatter(x_coords, y_coords, s=300)
    ax.set_xlim(0, 1)


    # y axis
    # ax.set_ylabel('Topic Weight')
    # ax.label_params(labelsize=20)

    if dynamic_y_coords:
        ax.set_ylim(min(y_coords) * 0.9, max(y_coords) * 1.1)
    else:
        ax.set_ylim(0.002, 0.1)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(b=True, which='minor', color='lightgray', linestyle='--')
    ax.tick_params(labelsize=15)

    # ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.scatter(x_coords, y_coords, c=color_codes, s=200, cmap='jet',
               norm=normalized_cmap)

    lc = LineCollection([], cmap='jet', norm=plt.Normalize(0.0, 1.0))

    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(lc,
                        cax=cbar_ax,
                        ticks = [0.0, 0.50, 1.0],
                        fraction=0.03)
    cbar.ax.set_yticklabels([
        min(color_codes),
        int(np.median(np.array(color_codes))),
        max(color_codes)
    ])
    cbar.ax.tick_params(labelsize=14)

    if figsize == 36:
        dpi = 300
    else:
        dpi = 600

    plt.savefig(Path(BASE_PATH, 'visualizations', 'gender_frequency_scatterplots', filename),
                dpi=dpi, transparent=transparent_image)
    plt.show()


def draw_set_of_gender_frequency_scatterplots():
    """
    Draws a set of three gender frequency scatterplots either using or not using the
    dataset with 500 documents per gender and five year period or not.

    - a large labeling copy that makes it easy to distinguish all of the labels
    - a transparent base layer for labeling only a subset of topics
    - a small version with all labels (though they look jumbled).

    :return:
    """

    for (name, use_equal_samples_dataset) in [('_eq_samples_dataset', True), ('', False)]:

        if use_equal_samples_dataset:
            dataset = JournalsDataset(use_equal_samples_dataset=use_equal_samples_dataset)
        else:
            dataset = JournalsDataset()

        draw_gender_frequency_scatterplot(
            dataset,
            figsize=36, show_labels=True, transparent_image=False,
            filename=f'gfs_labeling_copy{name}.png'
        )

        draw_gender_frequency_scatterplot(
            dataset,
            figsize=12, show_labels=False, transparent_image=True,
            filename=f'gfs_transparent_base_layer{name}.png'
        )

        draw_gender_frequency_scatterplot(
            dataset,
            figsize=12, show_labels=True, transparent_image=False,
            filename=f'gfs_standard_all_labels{name}.png',
        )

        break

def draw_scatterplots_of_journals():
    """
    Creates scatter plots for all journals as well as (JAH + AHR) and all journals
    minus History and Theory

    :return:
    """

    valid_journals = {
        'Comparative Studies in Society and History',
        'The Journal of Modern History',
        'The Journal of American History',
        'Journal of World History',
        'The Journal of Interdisciplinary History',
        'Journal of Social History',
        'The American Historical Review',
        'Reviews in American History',
        'History and Theory',
        'Ethnohistory'
    }

    for journal in valid_journals:

        dataset = JournalsDataset()
        dataset.filter_by_journal([journal])
        draw_gender_frequency_scatterplot(
            dataset,
            figsize=36, show_labels=True, transparent_image=False,
            filename=f'single_journal{journal.replace(" ", "_")}.png',
            dynamic_y_coords=True
        )

    # all except history and theory
    valid_journals.remove('History and Theory')
    dataset = JournalsDataset()
    dataset.filter_by_journal(list(valid_journals))
    draw_gender_frequency_scatterplot(
        dataset,
        figsize=36, show_labels=True, transparent_image=False,
        filename=f'all_except_history_and_theory.png',
        dynamic_y_coords=True

    )

    # AHR and JAH
    dataset = JournalsDataset()
    dataset.filter_by_journal([
        'The Journal of American History',
        'The American Historical Review'
    ])
    draw_gender_frequency_scatterplot(
        dataset,
        figsize=36, show_labels=True, transparent_image=False,
        filename=f'ahr_and_jah.png',
        dynamic_y_coords=True
    )





if __name__ == '__main__':
    draw_scatterplots_of_journals()