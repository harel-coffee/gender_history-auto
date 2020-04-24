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

import pandas as pd


import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

def draw_gender_frequency_scatterplot(
        dataset: Dataset,
        figsize: int,
        filename: str=None,
        show_labels: bool=True,
        transparent_image: bool=False,
        title: str=None,
        dynamic_y_coords: bool=False,
        show_plot: bool=True,
        use_absolute_weights: bool=False
):
    """

    dynamic_y_coords: default (False) uses 0.001 to 0.1. With dynamic y_coords, they are adjusted
    by local min/max

    :param figsize:
    :param filename:
    :param show_labels:
    :param transparent_image:

    :param use_absolute_weights: normally, this chart uses frequencies for men and women as if
    they had published the same number of articles. With use_absolute_weights, it displays the
    absolute weight contributed by men and women (which skews the chart heavily towards men).

    :return:
    """

    c1 = dataset.copy().filter(author_gender='female')
    c2 = dataset.copy().filter(author_gender='male')
    div = DivergenceAnalysis(dataset, c1, c2, sub_corpus1_name='female', sub_corpus2_name='male',
                             analysis_type='topics', sort_by='dunning')
    divergence_df = div.run_divergence_analysis()


    fig = plt.figure(figsize=(figsize, figsize))
    gs = gridspec.GridSpec(nrows=2,
                           ncols=1,
                           figure=fig,
                           width_ratios=[1],
                           height_ratios=[1, 0.1],
                           wspace=0.2, hspace=0.1
                           )

    ax = fig.add_subplot(gs[0, 0])

    x_coords = []
    y_coords = []

    x_coords_gender = []
    y_coords_gender = []

    x_coords_female_assoc = []
    y_coords_female_assoc = []


    for topic_id in range(1, 91):

        gen_approach = dataset.topics[topic_id]['gen_approach']
        if isinstance(gen_approach, str) and gen_approach.find('Noise') > -1:
            continue

        x = divergence_df[divergence_df['topic_id'] == topic_id]['frequency_score'].values[0]
        y = dataset.df[f'topic.{topic_id}'].mean()

        if use_absolute_weights:
            weight_female = divergence_df[divergence_df['topic_id'] == topic_id]['f female'].values[0]
            weight_both = divergence_df[divergence_df['topic_id'] == topic_id]['freq both'].values[0]
            x = weight_female * len(c1) / (weight_both * (len(c1) + len(c2)))


        if y < 0.0012:
            continue

        if topic_id in {46, 61, 71}:
            x_coords_gender.append(x)
            y_coords_gender.append(y)
        elif topic_id in {32, 76}:
            x_coords_female_assoc.append(x)
            y_coords_female_assoc.append(y)
        else:
            x_coords.append(x)
            y_coords.append(y)

        topic_name = dataset.topics[topic_id]['name']

        if show_labels:
            ax.annotate(topic_name, (x, y + 0.000))



    # ax.set_ylim(0, 1)
    # ax.set_xlim(dataset.start_year + 2, dataset.end_year - 2)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both')

    # # Set the values used for colormapping
    # normalized_cmap = matplotlib.colors.Normalize(vmin=min(color_codes),
    #                                               vmax=max(color_codes))

    # ax.scatter(x_coords, y_coords, s=300)
    ax.set_xlim(0, 1)


    # y axis
    # ax.set_ylabel('Topic Weight')
    # ax.label_params(labelsize=20)

    if dynamic_y_coords:
        ax.set_ylim(min(y_coords) * 0.9, max(y_coords) * 1.1)
    else:
        ax.set_ylim(0.001, 0.15)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    ax.set_yticks([0.001, 0.01, 0.1])
    ax.grid(b=True, which='minor', color='lightgray', linestyle='--')
    ax.tick_params(labelsize=15)

    # ax.yaxis.set_minor_locator(MultipleLocator(5))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ax.scatter(x_coords, y_coords, s=200, c=colors[7])
    ax.scatter(x_coords_gender, y_coords_gender, s=200, c=colors[1])
    ax.scatter(x_coords_female_assoc, y_coords_female_assoc, s=200, c='#fdbf6f')




    male_data, female_data, years = get_number_of_male_and_female_authored_articles_by_year(
        start_year=dataset.start_year, end_year=dataset.end_year + 1,
        dataset_name=dataset.dataset_type
    )

    gender_ax = fig.add_subplot(gs[1, 0])
    gender_ax.stackplot(years, female_data, male_data, colors=(colors[1], colors[0]))
    gender_ax.margins(0,0)
    gender_ax.tick_params(labelsize=15)
    gender_ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    gender_ax.set_yticks([np.round(female_data[0], 2)])

    x_ticks = sorted({y for y in years if y % 5 == 0})
                      # .union({min(years)})
                      # .union({max(years)}))
    gender_ax.set_xticks(x_ticks)

    if title:
        ax.set_title(label=title, weight='bold', fontsize=24,
                     pad=20)

    if figsize == 36:
        dpi = 300
    else:
        dpi = 300

    if filename:
        plt.savefig(Path(BASE_PATH, 'visualizations', 'gender_frequency_scatterplots', filename),
                dpi=dpi, transparent=transparent_image)
    if show_plot:
        plt.show()


def get_number_of_male_and_female_authored_articles_by_year(start_year, end_year, dataset_name
                                                            ) -> (list, list, list):
    """
    returns number of male and female authored articles and list of years for male/female subplot

    male + female data add up to 1, i.e. they show percentages of articles written by men and
    women

    :param start_year:
    :param end_year:
    :return:
    """

    if dataset_name == 'journals':
        d = JournalsDataset()
    else:
        d = DissertationDataset()

    male_data = [0] * (d.end_year - d.start_year + 1)
    female_data = [0] * (d.end_year - d.start_year + 1)

    for _, row in d.df.iterrows():
        if row.m_author_genders == 'male':
            male_data[row.m_year - d.start_year] += 1
        elif row.m_author_genders == 'female':
            female_data[row.m_year - d.start_year] += 1
        else:
            pass

    rolling_mean_male = pd.DataFrame(male_data).rolling(center=True, window=5, min_periods=1).mean()[0].tolist()
    rolling_mean_female = pd.DataFrame(female_data).rolling(center=True, window=5, min_periods=1).mean()[0].tolist()

    male_data = np.array(rolling_mean_male[start_year - d.start_year: end_year + 1 - d.start_year])
    female_data = np.array(rolling_mean_female[start_year - d.start_year: end_year + 1 - d.start_year])

    totals = male_data + female_data
    male_data = male_data / totals
    female_data = female_data / totals

    years = [i for i in range(start_year, end_year + 1)]

    assert len(male_data) == len(female_data) == len(years)

    return male_data, female_data, years


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

def draw_journal_and_dissertation_overview():

    for use_absolute_weights in [True, False]:
        for dataset_name in ['dissertations', 'journals']:

            if dataset_name == 'journals':
                dataset = JournalsDataset()
                dataset.filter(start_year=1951, end_year=2010)
                title = 'Journals, 1950-2010'
            else:
                dataset = DissertationDataset()
                dataset.filter(start_year=1980, end_year=2010)
                title = 'Dissertations, 1980-2010'

            filename = f'topic_scatter_{dataset_name}.png'
            if use_absolute_weights:
                filename = f'topic_scatter_{dataset_name}_absolute_weights.png'
                title += ', Absolute Weights'

            draw_gender_frequency_scatterplot(
                dataset,
                figsize=12, show_labels=True, transparent_image=False,
                dynamic_y_coords=False,
                filename=filename,
                show_plot=True,
                title=title,
                use_absolute_weights=use_absolute_weights
            )



def draw_all_years():

    for start_year in range(1960, 2010, 10):
        for dataset_name in ['dissertations', 'journals']:

            if dataset_name == 'journals':
                dataset = JournalsDataset()
            else:
                dataset = DissertationDataset()
                if start_year < 1980:
                    continue

            dataset.filter(start_year=start_year, end_year=start_year + 9)

            draw_gender_frequency_scatterplot(
                dataset,
                figsize=12, show_labels=True, transparent_image=False,
                dynamic_y_coords=False,
                filename=f'topic_scatter_{dataset_name}_{start_year}-{start_year+9}.png',
                show_plot=True,
                title=f'{dataset_name.capitalize()}, {start_year}s'
            )


if __name__ == '__main__':
    # draw_scatterplots_of_journals()

    #draw_journal_and_dissertation_overview()
    draw_all_years()

    # dataset = JournalsDataset()
    # dataset.filter(start_year=1951, end_year=1959)
    # draw_gender_frequency_scatterplot(
    #     dataset,
    #     figsize=12, show_labels=True, transparent_image=False,
    #     dynamic_y_coords=True,
    #     filename=f'{start_year}.png'
    # )