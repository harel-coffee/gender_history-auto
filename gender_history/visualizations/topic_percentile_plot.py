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

    percentile_ranges_to_display = [
        (0, 70),
        (70, 90),
        (90, 99),
        (99, 100)
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

    freq_male = []
    freq_female = []
    for start, end in percentile_ranges_to_display:
        freq_male.append(len(male_docs[(male_docs.m_percentile >= start) &
                          (male_docs.m_percentile < end)]) / len(male_docs) * 100)
        freq_female.append(len(female_docs[(female_docs.m_percentile >= start) &
                          (female_docs.m_percentile < end)]) / len(female_docs) * 100)


    imbalance_1p = freq_female[-1] / freq_male[-1]
    print(selection_column, selection_name, imbalance_1p)

        #
        # freq_male.append(percentile_freqs_male[start: end].m_percentile.sum() * 100)
        # freq_female.append(percentile_freqs_female[start: end].m_percentile.sum() * 100)

    # freq_male = ((male_docs.m_percentile.value_counts(normalize=True).reset_index().sort_values(by='index').m_percentile) * 100).tolist()
    # freq_female = ((female_docs.m_percentile.value_counts(normalize=True).reset_index().sort_values(by='index').m_percentile) * 100).tolist()


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
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(nrows=number_gp + 1,
                           ncols=5,
                           figure=fig,
                           width_ratios=[0.5, 0.5, 0.5, 3, 1],
                           height_ratios=[0.5] + [1] * number_gp,
                           wspace=0.2, hspace=0.05
                           )
    # ax = [None] * (number_gp + 1)
    # features = list(range(0, 100, 10))

    ax_percentile = fig.add_subplot(gs[0, 0])
    ax_percentile.text(0, 0.2, f'Percentile', fontsize=15)
    ax_min = fig.add_subplot(gs[0, 1])

    if selection_column.startswith('topic.') or selection_column.startswith('gen_approach'):
        indicator = 'Weight'
    else:
        indicator = 'Frequency'

    ax_min.text(0, 0.2, f'Min {indicator}', fontsize=15)
    ax_max = fig.add_subplot(gs[0, 2])
    ax_max.text(0, 0.2, f'Max {indicator}', fontsize=15)

    for a in [ax_percentile, ax_min, ax_max]:
        a.get_yaxis().set_visible(False)
        a.get_xaxis().set_visible(False)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        # a.spines["bottom"].set_visible(False)
        a.spines["left"].set_visible(False)


    # Create a figure, partition the figure into 7*2 boxes, set up an ax array to store axes objects, and create a list of age group names.
    for idx, (start, end) in enumerate(percentile_ranges_to_display):

        ax_percentile = fig.add_subplot(gs[idx + 1, 0])
        ax_percentile.text(0, 0, f'{start}-{end}', fontsize=15)
        min_weight = round(df[selection_column].quantile(start / 100), 6) * 100
        max_weight = round(df[selection_column].quantile(end / 100), 6) * 100


        ax_min = fig.add_subplot(gs[idx + 1, 1])
        ax_min.text(0, 0, '{:.4f}%'.format(min_weight), fontsize=15)
        ax_max = fig.add_subplot(gs[idx + 1, 2])
        ax_max.text(0, 0, '{:.4f}%'.format(max_weight), fontsize=15)

        for a in [ax_percentile, ax_min, ax_max]:
            a.get_yaxis().set_visible(False)
            a.get_xaxis().set_visible(False)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            a.spines["left"].set_visible(False)

        ax = fig.add_subplot(gs[idx + 1, 3])

        ax.set_xlim(1951, 2015)
        ax.set_yticks([])

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.spines['bottom'].set_edgecolor('#444444')
        ax.spines['bottom'].set_linewidth(2)
        #
        # # settings for x, y, varname
        # title = f'Percentile: {start}-{end}'
        # ax.text(1950, 0.0, title, fontsize=15)

        clip = (1951, 2015)

        sns.kdeplot(data=df[
            (df.m_author_genders == 'male') & (df.m_percentile >= start) & (df.m_percentile < end)
            ].m_year, ax=ax, shade=True, color="blue", legend=False, bw=.1, clip=clip)

#        ax[i].set(ylim=(0, 0.1))
        sns.kdeplot(data=df[
            (df.m_author_genders == 'female') & (df.m_percentile >= start) & (df.m_percentile < end)
            ].m_year,
                    ax=ax, shade=True, color="red", legend=False, bw=.1, clip=clip)

        # scale plots
        # kdeplots by default take up the maximum range, i.e. male and female plots always show the
        # same max value
        scale_male = freq_male[idx] / (freq_male[idx] + freq_female[idx] + 0.00000001)
        scale_female = freq_female[idx] / (freq_male[idx] + freq_female[idx] + 0.00000001)

        # line 1 -> male, line 2 -> female

        try:
            if freq_female[idx] > 0:
                ax.lines[1].set_ydata(ax.lines[1].get_ydata() * scale_female)
                female_collection = ax.collections[1].get_paths()
                female_collection[0].vertices[:, 1] *= scale_female
            if freq_male[idx] > 0:
                ax.lines[0].set_ydata(ax.lines[0].get_ydata() * scale_male)
                # Collections, i.e. the patches under the lines
                male_collection = ax.collections[0].get_paths()
                male_collection[0].vertices[:, 1] *= scale_male
        except IndexError:
            pass

        if idx < (number_gp - 1):
            ax.set_xticks([])


    # ax[.legend(['Male', 'Female'], facecolor='w')

    # adding legends on the top axes object
    bar_height = 3
    for idx in range(len((percentile_ranges_to_display))):
        ax = fig.add_subplot(gs[idx + 1, 4])

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



    dataset = JournalsDataset(use_equal_samples_dataset=True)

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

def get_1percent_ratios():

    dataset = JournalsDataset()
    # dataset.filter(start_year=1980)
    df = dataset.df
    results = {}
    for column in df.columns:
        if column.startswith('topic') or column.startswith('gen_a'):
            top1p = df[df[column] >= df[column].quantile(0.99)]
            male = len(top1p[top1p.m_author_genders == 'male'])
            female = len(top1p[top1p.m_author_genders == 'female'])

            if column.startswith('topic'):
                name = dataset.topics[int(column[6:])]['name']
            else:
                name = column

            print(column, name, female/male, male, female)

            results[name] = female / male * 7506 / 2016

    for x in sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(x)

def get_percentile_data(topic_id):

    d = JournalsDataset()
    men = d.copy().filter(author_gender='male')
    women = d.copy().filter(author_gender='female')
    years = [i for i in range(d.start_year, d.end_year + 1)]
    topic_weight_ranges = [
        (0, 0.001),
        (0.001, 0.01),
        (0.01, 0.1),
        (0.1, 1)
    ]

    output_df = pd.DataFrame()

    for weight_min, weight_max in topic_weight_ranges:
        data_men = []
        data_women = []
        for year in years:
            year_men = men.copy().filter(start_year=year, end_year=year)
            articles_men = len(year_men)
            year_women = women.copy().filter(start_year=year, end_year=year)
            articles_women = len(year_women)

            men_articles_in_weight = len(year_men.topic_score_filter(topic_id=topic_id,
                                     min_topic_weight=weight_min, max_topic_weight=weight_max))
            women_articles_in_weight = len(year_women.topic_score_filter(topic_id=topic_id,
                                     min_topic_weight=weight_min, max_topic_weight=weight_max))

            data_men.append(men_articles_in_weight / articles_men)
            data_women.append(women_articles_in_weight / articles_women)

        data_men_rolling = pd.DataFrame(data_men).rolling(center=True,
                                                          window=5).mean()[0].tolist()[2:-5]
        data_women_rolling =  pd.DataFrame(data_women).rolling(center=True,
                                                           window=5).mean()[0].tolist()[2:-5]

        output_df[f'men_{weight_min}-{weight_max}'] = data_men_rolling
        output_df[f'women_{weight_min}-{weight_max}'] = data_women_rolling

    output_df['years'] = years[2:-5]
    topic_name = d.topics[topic_id]['name'].replace(' ', '_')
    output_df.to_csv(Path(BASE_PATH, 'visualizations', 'plotly_data', f'{topic_name}_percentiles.csv'))

if __name__ == '__main__':

    get_percentile_data(71)


    # get_1percent_ratios()
    # plot_all_topics_and_general_approaches()


    # dataset = JournalsDataset(use_equal_samples_dataset=True)
    # # dataset.filter(start_year=2000)
    #
    # dataset = JournalsDataset()

    '''
    # post 2000
    1534 articles by men, 1040 mention womaen   68%
    751 by women, 633 mention womaen            84%
    
    # before 1970
    1633 by men, 526 mention womaen             32%
    140 by women, 63 mention womaen             45%
    


    '''
    dataset = JournalsDataset()
    # dataset.get_vocabulary_and_document_term_matrix(max_features=10000, use_frequencies=True,
    #                                                 store_in_df=True)
    # dataset.df['womaen'] = dataset.df['women'] + dataset.df['woman']

    # embed()


    # topic_percentile_plot(dataset=dataset, selection_column="gender",
    #                       selection_name='term: gender',
    #                       filename='term_gender.png')
    topic_percentile_plot(dataset=dataset, selection_column="topic.71",
                          selection_name='Women and Gender',
                          filename='61_percentiles.png')
    # topic_percentile_plot(dataset=dataset, selection_column="gender",
    #                       selection_name='term: gender',
    #                       filename='term_gender.png')
    # topic_percentile_plot(dataset=dataset, selection_column="womaen",
    #                       selection_name='terms: woman and women',
    #                       filename='term_womaen_post2000.png')
    # topic_percentile_plot(dataset=dataset, selection_column="topic.61",
    #                       selection_name='Topic Gender and Feminism',
    #                       filename='61_gender.png')
    # topic_percentile_plot(dataset=dataset, selection_column="gen_approach_Women's and Gender History",
    #                       selection_name="General Approach Women's and Gender History",
    #                       filename='gen_approach_gender.png')


    # plot_all_topics_and_general_approaches()
