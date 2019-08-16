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

def plot_gender_development_over_time(
        no_terms_or_topics_to_show=8,
        data='topics',
        display_selector='most_frequent',
        selected_terms_or_topics=None,
        show_plot=True,
        store_to_filename=None):

    """

    :param no_terms_or_topics_to_show: int
    :param data: 'topics', 'terms', 'terms_of_topics'
    :param display_selector: 'most_frequent', 'most_divergent', 'most_variable'
    :param selected_terms_or_topics: topic_id or list of terms
    :param show_plot: bool
    :param store_to_filename: bool or str
    :return:
    """

    if data == 'terms_of_topic':
        if not isinstance(selected_terms_or_topics, int):
            raise ValueError("When displaying 'terms_of_topic', please pass a topic_id for param"
                             "selected_terms_or_topics")

    # 0: find terms or topics to display
    d = Dataset()
    if data == 'topics':
        selected_terms_or_topics = [f'topic.{id}' for id in range(1, 71)]
        title_name = 'topics'
    elif data == 'terms':
        vocab = []
        for t in selected_terms_or_topics:
            vocab.append(t)
        d.get_document_term_matrix(vocabulary=vocab, store_in_df=True)
        title_name = 'terms'
    elif data == 'terms_of_topic':
        vocab = []
        topic_id = selected_terms_or_topics
        for term in TOPICS[topic_id]['terms_prob']:
            if term in d.vocabulary:
                vocab.append(term)
        selected_terms_or_topics = vocab
        d.get_document_term_matrix(vocabulary=vocab, store_in_df=True)
        title_name = f'terms of topic {topic_id}'
    else:
        raise ValueError('"data" has to be "terms" "topics" or "terms_of_topic"')

    df = d.df

    # 1: Load data
    data = {}
    for t in selected_terms_or_topics:
        data[t] = defaultdict(list)
    min_freq_total = 1
    max_freq_total = 0

    print(selected_terms_or_topics)

    for idx, year in enumerate(range(1985, 2011)):
        time_slice = df[(df.ThesisYear >= year - 5) & (df.ThesisYear <= year + 5)]
        time_slice_female = time_slice[time_slice.AdviseeGender == 'female']
        time_slice_male = time_slice[time_slice.AdviseeGender == 'male']

        for t in selected_terms_or_topics:
            freq_total = time_slice[t].mean()
            freq_female = time_slice_female[t].mean()
            freq_male = time_slice_male[t].mean()

#            if t == 'gender' and year == 2008:
#                embed()

            # if a term doesn't appear, it is neutral
            if (freq_male + freq_female) == 0:
                freq_score = 0.5
            else:
                freq_score = freq_female / (freq_female + freq_male)


            data[t]['year'].append(year)
            data[t]['freq_score'].append(freq_score)
            data[t]['freq_total'].append(freq_total)

            if freq_total < min_freq_total:
                min_freq_total = freq_total
            if freq_total > max_freq_total:
                max_freq_total = freq_total

            data[t]['mean_freq_score'] = np.mean(data[t]['freq_score'])
            data[t]['mean_freq_total'] = np.mean(data[t]['freq_total'])
            data[t]['freq_score_range'] = max(data[t]['freq_score']) - min(data[t]['freq_score'])


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
    ax.set_ylim(0, 1)
    ax.set_xlim(1985, 2010)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both')

    dot_scaler = MinMaxScaler((0.0, 50.0))
    dot_scaler.fit(np.array([min_freq_total, max_freq_total]).reshape(-1, 1))
    legends = []


    def draw_line(t, t_data, df):
        """
        Draws one line depending on t (term or topic string) and t_data (dict of data belonging
        to t)

        :param t: str
        :param t_data: dict
        :return:
        """
        y = t_data['freq_score']
        x = t_data['year']
        frequencies = t_data['freq_total']
        if t.startswith('topic.'):
            legend = TOPICS[int(t[6:])]['name']
        else:
            legend = '{:10s} ({})'.format(t, df[t].sum())

        x_spline = np.linspace(min(x), max(x), (2010 - 1985 + 1) * 1000)
        spl = make_interp_spline(x, y, k=1)  # BSpline object
        y_spline = spl(x_spline)

        line_interpolater = interp1d(x, frequencies)
        line_widths = line_interpolater(x_spline)
        line_widths = dot_scaler.transform(line_widths.reshape(-1, 1)).flatten()

        try:
            color = sns.color_palette()[len(legends)]
        except IndexError:
            color = sns.cubehelix_palette(100, start=2, rot=0, dark=0, light=.95)[len(legends)]

        ax.scatter(x_spline, y_spline, s=line_widths, antialiased=True,
                   color=color)
        legends.append(mpatches.Patch(color=color, label=legend))

    # 3: Plot
    if display_selector == 'most_frequent':
        ax.set_title(f'Most frequent {title_name} for female (top) and male authors (bottom)',
                     weight='bold', fontsize=18)
        sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['mean_freq_total'], reverse=True)
        for t, t_data in sorted_items[:no_terms_or_topics_to_show]:
            draw_line(t, t_data, df)
    elif display_selector == 'most_divergent':
        ax.set_title(f'Most divergent {title_name} for female (top) and male authors (bottom)',
                     weight='bold', fontsize=18)
        sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['mean_freq_score'], reverse=True)
        no_disp = no_terms_or_topics_to_show // 2
        for t, t_data in sorted_items[:no_disp] + sorted_items[::-1][:no_disp]:
            draw_line(t, t_data, df)
    elif display_selector == 'most_variable':
        ax.set_title(f'Most variable {title_name} for female (top) and male authors (bottom)',
                     weight='bold', fontsize=18)
        # sort by mean_freq_range second to preserve colors between plots
        sorted_items = sorted(data.items(), key=lambda k_v: k_v[1]['freq_score_range'], reverse=True)
        sorted_items = sorted_items[:no_terms_or_topics_to_show]
        sorted_items = sorted(sorted_items, key=lambda k_v: k_v[1]['mean_freq_score'], reverse=True)
        for t, t_data in sorted_items:
            draw_line(t, t_data, df)

    else:
        raise ValueError('display_selector has to be most_frequent, most_variable, or most_divergent')

    ax.legend(handles=legends, loc=4)
    print(min_freq_total, max_freq_total)

    if show_plot:
        plt.show()
    if store_to_filename:
        fig.savefig(Path('data', store_to_filename))

if __name__ == '__main__':

    g = ['women', 'gender', 'female', 'woman', 'men', 'work', 'feminist', 'live', 'male', 'family', 'mother'] #, femin, movement, experi, suffrag, sister, sphere, social, domest, status
    g = ['gender', 'queer', 'gay', 'lesbian']

    g = ['topic.35', 'topic.23', 'topic.17', 'topic.47', 'topic.25']

    g = ['mother', 'sphere', 'sister', 'status', 'gender']

    d = Dataset()
    vocab = d.get_vocabulary(max_terms=200)

    plot_gender_development_over_time(
        no_terms_or_topics_to_show=8,
        data='terms_of_topic',
        display_selector='most_divergent',
        selected_terms_or_topics=28,
        show_plot=True,
        store_to_filename=None)