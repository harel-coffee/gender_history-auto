from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.datasets.dataset import Dataset
from gender_history.utilities import BASE_PATH

from gender_history.divergence_analysis.stats import StatisticalAnalysis

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors

import pandas as pd
import pickle


import numpy as np
from IPython import embed
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict

import json

import networkx as nx



def get_topic_data():

    topic_data_path = Path(BASE_PATH, 'data', 'react_data', 'topic_data.json')

    if topic_data_path.exists():
        with open(topic_data_path, 'r') as infile:
            return json.load(infile)

    else:

        topic_data = {}

        d = JournalsDataset()
        c1 = d.copy().filter(author_gender='male')
        c2 = d.copy().filter(author_gender='female')

        vocabulary = DivergenceAnalysis.get_default_vocabulary()
        default_dtm, _ = d.get_vocabulary_and_document_term_matrix(vocabulary=vocabulary)

        div = DivergenceAnalysis(
            master_corpus=d,
            sub_corpus1=c1, sub_corpus2=c2,
            analysis_type='topics'
        )
        div_df = div.run_divergence_analysis(print_results=False)

        for topic_id in range(1, 91):

            print("\n\nTopic", topic_id)

            topic_name = d.topics[topic_id]['name']
            if topic_name.startswith('Noise'):
                continue

            div_row = div_df[div_df.topic_id == topic_id].iloc[0]

            embed()


            c1t = d.copy().topic_score_filter(topic_id=topic_id, min_percentile_score=95)

            topic_data[topic_id] = {
                'name': topic_name,
                'topic_id': topic_id,
                'dunning': div_row.dunning,
                'freq_score': div_row.frequency_score,
                'frequency': div_row['freq both'],

                'terms_prob': d.topics[topic_id]['terms_prob'][:10],
                'terms_frex': d.topics[topic_id]['terms_frex'][:10],
                'terms_dunning': get_distinctive_terms_for_intersection(c1t, default_dtm)
            }

        print("\n\n\nFinished with topics")

        with open(topic_data_path, 'w') as outfile:
            json.dump(topic_data, outfile, indent=4)

        return get_topic_data()


def get_edge_data():



    topic_intersections_path = Path(BASE_PATH, 'data', 'react_data', 'topic_edges.json')
    if topic_intersections_path.exists():
        with open(topic_intersections_path, 'r') as infile:
            topic_intersections = json.load(infile)
    else:

        d = JournalsDataset()
        vocabulary = DivergenceAnalysis.get_default_vocabulary()
        default_dtm, _ = d.get_vocabulary_and_document_term_matrix(vocabulary=vocabulary)


        c1 = d.copy().filter(author_gender='male')
        c2 = d.copy().filter(author_gender='female')
        topic_dtm = d.get_document_topic_matrix()

        c1_topic_dtm = c1.get_document_topic_matrix()
        c2_topic_dtm = c2.get_document_topic_matrix()
        stat = StatisticalAnalysis(topic_dtm, c1_topic_dtm, c2_topic_dtm,
                                   vocabulary=[f'topic.{i}' for i in range(1, 91)])
        correlations = stat.correlation_coefficient(return_correlation_matrix=True)

        # adjust female counts by 4.72
        female_adj = 1 / (len(c2) / (len(c1) + len(c2)))

        topic_intersections = defaultdict()

        for topic1_id in range(1, 91):
            topic_intersections[topic1_id] = defaultdict()

            print(topic1_id)

            for topic2_id in range(1, 91):
                print(topic1_id, topic2_id)

                if topic1_id == topic2_id: continue

                topic1_name = d.topics[topic1_id]['name']
                topic2_name = d.topics[topic2_id]['name']
                if topic1_name.startswith('Noise') or topic2_name.startswith('Noise'):
                    continue

                intersection = d.copy()\
                    .topic_score_filter(topic_id=topic1_id, min_percentile_score=90)\
                    .topic_score_filter(topic_id=topic2_id, min_percentile_score=90)

                count_male = len(intersection.df[intersection.df.m_author_genders == 'male'])
                count_female = len(intersection.df[intersection.df.m_author_genders == 'female'])
                adj_count_female = count_female * female_adj

                topic_intersections[topic1_id][topic2_id] = {
                    'number': len(intersection),
                    'gender_balance': adj_count_female / (count_male + adj_count_female),
                    'correlation': correlations[topic1_id - 1, topic2_id - 1],
                    'distinctive_terms': get_distinctive_terms_for_intersection(intersection,
                                                                                default_dtm)
                }


        with open(topic_intersections_path, 'w') as outfile:
            json.dump(topic_intersections, outfile, indent=4)

        return get_edge_data()

    all_intersections = []
    for i in range(1, 91):
        for j in range(1, 91):
            try:
                all_intersections.append(topic_intersections[str(i)][str(j)]['correlation'])
            except KeyError:
                pass

    percentile_99 = np.percentile(all_intersections, 99)
    percentile_95 = np.percentile(all_intersections, 95)

    print(f"99%: {percentile_99}. 95%: {percentile_95}")

    return topic_intersections


def get_distinctive_terms_for_intersection(intersection, default_dtm):

    vocabulary = DivergenceAnalysis.get_default_vocabulary()
    intersection_dtm, _ = intersection.get_vocabulary_and_document_term_matrix(
        vocabulary=vocabulary)

    s = StatisticalAnalysis(None, intersection_dtm, default_dtm, vocabulary,
                            skip_tf_transform=True)
    dunnings, _ = s.dunning_log_likelihood()

    distinctive_terms = []
    for idx in np.argsort(dunnings)[::-1][:10]:
        distinctive_terms.append(vocabulary[idx])

    return distinctive_terms


def get_distinctive_terms_for_correlated_topics(topic_id, correlated_topics_list):

    d = JournalsDataset()
    d.topic_score_filter(topic_id=topic_id, min_percentile_score=95)

    for cor_topic_id in correlated_topics_list:
        c = d.copy().topic_score_filter(topic_id=cor_topic_id, min_percentile_score=95)
        print(cor_topic_id, len(c.df))

        div = DivergenceAnalysis(d, d, c, analysis_type='terms')
        div.run_divergence_analysis()
        # div.print_articles_for_top_topics(top_terms_or_topics=5)




if __name__ == '__main__':

    get_distinctive_terms_for_correlated_topics(71, [61, 41, 46, 32, 45])

    #    get_topic_data()
    # get_edge_data()
