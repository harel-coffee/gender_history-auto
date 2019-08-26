import networkx as nx
from stats import StatisticalAnalysis
from dataset import Dataset
from topics import TOPICS
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from IPython import embed


def generate_topic_network():
    """
    Generates a gexf network graph to be visualized with gephi


    :return:
    """



    d = Dataset()
    vocabulary = [f'topic.{i}' for i in range(1, 71)]
    d_dtm = d.get_document_topic_matrix()
    male = d.copy().filter(author_gender='male')
    female = d.copy().filter(author_gender='female')
    c1_dtm = female.get_document_topic_matrix() * 300
    c2_dtm = male.get_document_topic_matrix() * 300
    s = StatisticalAnalysis(d_dtm, c1_dtm, c2_dtm, vocabulary)
    correlated_terms = s.correlation_coefficient(return_correlation_matrix=True)

    dunning, _ = s.dunning_log_likelihood()
    frequency_score, _ = s.frequency_score()
    mwr, _ = s.mann_whitney_rho()

    G = nx.Graph()
    for i in range(70):
        topic_id = i + 1

        # skip all the generic topics (this will be repeated a number of times. It's bad coding
        # style, I know
        if topic_id in [53, 62, 69, 70]:
            continue
        topic = TOPICS[topic_id]['name']
        G.add_node(
            topic,
            id=topic_id,
            frequency=d_dtm[:, i].mean(),
            dunning=dunning[i],
            freq_score=frequency_score[i],
            mwr=mwr[i]
        )

    for i in range(70):
        topic_id = i + 1
        if topic_id in [53, 62, 69, 70]:
            continue

        topic = TOPICS[topic_id]['name']

        no_added = 0
        for correlated_i in np.array(correlated_terms[i])[0].argsort()[::-1][1:20]:

            if correlated_i + 1 in [53, 62, 69, 70]:
                continue

            topic2 = TOPICS[correlated_i + 1]['name']
            cor = correlated_terms[i, correlated_i]
            if cor > 0.20 or (no_added == 1 and cor > 0.10) or no_added == 0:
                G.add_edge(topic, topic2, weight=cor )
                no_added += 1

    nx.write_gexf(G, Path('data', 'networks', 'topics_dense.gexf'))

    embed()

def plotly_graph():
    """
    Create interactive plotly graph

    :return:
    """
    import plotly.graph_objects as go
    import chart_studio.plotly as py

    d = Dataset()
    vocabulary = [f'topic.{i}' for i in range(1, 71)]
    d_dtm = d.get_document_topic_matrix()
    male = d.copy().filter(author_gender='male')
    female = d.copy().filter(author_gender='female')
    c1_dtm = female.get_document_topic_matrix() * 300
    c2_dtm = male.get_document_topic_matrix() * 300
    s = StatisticalAnalysis(d_dtm, c1_dtm, c2_dtm, vocabulary)
    correlated_terms = s.correlation_coefficient(return_correlation_matrix=True)
    for i in range(70):
        correlated_terms[i, i] = 0


    l = [TOPICS[i]['name'] for i in range(1, 71)]

    # interactive viz with plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlated_terms,
        x=l,
        y=l))

    fig.show()

    # online with plotly
    data = go.Heatmap(
            z=correlated_terms,
            x=l,
            y=l)

    py.plot([data], filename='topic_cors')

if __name__ == '__main__':
#    generate_topic_network()
    plotly_graph()