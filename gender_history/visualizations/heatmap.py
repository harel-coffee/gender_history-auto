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



def draw_heatmap():


    dataset = JournalsDataset()
    dataset.filter(author_gender='male')


    print(len(dataset))

    topic_selector = [f'topic.{i}' for i in range(1, 91)]
    topic_df = dataset.df[topic_selector]
    topic_id_to_name = {f'topic.{i}': dataset.topics[i]['name'] for i in range(1, 91)}
    topic_df = topic_df.rename(columns=topic_id_to_name)

    correlations = topic_df.corr()

    for i in range(90):
        correlations.iat[i, i] = 0.0

    sns.clustermap(correlations,
                   figsize=(20, 20),
                   row_cluster=True, col_cluster=True,
                   cmap='vlag',
                   vmin = -0.25, vmax = 0.25,

                   method='ward',

                   xticklabels=[dataset.topics[i]['name'] for i in range(1, 91)],
                   yticklabels=[dataset.topics[i]['name'] for i in range(1, 91)]


                   )
    plt.show()



if __name__ == '__main__':
    draw_heatmap()