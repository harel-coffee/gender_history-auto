from pathlib import Path
from typing import TypedDict

import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.utilities import BASE_PATH


class AnnualData(TypedDict):
    year: int
    percentage_women: float
    entropy_of_topic_distribution: float
    jensen_shannon_men_women: float
    kullback_leibler_men_women: float
    kullback_leibler_women_men: float
    number_of_articles: int


def calculate_entropy_for_all_years():
    """Store the entropy of topic distribution, Jensen-Shannon distance between men and
    women authored articles, percentage of articles authored by women, and number of articles
    for all years as a csv.
    """

    jd = JournalsDataset()
    entropy_data = []
    for year in range(jd.start_year + 2, jd.end_year - 2):
        print(year)
        entropy_data.append(calculate_entropy_data_for_year(year=year))

    output_path = Path(
        BASE_PATH, 'gender_history', 'topic_entropy', 'topic_entropy_data_with_kl_divergence.csv'
    )
    DataFrame(entropy_data).to_csv(output_path)


def calculate_entropy_data_for_year(year: int) -> AnnualData:
    """Return entropy of topic distribution, Jensen-Shannon distance between men and women authored
     articles, percentage of articles authored by women, and number of articles for one year.
     """

    jd = JournalsDataset()
    jd.filter(start_year=year - 2, end_year=year + 2)

    men_authored_articles = jd.copy().filter(author_gender='male')
    women_authored_articles = jd.copy().filter(author_gender='female')
    num_articles = len(men_authored_articles) + len(women_authored_articles)

    dtm_men = men_authored_articles.get_document_topic_matrix()
    dtm_women = women_authored_articles.get_document_topic_matrix()
    dtm_both = jd.get_document_topic_matrix()

    topic_weight_means_men = np.asarray(dtm_men.mean(axis=0))[0]
    topic_weight_means_women = np.asarray(dtm_women.mean(axis=0))[0]
    topic_weight_means_both = np.asarray(dtm_both.mean(axis=0))[0]

    return {
        'year': year,
        'number_of_articles': num_articles,
        'percentage_women': len(women_authored_articles) / num_articles,
        'jensen_shannon_men_women': jensenshannon(topic_weight_means_men, topic_weight_means_women),

        # With only pk, scipy calculates entropy as `S = -sum(pk * log(pk), axis=0)`
        'entropy_of_topic_distribution': entropy(pk=topic_weight_means_both),
        # If qk is provided, scipy calculates the Kullback-Leibler divergence:
        # `S = sum(pk * log(pk / qk), axis=0)`
        'kullback_leibler_men_women': entropy(pk=topic_weight_means_men, qk=topic_weight_means_women),
        'kullback_leibler_women_men': entropy(pk=topic_weight_means_women, qk=topic_weight_means_men),
    }


if __name__ == "__main__":
    calculate_entropy_for_all_years()