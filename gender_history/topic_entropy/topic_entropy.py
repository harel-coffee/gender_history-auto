from pathlib import Path
from typing import TypedDict

import numpy as np
from pandas import DataFrame
from scipy.stats import entropy

from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.utilities import BASE_PATH


class AnnualData(TypedDict):
    year: int
    percentage_women: float
    entropy_of_topic_distribution: float
    number_of_articles: int


def calculate_entropy_for_all_years():
    """Store entropy of topic distribution, % female authored journals, and number of
    articles in the journal dataset for all years as a csv.
    """

    jd = JournalsDataset()
    entropy_data = []
    for year in range(jd.start_year, jd.end_year + 1):
        print(year)
        entropy_data.append(calculate_entropy_data_for_year(year=year))

    output_path = Path(BASE_PATH, 'gender_history', 'topic_entropy', 'topic_entropy_data.csv')
    DataFrame(entropy_data).to_csv(output_path)


def calculate_entropy_data_for_year(year: int) -> AnnualData:
    """Return entropy of topic distribution, % female authored journals, and number
     of articles for one year.
     """

    jd = JournalsDataset()
    jd.filter(start_year=year, end_year=year)

    dtm = jd.get_document_topic_matrix()
    # calculate mean topic weights for the year
    topic_weight_means = np.asarray(dtm.mean(axis=0))[0]
    entropy_of_topic_distribution = entropy(topic_weight_means)

    male_authored = len(jd.copy().filter(author_gender='male'))
    female_authored = len(jd.copy().filter(author_gender='female'))

    return {
        'year': year,
        'percentage_women': female_authored / (female_authored + male_authored),
        'entropy_of_topic_distribution': entropy_of_topic_distribution,
        'number_of_articles': male_authored + female_authored
    }


if __name__ == "__main__":
    calculate_entropy_for_all_years()