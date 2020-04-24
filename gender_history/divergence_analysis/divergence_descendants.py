from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis

from gender_history.datasets.dataset import Dataset
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from scipy.stats import fisher_exact

from IPython import embed

def male_female_descendant_numbers_by_decade():

    for start_year in [1980, 1990, 2000]:
        d = DissertationDataset()
        d.filter(start_year=start_year, end_year=start_year + 9)
        df = d.df

        male_desc = len(df[(df.m_author_genders == 'male') & (df.m_descendants > 0)])
        male_nodesc = len(df[(df.m_author_genders == 'male') & (df.m_descendants == 0)])
        female_desc = len(df[(df.m_author_genders == 'female') & (df.m_descendants > 0)])
        female_nodesc = len(df[(df.m_author_genders == 'female') & (df.m_descendants == 0)])

        _, fisher_p = fisher_exact([[male_desc, male_nodesc], [female_desc, female_nodesc]])

        print(f'\n{start_year}s')
        print('Men:   {:5d} theses, {:4d} with descendants. {:2.2f}%'.format(
            male_desc + male_nodesc, male_desc, male_desc / (male_desc + male_nodesc) * 100
        ))
        print('Women: {:5d} theses, {:4d} with descendants. {:2.2f}%'.format(
            female_desc + female_nodesc, female_desc, female_desc / (female_desc + female_nodesc) * 100
        ))
        print('Fisher\'s Exact Test p-value: {:0.3f}'.format(fisher_p))


def topics_and_descendants():

    d = DissertationDataset()
    d.filter(start_year=1980, end_year=1999)

    c1 = d.copy().filter(has_descendants=True)
    c2 = d.copy().filter(has_descendants=False)

    div = DivergenceAnalysis(d, c1, c2,
                             sub_corpus1_name='has descendants',
                             sub_corpus2_name='no descendants',
                             analysis_type='topics', sort_by='dunning')
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)

def topics_and_descendants_male():

    d = DissertationDataset()
    d.filter(start_year=1980, end_year=1999)
    d.filter(author_gender='male')

    c1 = d.copy().filter(has_descendants=True)
    c2 = d.copy().filter(has_descendants=False)

    div = DivergenceAnalysis(d, c1, c2,
                             sub_corpus1_name='has descendants',
                             sub_corpus2_name='no descendants',
                             analysis_type='topics', sort_by='dunning')
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)


def topics_and_descendants_female():

    d = DissertationDataset()
    d.filter(start_year=1980, end_year=1999)
    d.filter(author_gender='female')

    c1 = d.copy().filter(has_descendants=True)
    c2 = d.copy().filter(has_descendants=False)

    div = DivergenceAnalysis(d, c1, c2,
                             sub_corpus1_name='has descendants',
                             sub_corpus2_name='no descendants',
                             analysis_type='topics', sort_by='dunning')
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)




if __name__ == '__main__':
    topics_and_descendants()
    # male_female_descendant_numbers_by_decade()