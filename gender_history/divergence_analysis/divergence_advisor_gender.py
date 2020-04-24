from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis

from gender_history.datasets.dataset import Dataset
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from scipy.stats import fisher_exact

from IPython import embed

def male_female_advisor_numbers():

    for start_year in [1980, 1990, 2000, 2010]:

        d = DissertationDataset()
        d.filter(start_year=start_year, end_year=start_year + 9)
        df = d.df

        male_students = df[(df.m_author_genders == 'male')]
        female_students = df[(df.m_author_genders == 'female')]

        male_maleadv = len(male_students[male_students.m_advisor_gender == 'male'])
        male_femaleadv = len(male_students[male_students.m_advisor_gender == 'female'])
        male_unkadv = len(male_students[male_students.m_advisor_gender == 'unknown'])


        female_maleadv = len(female_students[female_students.m_advisor_gender == 'male'])
        female_femaleadv = len(female_students[female_students.m_advisor_gender == 'female'])
        female_unkadv = len(female_students[female_students.m_advisor_gender == 'unknown'])


        print(f'\n{start_year}s')

        print('Men:   {:4d} male advisors, {:4d} female advisors. {:4d} unknown advisors. {:2.2f}% female advisors'.format(
            male_maleadv, male_femaleadv, male_unkadv, male_femaleadv / (male_maleadv + male_femaleadv) * 100
        ))
        print('Women: {:4d} male advisors, {:4d} female advisors. {:4d} unknown advisors. {:2.2f}% female advisors'.format(
            female_maleadv, female_femaleadv, female_unkadv, female_femaleadv / (female_maleadv + female_femaleadv) * 100
        ))


def topics_and_descendants_overall():

    d = DissertationDataset()
    d.filter(start_year=1990, end_year=2015)

    c1 = d.copy().filter(advisor_gender='female')
    c2 = d.copy().filter(advisor_gender='male')

    div = DivergenceAnalysis(d, c1, c2,
                             sub_corpus1_name='female advisor',
                             sub_corpus2_name='male advisor',
                             analysis_type='topics', sort_by='dunning')
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)


def topics_and_descendants_male_student():

    d = DissertationDataset()
    d.filter(start_year=1990, end_year=2015)
    d.filter(author_gender='male')

    c1 = d.copy().filter(advisor_gender='female')
    c2 = d.copy().filter(advisor_gender='male')

    div = DivergenceAnalysis(d, c1, c2,
                             sub_corpus1_name='man with female advisor',
                             sub_corpus2_name='man with male advisor',
                             analysis_type='topics', sort_by='dunning')
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)


def topics_and_descendants_female_student():

    d = DissertationDataset()
    d.filter(start_year=1990, end_year=2015)
    d.filter(author_gender='female')

    c1 = d.copy().filter(advisor_gender='female')
    c2 = d.copy().filter(advisor_gender='male')

    div = DivergenceAnalysis(d, c1, c2,
                             sub_corpus1_name='woman with female advisor',
                             sub_corpus2_name='woman with male advisor',
                             analysis_type='gen_approach', sort_by='dunning')
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)




if __name__ == '__main__':
    topics_and_descendants_female_student()
#    male_female_advisor_numbers()