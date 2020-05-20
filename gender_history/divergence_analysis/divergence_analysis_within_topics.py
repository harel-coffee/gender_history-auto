from gender_history.datasets.dataset import Dataset
from gender_history.datasets.dataset_journals import JournalsDataset
from gender_history.datasets.dataset_dissertation import DissertationDataset
from gender_history.divergence_analysis.stats import StatisticalAnalysis
from gender_history.divergence_analysis.divergence_analysis import DivergenceAnalysis



def overall_analyses():

    for dataset_name in ['journals', 'dissertations']:
        for analysis_type in ['topics', 'terms']:

            print('\n\n\n', dataset_name, analysis_type)

            if dataset_name == 'journals':
                d = JournalsDataset()
            else:
                d = DissertationDataset()

            # Create two sub-datasets, one for female authors and one for male authors
            c1 = d.copy().filter(author_gender='female')
            c2 = d.copy().filter(author_gender='male')

            # Run the divergence analysis
            div = DivergenceAnalysis(d, c1, c2, sub_corpus1_name='women', sub_corpus2_name='men',
                                     analysis_type=analysis_type, sort_by='dunning')
            div.run_divergence_analysis(number_of_terms_or_topics_to_print=12)

def analysis_military_history():

    for dataset_name in ['journals', 'dissertations']:
        for analysis_type in ['topics']:

            print('\n\n\n', dataset_name, analysis_type)

            if dataset_name == 'journals':
                d = JournalsDataset()
            else:
                d = DissertationDataset()

            if analysis_type == 'topics':
                compare_to_overall_weights = True
            else:
                compare_to_overall_weights = False

            # retain only the articles scoring in the top 5% for topic 31 (military history)
            d.topic_score_filter(26, min_percentile_score=95)

            # Create two sub-datasets, one for female authors and one for male authors
            c1 = d.copy().filter(author_gender='female')
            c2 = d.copy().filter(author_gender='male')

            div = DivergenceAnalysis(d, c1, c2, sub_corpus1_name='women', sub_corpus2_name='men',
                                     analysis_type=analysis_type, sort_by='dunning',
                                     compare_to_overall_weights=compare_to_overall_weights)
            div.run_divergence_analysis(number_of_terms_or_topics_to_print=10)

            div.print_articles_for_top_topics(top_terms_or_topics=10, articles_per_term_or_topic=5)

def analysis_term_gender():

    d = JournalsDataset()
    d.filter(term_filter={'term': 'gender', 'min_count': 10})
    c1 = d.copy().filter(author_gender='male')
    c2 = d.copy().filter(author_gender='female')

    print(len(c1), len(c2), len(d))

    # Run the divergence analysis
    div = DivergenceAnalysis(d, c1, c2, sub_corpus1_name='male', sub_corpus2_name='female',
                             analysis_type='terms', sort_by='frequency_score',
                             compare_to_overall_weights=False)
    div.run_divergence_analysis(number_of_terms_or_topics_to_print=100)

    div.print_articles_for_top_topics(top_terms_or_topics=10, articles_per_term_or_topic=5)



if __name__ == '__main__':
    analysis_term_gender()