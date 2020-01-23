
from pathlib import Path
import pandas as pd
from IPython import embed
import numpy as np
from tabulate import tabulate

from dataset import Dataset
from dataset_dissertation import DissertationDataset
from dataset_journals import JournalsDataset
from stats import StatisticalAnalysis

from scipy.sparse import vstack

def divergence_analysis(master_dataset:Dataset,
                        c1:Dataset,                 # sub corpus 1
                        c2:Dataset,                 # sub corpus 2
                        topics_or_terms='terms',
                        number_of_terms_to_print=30,
                        c1_name=None, c2_name=None,
                        print_results=True, sort_by=None,
                        min_appearances_per_term=50):
    if not c1_name:
        c1_name = c1.name
    if not c2_name:
        c2_name = c2.name

    if not sort_by:
        if topics_or_terms == 'topics':
            sort_by = 'frequency_score'
        else:
            sort_by = 'dunning'


    if topics_or_terms == 'terms':
        vocabulary = master_dataset.get_vocabulary(max_terms=10000,
                                                   min_appearances=min_appearances_per_term,
                                                   include_2grams=True)
        c1_dtm = c1.get_document_term_matrix(vocabulary=vocabulary)
        c2_dtm = c2.get_document_term_matrix(vocabulary=vocabulary)
    else:
        vocabulary = [f'X{i}' for i in range(1, 101)]
        c1_dtm = c1.get_document_topic_matrix() * 4000
        c2_dtm = c2.get_document_topic_matrix() * 4000

    master_dtm = vstack([c1_dtm, c2_dtm])


    s = StatisticalAnalysis(master_dtm, c1_dtm, c2_dtm, vocabulary)
    dunning, _ = s.dunning_log_likelihood()
    frequency_score, _ = s.frequency_score()
#    mwr, _ = s.mann_whitney_rho()
#    correlated_terms = s.correlation_coefficient()

    total_terms_all = master_dtm.sum()
    total_terms_c1 = c1_dtm.sum()
    total_terms_c2 = c2_dtm.sum()

    column_sums_all = np.array(master_dtm.sum(axis=0))[0]
    column_sums_c1 = np.array(c1_dtm.sum(axis=0))[0]
    column_sums_c2 = np.array(c2_dtm.sum(axis=0))[0]

    data = []
    for term_idx in range(len(vocabulary)):
        count_all = column_sums_all[term_idx]
        count_c1 = column_sums_c1[term_idx]
        count_c2 = column_sums_c2[term_idx]

        if topics_or_terms == 'terms':
            term = vocabulary[term_idx]
            if count_all < min_appearances_per_term:
                continue
        else:
            topic_idx = term_idx + 1
            term = f'({topic_idx}) {c1.topics[topic_idx]["name"]}'

        data.append({
            'term': term,
            'dunning': dunning[term_idx],
            'frequency_score': frequency_score[term_idx],
            'count_total': count_all,
            f'count {c1_name}': count_c1,
            f'count {c2_name}': count_c2,
            'frequency_total': count_all / total_terms_all,
            f'frequency {c1_name}': count_c1 / total_terms_c1,
            f'frequency {c2_name}': count_c2 / total_terms_c2,

#            'mwr': mwr[term_idx],
#            'correlated_terms': correlated_terms[vocabulary[term_idx]]
        })
    df = pd.DataFrame(data)


    df.sort_values(by=sort_by, inplace=True)
    df.reset_index(inplace=True)

    if print_results:

        if topics_or_terms == 'topics':
            headers = ['term', 'dunning', 'frequency_score', 'frequency_total',
                       f'frequency {c1_name}', f'frequency {c2_name}']
        else:
            headers = ['term', 'dunning', 'frequency_score', 'count_total',
                       f'count {c1_name}', f'count {c2_name}']

        year_df = {}

        for years_range in [(1976, 1984), (1985, 1989), (1990, 1994), (1995, 1999), (2000, 2004),
                             (2005, 2009), (2010, 2015)]:
            y1, y2 = years_range
            c1_count = len(c1.df[(c1.df['year'] >= y1) & (c1.df['year'] <= y2)])
            c2_count = len(c2.df[(c2.df['year'] >= y1) & (c2.df['year'] <= y2)])
            if c1_count > 0 or c2_count > 0:
                year_df[f'{y1}-{y2}'] = {
                    f'{c1_name}': c1_count,
                    f'{c1_name} freq': c1_count / len(c1),
                    f'{c2_name}': c2_count,
                    f'{c2_name} freq': c2_count / len(c2),
                }
        year_df = pd.DataFrame(year_df).transpose()
        print(tabulate(year_df, headers='keys'))


        print(f'\n\nTerms distinctive for Corpus 1: {c1_name}. {len(c1)} Theses\n')
        print(tabulate(df[headers][::-1][0:number_of_terms_to_print], headers='keys'))

        print(f'\n\nTerms distinctive for Corpus 2: {c2_name}. {len(c2)} Theses\n')
        print(tabulate(df[headers][0:number_of_terms_to_print], headers='keys'))


    return df

def wordcloud(gender='female', relative_scaling=0.0):

    # local imports so Pillow and wordclouds are not hard requirements for running any code
    from PIL import Image, ImageFont, ImageDraw
    from wordcloud import WordCloud

    def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

        color = icon.getpixel((int(position[1] + 2), int(position[0] + 2)))

        print(color)

        if color[0] == color[1] == color[2]: color = (0, 0, 0, 255)
        # if color[0] > 200: color = (220, 0, 0, 255)

        if (color[0] + color[1] + color[2]) > 230:
            print(word, color, font_size)
        return color

    if gender == 'female':
        icon_path = Path('data', 'plots', 'word_clouds', 'icon_female.png')
    else:
        icon_path = Path('data', 'plots', 'word_clouds', 'icon_male.png')

    icon = Image.open(icon_path)

    basewidth = 3000
    wpercent = (basewidth / float(icon.size[0]))
    hsize = int((float(icon.size[1]) * float(wpercent)))
    icon = icon.resize((basewidth, hsize))#, icon.ANTIALIAS)
    icon = icon.convert('RGBA')

    d = Dataset()
    c1 = d.copy().filter(author_gender='female')
    c2 = d.copy().filter(author_gender='male')
    data = divergence_analysis(d, c1, c2)

    mask = Image.new("RGBA", icon.size, (255, 255, 255))
    mask.paste(icon, icon)
    mask = np.array(mask)


    word_dict = {}
    for _, row in data.iterrows():
        dunning = row['dunning']

        if (gender == 'female' and dunning > 0):
            word_dict[row['term']] = dunning
        if gender == 'male' and dunning < 0:
            word_dict[row['term']] = -dunning

    print("Total tokens: {}".format(len(word_dict)))

    wc = WordCloud(background_color='white', max_font_size=300, mask=mask,
                   max_words=2000, relative_scaling=relative_scaling, min_font_size=4)
    wc.generate_from_frequencies(word_dict)
    wc.recolor(color_func=grey_color_func)


    wc.to_file(Path('data', 'plots', 'word_clouds', f'{gender}_{relative_scaling}.png'))





if __name__ == '__main__':
    # d = DivergenceAnalysis(
    #     c1_start_year=2000, c1_end_year=2010, c1_gender='both',
    #     c2_start_year=2010, c2_end_year=2020, c2_gender='both'
    # )
    # d.run_analysis(analysis_type='dunning')
    #

#    wordcloud()

    # d = Dataset()
    # c1 = d.copy().filter(institution_filter='princeton')
    # c2 = d.copy().filter(institution_filter='not_princeton')
    # divergence_analysis(d, c1, c2)



    # d = Dataset()
    # c1 = d.copy().filter(advisor_gender='male')
    # c2 = d.copy().filter(advisor_gender='female')
    # divergence_analysis(d, c1, c2)

    # d = Dataset()
    # d.filter(advisor_gender='female')
    # c1 = d.copy().filter(author_gender='female')
    # c2 = d.copy().filter(author_gender='male')
    # divergence_analysis(d, c1, c2, topics_or_terms='topics')

#     d = Dataset()
#     d.filter(author_gender='male', start_year=1976, end_year=1999)
#     c1 = d.copy().filter(advisor_gender='female')
#     c2 = d.copy().filter(advisor_gender='male')
# #    c1.normalize_dataset_by_5year_interval()
# #    c2.normalize_dataset_by_5year_interval()
#     divergence_analysis(d, c1, c2, topics_or_terms='terms',
#                         c1_name='female advisor', c2_name='male advisor')
#     divergence_analysis(d, c1, c2, topics_or_terms='topics',
#                         c1_name='female advisor', c2_name='male advisor')

    # Loads the entire dataset of dissertations

#    d.topic_percentile_score_filter(24, min_percentile_score=80)
#    d.filter(author_gender='male', start_year=2000, end_year=2015)
#    d.filter()
#     d = Dataset()
#     d.filter(start_year=1980, end_year=2004)
#
#     # Create two sub-datasets, one for female authors and one for male authors
#     c1 = d.copy().filter(has_descendants=True)
#     c2 = d.copy().filter(has_descendants=False)
#
#     # divergence_analysis(d, c1, c2, c1_name='female author', c2_name='male author',
#     #                     topics_or_terms='terms', sort_by='frequency_score',
#     #                     number_of_terms_to_print=80)
#
#     divergence_analysis(d, c1, c2, topics_or_terms='topics',
#                         c1_name='has descendants', c2_name='no descendants', sort_by='dunning')

    d = JournalsDataset()


    c1 = d.copy().filter(author_gender='male')
    c2 = d.copy().filter(author_gender='female')
    divergence_analysis(d, c1, c2, topics_or_terms='topics',
                        c1_name='male', c2_name='female', sort_by='frequency_score',
                        number_of_terms_to_print=50)




    # divergence_analysis(d, c1, c2)

    # networks generated with correlation
    # maybe better: use co-use of topics -> get topic-overlap matrix, i.e. to what degree are
    # topics co-used.
    # stemmer: integrated in STM R package.

