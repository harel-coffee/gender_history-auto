import sqlite3
from pathlib import Path
from IPython import embed
import pandas as pd
from nameparser import HumanName

from gender_inference import guess_gender_census, guess_gender_first_name_only_usa, \
    guess_gender_with_middle_name_and_international_names

from collections import defaultdict, Counter


GENERAL_JOURNALS = {
    'The Journal of American History',
    'The American Historical Review',
    'Comparative Studies in Society and History',
    'Journal of Social History',
    'The Journal of Modern History',
    'History and Theory',
    'The Journal of Interdisciplinary History',
    'Ethnohistory',
    'The Mississippi Valley Historical Review',
    'Journal of World History',
    'Reviews in American History'
}

def generate_journal_csv():
    """
    Uses the JSTOR_full_cleaned database to generate a csv that lists the number of research articles
    by journal and decade from 1940 to the present, english only.

    :return:
    """

    journal_data = defaultdict(defaultdict)

    db = sqlite3.connect(Path('data', 'JSTOR_full_cleaned.db'))
    cur = db.cursor()
    cur.execute('''select journal, year from article_pub_info 
                      where article_type="research article" and year > 1940 and language="eng";''')
    while True:
        row = cur.fetchone()
        if not row:
            break

        journal = row[0]
        # 1988 -> 1980
        decade = int(row[1]) // 10 * 10

        try:
            journal_data[journal][decade] += 1
        except KeyError:
            journal_data[journal][decade] = 1

    df = pd.DataFrame(journal_data).fillna(0)
    df = df.sort_index().transpose()
    df['total'] = df[1940] + df[1950] + df[1960] + df[1970] + df[1980] + df[1990] + df[2000] + df[2010]
    df = df.sort_values(by='total', ascending=False)

    df.to_csv(Path('data', 'journals_by_decade.csv'))


def generate_journal_dataset(dataset_type='AHR'):

    db = sqlite3.connect(Path('data', 'JSTOR_full_cleaned.db'))
    cur = db.cursor()
    cur2 = db.cursor()

    journal = 'The American Historical Review'

    cur.execute(f'''select ID_doi, ID_jstor, article_type, pages, title, language, year, volume, issue, journal 
                        FROM article_pub_info
                        WHERE article_type="research article" and year > 1940 and language="eng"
                            and journal="{journal}";''')
    articles = cur.fetchall()
    for ID_doi, ID_jstor, article_type, pages, title, language, year, volume, issue, journal in articles:

        author_names, author_genders = get_author_info(ID_doi, ID_jstor)

        embed()


def get_article_text(ID_doi, ID_jstor):

def get_author_info(ID_doi, ID_jstor):
    """
    Gets authors by doi or jstor id.
    Returns the names of all authors as a joined string as well as the overall author genders
    overall author gender will be male, female, mixed, unknown


    :param ID_doi:
    :param ID_jstor:
    :return:

    >>> get_author_info('10.2307_1857439', None)
    ('Walter Goffart', 'male')

    """

    db = sqlite3.connect(str(Path('data', 'JSTOR_full_cleaned.db')))
    cur = db.cursor()

    if ID_doi:
        cur.execute(f'SELECT name, surname, role FROM contributors WHERE ID_doi = "{ID_doi}"')
    elif ID_jstor:
        cur.execute(f'SELECT name, surname, role FROM contributors WHERE ID_jstor = "{ID_jstor}"')
    else:
        raise ValueError(f"NO id for doi: {ID_doi}, jstor id: {ID_jstor}.")


    genders = set()
    names = []
    for first_name, last_name, role in cur.fetchall():

        human_name = HumanName(f'{last_name}, {first_name}')
        gender_census = guess_gender_census(human_name)
        gender_guess = guess_gender_with_middle_name_and_international_names(human_name)

        names.append(f'{first_name} {last_name}')

        if gender_census == gender_guess and (gender_census == 'male' or gender_census == 'female'):
            genders.add(gender_census)
        else:
            genders.add('unknown')

    if 'unknown' in genders:
        combined_gender = 'unknown'
    elif 'male' in genders and 'female' in genders:
        combined_gender = 'mixed'
    elif genders == {'male'}:
        combined_gender = 'male'
    elif genders == {'female'}:
        combined_gender = 'female'
    else:
        raise ValueError("How did you get here?", names, genders)
    
    combined_names = "; ".join(names)
    
    return combined_names, combined_gender
    


def get_names_and_genders_from_journals():


    authors_counter = Counter()
    journalc = Counter()

    db = sqlite3.connect(Path('data', 'JSTOR_full_cleaned.db'))
    cur = db.cursor()
    cur2 = db.cursor()
    cur.execute('''select journal, ID_doi, ID_jstor from article_pub_info 
                      where article_type="research article" and year > 1950 and language="eng";''')

    rows = cur.fetchall()

    for article_id, (journal, ID_doi, ID_jstor) in enumerate(rows):
        print(article_id, len(rows))
        if journal in GENERAL_JOURNALS:
            journalc[journal] += 1

            if ID_doi:
                cur2.execute(f'SELECT name, surname, role FROM contributors WHERE ID_doi = "{ID_doi}"')
            elif ID_jstor:
                cur2.execute(f'SELECT name, surname, role FROM contributors WHERE ID_jstor = "{ID_jstor}"')
            else:
                raise ValueError("NO id for ", journal)

            article_authors = cur2.fetchall()
            for first_name, last_name, role in article_authors:
                # some last names contain commas, which trip up the gender guesser
                last_name = last_name.strip(',')
                authors_counter[(first_name, last_name)] += 1

    authors = []
    for author in authors_counter:
        first_name ,last_name = author
        human_name = HumanName(f'{last_name}, {first_name}')


        guess_census = guess_gender_census(human_name)
        guess_first_name_usa = guess_gender_first_name_only_usa(human_name)
        guess_first_middle_name_international = guess_gender_with_middle_name_and_international_names(human_name)

        human_check_necessary = True
        if (guess_census == guess_first_middle_name_international and
                (guess_census == 'male' or guess_census == 'female')):
            human_check_necessary = False

        authors.append({
            'first_name': first_name,
            'last_name': last_name,
            'count': authors_counter[author],
            'prob_male_census': guess_gender_census(human_name, return_type='probability_male'),
            'guess_census': guess_census,
            # 'guess_first_name_usa': guess_first_name_usa,
            'guess_first_middle_name_international': guess_first_middle_name_international,
            'human_check_necessary': human_check_necessary
        })

    df = pd.DataFrame(authors)
    df.to_csv(Path('data', 'ambiguous_author_gender.csv'), encoding='utf8')

    embed()



if __name__ == '__main__':
#    get_names_and_genders_from_journals()
    generate_journal_dataset()