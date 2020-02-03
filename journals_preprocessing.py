import sqlite3
from pathlib import Path
from IPython import embed
import pandas as pd
from nameparser import HumanName
import re

from name_to_gender import GenderGuesser

# from gender_inference import guess_gender_census, guess_gender_first_name_only_usa, \
#     guess_gender_with_middle_name_and_international_names, get_hand_coded_gender

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

def generate_journal_csv(item_type='journals'):
    """
    Uses the JSTOR_full_cleaned database to generate a csv that lists the number of research articles
    by journal and decade from 1940 to the present, english only.

    :return:
    """

    journal_data = defaultdict(defaultdict)

    db = sqlite3.connect(str(Path('data', 'JSTOR_full_cleaned.db')))
    cur = db.cursor()

    if item_type == 'journals':
        cur.execute('''select journal, year from article_pub_info 
                      where article_type="research article" and year > 1940 and language="eng";''')
    elif item_type == 'books':
        cur.execute('''SELECT publisher, year from book_pub_info 
                      WHERE year > 1940 AND language="eng";''')
    else:
        raise ValueError('item_type has to be books or journals')

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

    df.to_csv(Path('data', f'{item_type}_by_decade.csv'))


def generate_journal_dataset(dataset_type='general_journals', dataset_items='full_texts'):


    if not dataset_items in ['full_texts', 'sections']:
        raise ValueError("dataset type needs to be either full_texts or sections.")

    if dataset_type == 'AHR':
        journals = ['The American Historical Review']
    elif dataset_type == 'general_journals':
        journals = GENERAL_JOURNALS
    else:
        raise ValueError("dataset_type needs to be AHR or general_journals")

    gender_guesser = GenderGuesser()

    db = sqlite3.connect(str(Path('data', 'JSTOR_full_cleaned.db')))
    cur = db.cursor()

    articles_sql = []
    for journal in journals:
        cur.execute(f'''select ID_doi, ID_jstor, article_type, pages, title, language, year, volume, issue, journal 
                        FROM article_pub_info
                        WHERE article_type="research article" and year > 1950 and language="eng"
                            and journal="{journal}";''')
        articles_sql += cur.fetchall()

    count = 0
    articles_list = []
    for article in articles_sql:
        ID_doi, ID_jstor, article_type, pages, title, language, year, volume, issue, journal = article

        count += 1
        if count % 100 == 0:
            print(count)


        author_names, author_genders = get_author_info(ID_doi, ID_jstor, gender_guesser)
        try:
            text = get_article_text(ID_doi, ID_jstor)
        except FileNotFoundError:
            print("\nMissing:")
            print(title)
            print(ID_doi, ID_jstor)
            continue

        # either split text into sections or keep as one large chunk
        if dataset_items == 'full_texts':
            sections = [text]
        else:
            sections = split_text_into_chunks(text, chunk_length=300)

        for section in sections:
            articles_list.append({
                'ID_doi': str(ID_doi),
                'ID_jstor': str(ID_jstor),
                'article_type': article_type,
                'pages': pages,
                'title': title,
                'language': language,
                'year': year,
                'volume': volume,
                'issue': issue,
                'journal': journal,
                'authors': author_names,
                'author_genders': author_genders,
                'text': section
            })

    df = pd.DataFrame(articles_list)
    df.to_csv(Path('data', f'{dataset_type}_{dataset_items}.csv'), encoding='utf8')


def get_article_text(ID_doi, ID_jstor):


    full_data_path = Path('/', 'pcie', 'gender_history', 'ocr')
    repo_data_path = Path('data', 'journal_texts')

    if ID_doi:
        ocr_path_repo = Path(repo_data_path, f'journal-article-{ID_doi}.txt')
    else:
        ocr_path_repo = Path(repo_data_path, f'journal-article-10.2307_{ID_jstor}.txt')

    try:
        with open (ocr_path_repo) as f:
            return f.read()

    except FileNotFoundError:
        if ID_doi:
            ocr_path = Path(full_data_path, f'journal-article-{ID_doi}.txt')
        else:
            ocr_path = Path(full_data_path, f'journal-article-10.2307_{ID_jstor}.txt')

        with open(ocr_path) as f:
            text = f.read()
            # strip xml tags
            text = re.sub(r'<.{1,50}>', '', text)

        # if not ID_doi:
        #     print(text[:300], "\n\n")

        with open(ocr_path_repo, 'w') as out:
            out.write(text)

        return get_article_text(ID_doi, ID_jstor)

def split_text_into_chunks(text, chunk_length=300):
    """
    Splits text into chunks of equal word count with approximately chunk_length

    First, we figure out how many chunks we should split the text into, e.g. if we want
    chunk_length = 300 and have a text with length 2860, the closest we can get is
    10 chunks with length 292.
    (We could have 9 chunks with length 300 and one with length 160 but it's probably better if
    the chunks are also close in length)

    :param text:
    :param chunk_length:
    :return:
    """
    tokenized_text = text.split()
    number_of_chunks = round(len(tokenized_text) / chunk_length)

    # if text less than 150 words:
    if number_of_chunks == 0:
        print(text)
        return [text]

    chunk_length = round(len(tokenized_text) / number_of_chunks)
    chunks = [" ".join(tokenized_text[i:i+chunk_length]) for i in range(0, number_of_chunks * chunk_length, chunk_length)]
    if len(chunks[-1].split()) < 150:
        embed()

    return chunks


NAS = []
def get_author_info(ID_doi, ID_jstor, gender_guesser):
    """
    Gets authors by doi or jstor id.
    Returns the names of all authors as a joined string as well as the overall author genders
    overall author gender will be male, female, mixed, unknown


    :param ID_doi:
    :param ID_jstor:
    :return:

    >>> gg = GenderGuesser()
    >>> get_author_info('10.2307_1857439', None, gg)
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
        last_name = last_name.strip(',')

        human_name = HumanName(f'{last_name}, {first_name}')

        try:
            gen = gender_guesser.get_handcoded_gender_from_human_name(human_name)
        except:
            print("missing human name", human_name)
            gen = 'unknown'
        # gen = get_hand_coded_gender(human_name)
        genders.add(gen)
        if gen == 'n/a':
            NAS.append(f'{first_name} {last_name}')
#        gender_census = guess_gender_census(human_name)
#        gender_guess = guess_gender_with_middle_name_and_international_names(human_name)

        names.append(f'{first_name} {last_name}')

#        if gender_census == gender_guess and (gender_census == 'male' or gender_census == 'female'):
#            genders.add(gender_census)
#        else:
#            genders.add('unknown')

    if 'unknown' in genders:
        combined_gender = 'unknown'
    elif 'n/a' in genders:
        combined_gender = 'unknown'
    elif 'male' in genders and 'female' in genders:
        combined_gender = 'mixed'
    elif genders == {'male'}:
        combined_gender = 'male'
    elif genders == {'female'}:
        combined_gender = 'female'

    # if no authors, return None for author names and unknown for gender
    elif len(genders) == 0:
        return 'None', 'unknown'
    else:
        raise ValueError("How did you get here?", names, genders)
    
    combined_names = "; ".join(names)

    return combined_names, combined_gender
    


def get_names_and_genders_from_journals():

    """
    Creates a csv that identifies names that we need to clean by hand.

    :return:
    """

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


if __name__ == '__main__':

    generate_journal_dataset(dataset_type='general_journals', dataset_items='sections')
