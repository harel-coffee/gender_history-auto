from dataset_dissertation import DissertationDataset
from nameparser import HumanName
from IPython import embed
from pathlib import Path
import json
import pandas as pd

from collections import defaultdict, Counter

import gender_guesser.detector
GENDER_GUESSER = gender_guesser.detector.Detector()

# GENDERED_NAMES = {
#     # mostly female to female
#     'Mary':         'female',
#     'Carol':        'female',
#     'Sharon':       'female',
#     'Kimberly':     'female',
#     'Erin':         'female',
#     'Lauren':       'female',
#     'Shirley':      'female',
#     'Meredith':     'female',
#     'Carmen':       'female',
#     'Courtney':     'female',
#
#     # male to andy
#     'Alex':         'andy',
#     'Ali':          'andy',
#     'Micah':        'andy',
#     'Jean':         'andy',
#
#     # male USA, female international (or vice versa) -> andy
#     'Marian':       'andy',
#     'Patrice':      'andy',
#     'Toni':         'andy',
#     'Laurence':     'andy',
#     'Eli':          'andy',
#     'Sasha':        'andy',
#     'Lon':          'andy',
#
#     # male to andy
#     'Jaime':        'andy',
#     'Glen':         'andy',
#     'Rory':         'andy',
#     'Jess':         'andy',
#     'Ricki':        'andy',
#
#     # mostly male to andy
#     'Whitney':      'andy',
#     'Leslie':       'andy',
#
#     # female to andy
#     'Haven':        'andy',
#     'Shannan':      'andy',
#
#     # mostly female to andy
#     'Robin':        'andy',
#
#
#     # mostly male to male
#     'George':       'male',
#     'Ryan':         'male',
#     'Jerry':        'male',
#     'Kyle':         'male',
#     'Christian':    'male',
# }

# GENDER_SCORE = {
#     'female':           5,
#     'mostly_female':    4,
#     'unknown':          3,
#     'andy':             3,
#     'mostly_male':      2,
#     'male':             1
# }

def compare_name_guessers():

    d = DissertationDataset()

    stanford_name_to_gender_dict = {}

    df = []
    for _, row in d.df.iterrows():
        name = row.AdviseeID.split(':')[0].replace('_', ' ').replace('-', ' ')
        name = " ".join([x.capitalize() for x in name.split()])
        human_name = HumanName(name)

        guess_proquest = row['AdviseeGender.1']
        score_proquest = GENDER_SCORE[guess_proquest]

        stanford_name_to_gender_dict[human_name.first] = guess_proquest

        guess_naive = guess_gender_first_name_only_usa(human_name)
        score_naive = GENDER_SCORE[guess_naive]

        guess_complex = guess_gender_with_middle_name_and_international_names(human_name)
        score_complex = GENDER_SCORE[guess_complex]

        df.append({
            'name_raw':     row.AdviseeID,
            'name':         name,

            'guess_orig':   guess_proquest,
            'score_orig':   score_proquest,

            'guess_naive':  guess_naive,
            'score_naive':  score_naive,

            'guess_complex': guess_complex,
            'score_complex': score_complex,

            'dif_orig_naive':    abs(score_proquest - score_naive),
            'dif_orig_complex':  abs(score_proquest - score_complex),
            'dif_naive_complex': abs(score_naive - score_complex)
        })

    # store dict of stanford names to gender
    with open(Path('data', 'gender_inference', 'stanford_name_to_gender.json'), 'w') as out:
        json.dump(stanford_name_to_gender_dict, out, sort_keys=True, indent=4)


# with open(Path('data', 'stanford_name_to_gender.json'), 'r') as infile:
#     STANFORD_NAME_TO_GENDER = json.load(infile)

#
# def guess_gender_stanford(human_name: HumanName):
#     """
#     Returns the gender guessed by the model trained by Dan's group at Stanford
#
#     (uses the dissertation dataset for inference)
#
#     >>> h = HumanName('Perez, Ali Abraham')
#     >>> guess_gender_stanford(h)
#     'female'
#
#     :param human_name:
#     :return:
#     """
#
#     if len(human_name.first) == 1:
#         name_to_guess = human_name.middle
#     elif len(human_name.first) == 2 and human_name.first[1] == '.':
#         name_to_guess = human_name.middle
#     else:
#         name_to_guess = human_name.first
#
#     if human_name.first in STANFORD_NAME_TO_GENDER:
#         return STANFORD_NAME_TO_GENDER[name_to_guess]
#     else:
#         return 'unknown'


CENSUS_DF = pd.read_csv(Path('data', 'census_gender.csv'), sep=';')
CENSUS_NAME_TO_MALE_PROBABILITY = {}
for _, row in CENSUS_DF.iterrows():
    CENSUS_NAME_TO_MALE_PROBABILITY[row['firstname']] = row['percm']

HAND_CODED_DF = pd.read_csv(Path('data', 'journal_author_genders.csv'))
HAND_CODED_DF.fillna('',inplace=True)
NAME_TO_HAND_CODED_GENDER = {}
for _, row in HAND_CODED_DF.iterrows():
    full_name = row['first_name']
    if row['last_name']: full_name += f' {row["last_name"]}'
    NAME_TO_HAND_CODED_GENDER[full_name] = row['final_gender']


def get_hand_coded_gender(human_name: HumanName):
    """
    Uses the hand-coded gender data. Returns 'unknown' if name is not in dataset


    >>> h = HumanName('Palmier, Leslie H.')
    >>> get_hand_coded_gender(h)
    'male'

    >>> h = HumanName('Palmier, Leslie Henry')
    >>> get_hand_coded_gender(h)
    'n/a'

    >>> h = HumanName('Purcell, Edward A.')
    >>> get_hand_coded_gender(h)
    'male'


    :param human_name:
    :return:
    """
    full_name = human_name.first
    if human_name.middle:
        full_name += f' {human_name.middle}'
    if human_name.last:
        full_name += f' {human_name.last}'

    if full_name in NAME_TO_HAND_CODED_GENDER:
        gender = NAME_TO_HAND_CODED_GENDER[full_name]
        if gender == '':
            gender = 'unknown'
        return gender
    else:
        print(full_name, "n/a")
        return 'n/a'


def guess_gender_census(human_name: HumanName, return_type='gender'):
    """
    Use census data to infer gender.


    >>> h = HumanName('Perez, Ali Abraham')
    >>> guess_gender_census(h, return_type='gender')
    'andy'

    # the function can also be used to get the probabily that a name is male in the census
    >>> guess_gender_census(h, return_type='probability_male')
    '72% male'

    >>> h = HumanName('Perez, A. Abraham')
    >>> guess_gender_census(h)
    'male'

    >>> h = HumanName('Perez, A Abraham')
    >>> guess_gender_census(h)
    'male'


    :param human_name:
    :param return_type:
    :return:
    """

    if len(human_name.first) == 1:
        name_to_guess = human_name.middle.lower()
    elif len(human_name.first) == 2 and human_name.first[1] == '.':
        name_to_guess = human_name.middle.lower()
    else:
        name_to_guess = human_name.first.lower()


    if name_to_guess in CENSUS_NAME_TO_MALE_PROBABILITY:
        male_prob = CENSUS_NAME_TO_MALE_PROBABILITY[name_to_guess]
        if male_prob < 5:
            inferred_gender = 'female'
        elif male_prob < 15:
            inferred_gender = 'mostly_female'
        elif male_prob < 85:
            inferred_gender = 'andy'
        elif male_prob < 95:
            inferred_gender = 'mostly_male'
        else:
            inferred_gender = 'male'

        if return_type == 'gender':
            return inferred_gender
        elif return_type == 'probability_male':
            return '{:2.0f}% male'.format(male_prob)

    else:
        return 'unknown'

#
# def guess_gender_first_name_only_usa(human_name: HumanName):
#     """
#     Guesses the gender of a HumanName object
#     Uses only the first name and only the U.S. dictionary.
#
#     >>> h = HumanName('Perez, Ali Abraham')
#     >>> guess_gender_with_middle_name_and_international_names(h)
#     'male'
#
#     :param human_name:
#     :return:
#     """
#
#     if human_name.first in GENDERED_NAMES:
#         return GENDERED_NAMES[human_name.first]
#     else:
#         return GENDER_GUESSER.get_gender(human_name.first, 'usa')
#
#
# def guess_gender_with_middle_name_and_international_names(human_name:HumanName):
#     """
#     Guesses the gender of a complete HumanName object
#     Uses first and middle name and both u.s. and international dictionaries.
#
#     >>> h = HumanName('Perez, Haven')
#     >>> guess_gender_first_name_only_usa(h)
#     'andy'
#
#     :param human_name:
#     :return:
#
#
#     >>> h = HumanName('Perez, Haven')
#     >>> guess_gender_first_name_only_usa(h)
#     'andy'
#
#     >>> h = HumanName('Perez, Haven Abraham')
#     >>> guess_gender_with_middle_name_and_international_names(h)
#     'male'
#
#     """
#
#     # Turn "Marten L." into "Marten
#     if human_name.middle:
#         middle = human_name.middle.split()[0]
#     else:
#         middle = None
#
#     gender_first = guess_gender_of_first_name(human_name.first)
#
#     if gender_first in ['male', 'female']:
#         return gender_first
#
#     # if middle name exists and is not initial ("A."), examine it
#     elif middle and (len(middle) > 2 or middle[-1] != '.'):
#
#         gender_middle = guess_gender_of_first_name(middle)
#
#         # if middle name doesn't give us any new info -> return first name info
#         if gender_middle in ['unknown', 'andy']:
#             return gender_first
#
#         elif gender_middle == 'male' and gender_first in ['mostly_male', 'andy', 'unknown']:
#             return 'male'
#         elif gender_middle == 'female' and gender_first in ['mostly_female', 'andy', 'unknown']:
#             return 'female'
#
#         else:
#             if gender_middle in ['male', 'female']:
#                 print(f'{human_name.first} {human_name.middle} {human_name.last}', gender_first,
#                   gender_middle)
#             return gender_first
#
#     else:
#         return gender_first
#
#
#
#
#
# def guess_gender_of_first_name(name):
#     """
#     Guesses gender of a name
#
#     >>> guess_gender_of_first_name('Mary')
#     female
#
#     >>> guess_gender_of_first_name('L.')
#     unknown
#
#     :return:
#     """
#
#     if name in GENDERED_NAMES:
#         return GENDERED_NAMES[name]
#     else:
#         gen_usa = GENDER_GUESSER.get_gender(name, 'usa')
#         gen_all = GENDER_GUESSER.get_gender(name)
#
#         if gen_usa in ['female', 'male']:
#             gender = gen_usa
#
#         # applies to e.g. George, Jerry, Christian, Max, Gabriel
#         elif gen_usa == 'mostly_male' and gen_all == 'male':
#             gender = 'male'
#
#         # applies mostly to international names
#         # Jorge, Jaime*, Eduardo, Pablo, Mohamed, Fernando,
#         elif gen_usa == 'andy' and gen_all == 'male':
#             gender = 'male'
#
#         # applies to Sharon, Dana, Shirley, Carmen, Clare, Sandie
#         elif gen_usa == 'mostly_female' and gen_all == 'female':
#             gender = 'female'
#
#         # applies mostly to international names
#         # Ana, Karin, Elisabeth, Marta, Elena, Marguerite, Olga
#         elif gen_usa == 'andy' and gen_all == 'female':
#             gender = 'female'
#
#         else:
#             gender = gen_usa
#
#         return gender

if __name__ == '__main__':
    compare_name_guessers()
