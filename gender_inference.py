from dataset import Dataset
from nameparser import HumanName
from IPython import embed

from collections import defaultdict, Counter

import gender_guesser.detector
GENDER_GUESSER = gender_guesser.detector.Detector()

GENDERED_NAMES = {
    # mostly female to female
    'Mary':         'female',
    'Carol':        'female',
    'Sharon':       'female',
    'Kimberly':     'female',
    'Erin':         'female',
    'Lauren':       'female',
    'Shirley':      'female',
    'Meredith':     'female',
    'Carmen':       'female',
    'Courtney':     'female',

    # male to andy
    'Alex':         'andy',
    'Ali':          'andy',
    'Micah':        'andy',
    'Jean':         'andy',

    # male USA, female international (or vice versa) -> andy
    'Marian':       'andy',
    'Patrice':      'andy',
    'Toni':         'andy',
    'Laurence':     'andy',
    'Eli':          'andy',
    'Sasha':        'andy',
    'Lon':          'andy',

    # male to andy
    'Jaime':        'andy',
    'Glen':         'andy',
    'Rory':         'andy',
    'Jess':         'andy',
    'Ricki':        'andy',

    # mostly male to andy
    'Whitney':      'andy',
    'Leslie':       'andy',

    # female to andy
    'Haven':        'andy',
    'Shannan':      'andy',

    # mostly female to andy
    'Robin':        'andy',


    # mostly male to male
    'George':       'male',
    'Ryan':         'male',
    'Jerry':        'male',
    'Kyle':         'male',
    'Christian':    'male',

}

GENDER_SCORE = {
    'female':           5,
    'mostly_female':    4,
    'unknown':          3,
    'andy':             3,
    'mostly_male':      2,
    'male':             1
}

def compare_name_guessers():

    d = Dataset()

    c = defaultdict(Counter)
    ctot = Counter()

    df = []
    for _, row in d.df.iterrows():
        name = row.AdviseeID.split(':')[0].replace('_', ' ').replace('-', ' ')
        name = " ".join([x.capitalize() for x in name.split()])
        human_name = HumanName(name)

        guess_proquest = row['AdviseeGender.1']
        score_proquest = GENDER_SCORE[guess_proquest]

        guess_naive = GENDER_GUESSER.get_gender(human_name.first, 'usa')
        score_naive = GENDER_SCORE[guess_naive]

        guess_complex = guess_gender_of_name(human_name)
        score_complex = GENDER_SCORE[guess_complex]

#        guesser_gender = guess_gender_of_name(human_name)
#        c[guesser_gender][human_name.first] += 1
#        ctot[guesser_gender] += 1

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


def guess_gender_of_name(human_name:HumanName):
    """
    Guesses the gender of a complete HumanName object

    :param human_name:
    :return:


    >>> h = HumanName('Perez, Haven')
    >>> guess_gender_of_name(h)
    andy

    >>> h = HumanName('Perez, Haven Abraham')
    male

    """

    # Turn "Marten L." into "Marten
    if human_name.middle:
        middle = human_name.middle.split()[0]
    else:
        middle = None

    gender_first = guess_gender_of_first_name(human_name.first)

    if gender_first in ['male', 'female']:
        return gender_first

    # if middle name exists and is not initial ("A."), examine it
    elif middle and (len(middle) > 2 or middle[-1] != '.'):

        gender_middle = guess_gender_of_first_name(middle)

        # if middle name doesn't give us any new info -> return first name info
        if gender_middle in ['unknown', 'andy']:
            return gender_first

        elif gender_middle == 'male' and gender_first in ['mostly_male', 'andy', 'unknown']:
            return 'male'
        elif gender_middle == 'female' and gender_first in ['mostly_female', 'andy', 'unknown']:
            return 'female'

        else:
            if gender_middle in ['male', 'female']:
                print(f'{human_name.first} {human_name.middle} {human_name.last}', gender_first,
                  gender_middle)
            return gender_first

    else:
        return gender_first





def guess_gender_of_first_name(name):
    """
    Guesses gender of a name

    >>> guess_gender_of_first_name('Mary')
    female

    >>> guess_gender_of_first_name('L.')
    unknown

    :return:
    """

    if name in GENDERED_NAMES:
        return GENDERED_NAMES[name]
    else:
        gen_usa = GENDER_GUESSER.get_gender(name, 'usa')
        gen_all = GENDER_GUESSER.get_gender(name)

        if gen_usa in ['female', 'male']:
            gender = gen_usa

        # applies to e.g. George, Jerry, Christian, Max, Gabriel
        elif gen_usa == 'mostly_male' and gen_all == 'male':
            gender = 'male'

        # applies mostly to international names
        # Jorge, Jaime*, Eduardo, Pablo, Mohamed, Fernando,
        elif gen_usa == 'andy' and gen_all == 'male':
            gender = 'male'

        # applies to Sharon, Dana, Shirley, Carmen, Clare, Sandie
        elif gen_usa == 'mostly_female' and gen_all == 'female':
            gender = 'female'

        # applies mostly to international names
        # Ana, Karin, Elisabeth, Marta, Elena, Marguerite, Olga
        elif gen_usa == 'andy' and gen_all == 'female':
            gender = 'female'

        else:
            gender = gen_usa

        return gender

if __name__ == '__main__':
    compare_name_guessers()
