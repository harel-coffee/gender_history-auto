from pathlib import Path
import os
import re

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent

# Split words but keep apostrophes, e.g. women's
WORD_SPLIT_REGEX = re.compile(r"\b\w[\w']+\b")