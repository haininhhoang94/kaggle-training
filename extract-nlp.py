# %%
# https://www.shiksha.com/online-courses/articles/extracting-information-from-text-data-using-spacy-in-nlp/
import pandas as pd

# Split the text and create a list
with open("./data/data.txt", "r") as f:
    data = [line for line in f.readlines()]

# %%
# Create a dataframe using the list
df = pd.DataFrame(data, columns=["text"])
df.head()

# %%
import spacy
import en_core_web_sm

text = df["text"][0]
nlp = en_core_web_sm.load()
doc = nlp(text)

# %%
features = []
for token in doc:
    features.append({"token": token.text, "pos": token.pos_})

# %%
fdf = pd.DataFrame(features)
fdf.head(len(fdf))
# %%
first_tokens = ["winner", "name"]
last_tokens = ["was", "born"]

# %%
pattern_winner = [
    [
        {
            "LOWER": {"IN": first_tokens}
        },  # preceding words  without case matching
        {"POS": "PROPN", "OP": "+"},  # searching for PROPN  one or more times
        {"LOWER": {"IN": last_tokens}},
    ]
]  # following words  without case matching

# %%
from spacy.matcher import Matcher


def get_winner(x):
    nlp = en_core_web_sm.load()
    doc = nlp(x)
    matcher = Matcher(nlp.vocab)
    matcher.add("matching_winner", pattern_winner)
    matches = matcher(doc)
    sub_text = ""
    if len(matches) > 0:
        span = doc[matches[0][1] : matches[0][2]]
        sub_text = span.text
    tokens = sub_text.split(" ")

    name, surname = tokens[1:-1]
    return name, surname


# %%
new_columns = ["scientist name", "surname"]
for n, col in enumerate(new_columns):
    df[col] = df["text"].apply(lambda x: get_winner(x)).apply(lambda x: x[n])

df.head()

# %%
first_tokens = ["in"]
last_tokens = ["."]
pattern_country = [
    [
        {"LOWER": {"IN": first_tokens}},
        {"POS": "PROPN", "OP": "+"},
        {"LOWER": {"IN": last_tokens}},
    ]
]

from spacy.matcher import Matcher


def get_country(x):
    nlp = en_core_web_sm.load()
    doc = nlp(x)
    matcher = Matcher(nlp.vocab)
    matcher.add("matching_country", pattern_country)
    matches = matcher(doc)
    sub_text = ""
    if len(matches) > 0:
        span = doc[matches[0][1] : matches[0][2]]
        sub_text = span.text

    # remove punct
    sub_text = sub_text[:-1]
    tokens = sub_text.split(" ")

    return " ".join(tokens[1:])


# %%
df["country"] = df["text"].apply(lambda x: get_country(x))
df.head()


# %%
def get_date(x):
    months = {
        "January": "01",
        "February": "02",
        "March": "03",
        "April": "04",
        "May": "05",
        "June": "06",
        "July": "07",
        "August": "08",
        "September": "09",
        "October": "10",
        "November": "11",
        "December": "12",
    }
    tokens = x.split(" ")
    # month
    month = months[tokens[1]]
    # day
    day = tokens[2]
    if len(day) == 1:
        day = "0" + day

    # year
    year = x.split(" ")[3]

    return year + "-" + month + "-" + day


# %%
df["birthdate"] = df["text"].apply(lambda x: get_date(x))
df.head()


# %%
def get_gender(x):
    if "He" in x:
        return "M"
    return "F"


df["gender"] = df["text"].apply(lambda x: get_gender(x))
df.head()

# %%
