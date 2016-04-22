import xkcdfinders
import pandas as pd
from collections import Counter
from pathlib import Path
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk import word_tokenize
import re

# for a list of words seen in training data, what percentage of them do we want in the vocab?
VOCAB_RATIO = 0.6

topics = ["physics",
          "math",
          "chemistry",
          "biology",
          "engineering",
          "algorithms",
          "robotics",
          "school",
          "college",
          "sociology",
          "politics",
          "elections",
          "security",
          "animals",
          "internet",
          "sports",
          "movies",
          "television",
          "superheroes",
          "philosophy",
          "literature",
          "driving",
          "family",
          "love",
          "childhood",
          "statistics",
          "homework",
          "jobs",
          "celebrities",
          "space",
          "friendship",
          "technology",
          "writing",
          "computers",
          "programming",
          "gaming",
          "medicine",
          "health",
          "culture"]

def data_from_csv(file_loc):
    dataset = pd.read_csv(file_loc, header=0, encoding="utf8")
    return dataset.to_dict("list")

def get_test_data(file_loc):
    result = {}
    raw_data = data_from_csv(file_loc)
    for i in range(len(raw_data["Timestamp"])):
        xkcd_num = raw_data["Which xkcd did you get? (enter the number you got)"][i]
        topic = raw_data["Which topic did you get?"][i].lower()
        user_response = raw_data["Was the xkcd relevant to the topic?"][i]

        # if we haven't seen this xkcd yet, make a new entry for it
        if xkcd_num not in result:
            result[xkcd_num] = Counter()

        # positive counts indicate relevance, negative counts indicate irrelevance
        if user_response == "Yes":
            result[xkcd_num][topic] += 1
        else:
            result[xkcd_num][topic] -= 1

    return result

def get_training_data(pathname):
    result = []
    for filename in pathname.glob('*'):
        file_contents = ""
        for line in filename.open():
            file_contents += line
        result.append(file_contents)
    return result

def build_vocab(training_data):
    """
    Construct a vocabulary from the given training data
    """
    stemmer = PorterStemmer()
    words_encountered = FreqDist()
    for text in training_data:
        tokens = word_tokenize(text)
        words_encountered.update([stemmer.stem(word.lower()) for word in tokens if re.search("[A-Za-z0-9]", word) is not None])
    result = list(v[0] for v in words_encountered.most_common(int(words_encountered.B() * VOCAB_RATIO)))
    # add the unknown word token
    result.append("<UNK>")
    return result

if __name__ == '__main__':
    td = get_test_data("training/xkcd_responses.csv")
    train = get_training_data(Path("data"))
    vocab = build_vocab(train)
    print("Testing baseline system...")
    baseline = xkcdfinders.TextualSearchXkcdFinder(debug=True)
    xkcdfinders.evaluate(baseline, topics, td)
    print("Testing tf-idf system...")
    tfidf = xkcdfinders.TfIdfXkcdFinder(vocab=vocab,debug=True)
    tfidf.train(train)
    xkcdfinders.evaluate(tfidf, topics, td)
