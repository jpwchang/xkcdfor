import xkcdfinders
import pandas as pd
from collections import Counter

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

if __name__ == '__main__':
    td = get_test_data("training/xkcd_responses.csv")
    baseline = xkcdfinders.TextualSearchXkcdFinder(debug=True)
    xkcdfinders.evaluate(baseline, topics, td)
