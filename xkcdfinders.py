from abc import ABCMeta, abstractmethod
from pyxkcd import get_all_xkcds
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize

class AbstractXkcdFinder(object):
    """
    Abstract base class for all xkcd finder implementations
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def find(self, topic):
        pass

class TextualSearchXkcdFinder(AbstractXkcdFinder):
    """
    Naive xkcd finder implementation that just searches through xkcd's for
    instances of the given word. While it will do basic preprocessing like
    stemming the word (e.g. programming -> program), it will not do any more
    sophisticated steps like word similarity (so programming and coding will not
    both be detected in the same search). This is intended only as a baseline
    system for performance comparisons, and should not be used in production.
    """

    def __init__(self, debug=False):
        self.stemmer = PorterStemmer()
        self.debug = debug
        self.xkcds = None

    def find(self, topic):
        """
        Find all xkcds that directly mention the given topic.
        """
        results = []
        topic = topic.lower() # normalize to lowercase before doing comparisons

        # grab a list of all xkcds as strings. Use the cached version if one exists
        if not self.xkcds:
            if self.debug:
                print("Retrieving xkcds, please wait...")
            self.xkcds = get_all_xkcds()

        for i in range(len(self.xkcds)):
            if self.debug:
                print("Searching xkcd {} of {}".format(i+1, len(self.xkcds)), end='')

            xkcd_text = self.xkcds[i]

            # look for the word inside the current xkcd, stemming words to allow for differences in tense
            if self.stemmer.stem(topic) in [self.stemmer.stem(w.lower()) for w in word_tokenize(xkcd_text)]:
                results.append(i+1)
            if self.debug:
                print('\n' if i+1 == len(self.xkcds) else '\r', end='')

        return results

def evaluate(finder, searches, labeled_data):
    """
    Evaluate the given XkcdFinder by performing searches for each term in the given
    list of searches, and for each search, checking if the resulting comics are found
    in labeled_data. labeled_data is expected to be a dictionary mapping from the search
    terms in searches to lists of xkcds that humans have determined to be relevant to
    each term.
    """
    total_comics = 0
    total_correct = 0
    for search in searches:
        comics = finder.find(search)
        for comic in comics:
            total_comics += 1
            if comic in labeled_data[search]:
                total_correct += 1
    return total_correct / total_comics
