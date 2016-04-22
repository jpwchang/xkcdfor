from abc import ABCMeta, abstractmethod
from pyxkcd import get_all_xkcds
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.probability import FreqDist
import numpy, re

class AbstractXkcdFinder(object):
    """
    Abstract base class for all xkcd finder implementations
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, training_data):
        pass

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

    def __init__(self, num_results=20, debug=False):
        self.stemmer = PorterStemmer()
        self.num_results = num_results
        self.debug = debug
        self.xkcds = None

    def train(self, training_data):
        """
        dummy training routine.
        for a raw text search, no training is necessary.
        """
        return 1

    def find(self, topic):
        """
        Find all xkcds that directly mention the given topic.
        """

        # grab a list of all xkcds as strings. Use the cached version if one exists
        if not self.xkcds:
            if self.debug:
                print("Retrieving xkcds, please wait...")
            self.xkcds = get_all_xkcds()

        scores = numpy.zeros(len(self.xkcds))
        topic = self.stemmer.stem(topic.lower()) # normalize to lowercase before doing comparisons

        for i in range(len(self.xkcds)):
            if self.debug:
                print("Searching xkcd {} of {}".format(i+1, len(self.xkcds)), end='')

            xkcd_text = self.xkcds[i]

            # look for the word inside the current xkcd, stemming words to allow for differences in tense
            for token in word_tokenize(xkcd_text):
                if topic == self.stemmer.stem(token.lower()):
                    scores[i] += 1
            if self.debug:
                print('\n' if i+1 == len(self.xkcds) else '\r', end='')

        results = numpy.argpartition(-scores, self.num_results)[:self.num_results]
        return results + 1

class TfIdfXkcdFinder(AbstractXkcdFinder):
    """
    XkcdFinder implementation which computes an xkcd's relevance score by summing up
    vector similarity between the input query and the string representation of the
    xkcd. This implementation uses tf-idf weighted word-document matrix as its
    vector space model.
    """

    def __init__(self, vocab=None, vocab_ratio=0.6, num_results=20, debug=False):
        self.stemmer = PorterStemmer()
        self.vocab = vocab
        self.vocab_ratio = vocab_ratio
        self.num_results = num_results
        self.debug = debug
        self.xkcds = None

    def train(self, training_data):
        """
        Trains this finder's ti-idf model on the provided training data.
        The training data is expected to come in the form of a list of strings.
        """

        # first, build a vocabulary if we were not given one
        if not self.vocab:
            if self.debug:
                print("Building vocabulary, please wait...")
            self.vocab = self._build_vocab(training_data)

        # create a map from vocabulary words to indices in vocabulary, allowing constant-time
        # conversion of words to indices in the vocabulary (allowing quicker lookup in the
        # co-ocurrence matrix)
        self.word_index = {}
        for i in range(len(self.vocab)):
            self.word_index[self.vocab[i]] = i

        # grab a list of all xkcds as strings. Use the cached version if one exists
        if not self.xkcds:
            if self.debug:
                print("Retrieving xkcds, please wait...")
            self.xkcds = get_all_xkcds()

        co_matrix = self._build_coocurrence_matrix(training_data)

        # weight the co-ocurrence matrix using tf-idf weighting
        if self.debug:
            print("Performing ti-idf weighting, please wait...")
        self.word_vectors = numpy.copy(co_matrix)
        # first, fill the vector model with raw term frequencies
        for td_idx in range(len(training_data)):
            self.word_vectors[:,td_idx] = self.word_vectors[:,td_idx] / numpy.sum(self.word_vectors[:,td_idx])
        # next, compute the idf for each word
        idf_matrix = numpy.zeros(len(self.vocab))
        for word_idx in range(len(self.vocab)):
            for td_idx in range(len(training_data)):
                if co_matrix[word_idx, td_idx] > 0:
                    idf_matrix[word_idx] += 1
        idf_matrix = numpy.log(len(training_data) / idf_matrix)
        # finally, multiply to get the final ti-idf matrix
        self.word_vectors *= idf_matrix.reshape(-1,1)

        if self.debug:
            print("Training completed")

    def find(self, topic):
        """
        Find all xkcds with similar words to the given topic.
        """
        doc_vectors = numpy.zeros([len(self.word_vectors[0,:]), len(self.xkcds)])
        scores = numpy.zeros(len(self.xkcds))
        topic = self.stemmer.stem(topic.lower())

        for i in range(len(self.xkcds)):
            if self.debug:
                print("Searching xkcd {} of {}".format(i+1, len(self.xkcds)), end='')

            xkcd_text = self.xkcds[i]
            words_considered = 0

            # compute similarities for each word in the text of the current xkcd
            for token in word_tokenize(xkcd_text):
                if re.search("[A-Za-z0-9]", token) is not None:
                    cur_word = self.stemmer.stem(token.lower())
                    if cur_word not in self.word_index:
                        cur_word = "<UNK>"
                    #scores[i] += self._stemmed_similarity(topic, cur_word)
                    doc_vectors[:,i] += self.word_vectors[self.word_index[cur_word],:]
                    words_considered += 1

            # to avoid wrongly preferring longer xkcds, we normalize by document length
            #scores[i] /= words_considered

            if self.debug:
                print('\n' if i+1 == len(self.xkcds) else '\r', end='')

        topic_vec = self.word_vectors[self.word_index[topic],:]

        for i in range(len(self.xkcds)):
            scores[i] = topic_vec.dot(doc_vectors[:,i]) / (numpy.linalg.norm(topic_vec) * numpy.linalg.norm(doc_vectors[:,i]))
        results = numpy.argpartition(-scores, self.num_results)[:self.num_results]
        return results + 1

    def similarity(self, word1, word2):
        """
        Uses the trained vector space model to compute similarity between two words
        """
        word1 = self.stemmer.stem(word1.lower())
        word2 = self.stemmer.stem(word2.lower())
        if word1 not in self.word_index:
            print("Warning: unrecognized word {}".format(word1))
            word1 = "<UNK>"
        if word2 not in self.word_index:
            print("Warning: unrecognized word {}".format(word2))
            word2 = "<UNK>"
        return self._stemmed_similarity(word1, word2)

    def _stemmed_similarity(self, word1, word2):
        index1 = self.word_index[word1]
        index2 = self.word_index[word2]
        vector1 = self.word_vectors[index1,:]
        vector2 = self.word_vectors[index2,:]
        return vector1.dot(vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))

    def _build_vocab(self, training_data):
        """
        Construct a vocabulary from the given training data
        """
        words_encountered = FreqDist()
        for text in training_data:
            tokens = word_tokenize(text)
            words_encountered.update([self.stemmer.stem(word.lower()) for word in tokens if re.search("[A-Za-z0-9]", word) is not None])
        result = list(v[0] for v in words_encountered.most_common(int(words_encountered.B() * self.vocab_ratio)))
        # add the unknown word token
        result.append("<UNK>")
        return result

    def _build_coocurrence_matrix(self, training_data):
        """
        build a word-document co-ocurrence matrix with the given training data.
        """
        if self.debug:
            print("Building co-occurence matrix, please wait...")
        # co-occurence matrix is indexed by (word,document)
        co_matrix = numpy.zeros([len(self.vocab), len(training_data)])
        for td_idx in range(len(training_data)):
            text = training_data[td_idx]
            tokens = word_tokenize(text)
            for token in tokens:
                clean_token = self.stemmer.stem(token.lower())
                if clean_token not in self.word_index:
                    clean_token = "<UNK>"
                # only consider tokens that have some alphanumeric characters to be real words
                if re.search("[A-Za-z0-9]", clean_token) is not None:
                    word_idx = self.word_index[clean_token]
                    co_matrix[word_idx, td_idx] += 1
        return co_matrix

def evaluate(finder, searches, labeled_data):
    """
    Evaluate the given XkcdFinder by performing searches for each term in the given
    list of searches, and for each search, checking if the resulting comics are found
    in labeled_data. labeled_data is expected to be a dictionary where the keys are
    xkcd comic numbers and the entries are Python Counter objects, keyed by topic,
    where a negative value represents a human label of irrelevant, and a positive
    value represents a human label of relevant.
    """
    total_relevant = 0
    total_irrelevant = 0
    for search in searches:
        comics = finder.find(search)
        for comic in comics:
            if comic in labeled_data and search in labeled_data[comic]:
                if labeled_data[comic][search] > 0:
                    total_relevant += 1
                elif labeled_data[comic][search] < 0:
                    total_irrelevant += 1
    print("Total comics found that were relevant: {}".format(total_relevant))
    print("Total comics found that were irrelevant: {}".format(total_irrelevant))
    print("Relevance rate: {}".format(total_relevant/(total_relevant+total_irrelevant)))
