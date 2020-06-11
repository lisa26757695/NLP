import math
import collections


class SmoothUnigramModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 1)
        self.total = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            for datum in sentence.data:
                token = datum.word
                self.unigramCounts[token] += 1
                self.total += 1
        # add UNK
        self.unigramCounts["UNK"] = 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        for token in sentence:
            if token in self.unigramCounts:
                score += math.log(self.unigramCounts[token])
            else:
                score += math.log(self.unigramCounts["UNK"])
            score -= math.log(self.total + len(self.unigramCounts))
        return score
