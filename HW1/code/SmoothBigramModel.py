import math
import collections


class SmoothBigramModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 1)
        self.total = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            prev = " "
            for datum in sentence.data:
                self.total += 1
                token = datum.word
                self.unigramCounts[token] += 1
                bigram = "%s %s" % (prev, token)
                self.bigramCounts[bigram] += 1
                prev = token

        self.bigramCounts["UNK"] = 1
        self.unigramCounts["UNK"] = 0

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        prev = " "
        for token in sentence:
            # length = len(self.bigramCounts)
            bigram = "%s %s" % (prev, token)
            # bigram = prev + " " + token
            if bigram in self.bigramCounts:
                score += math.log(self.bigramCounts[bigram])
            else:
                score += math.log(self.bigramCounts["UNK"])
            if prev in self.unigramCounts:
                score -= math.log(self.unigramCounts[prev] +
                                  len(self.bigramCounts))
            else:
                score -= math.log(self.unigramCounts["UNK"] +
                                  len(self.bigramCounts))
            prev = token
            # if length != len(self.bigramCounts):
            #     print("Alert!")

        return score
