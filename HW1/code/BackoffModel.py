import math
import collections


class BackoffModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 1)
        self.bigramCounts = collections.defaultdict(lambda: 0)
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

        self.bigramCounts["UNK"] = 0
        self.unigramCounts["UNK"] = 1

        # count = 0
        # for data in self.unigramCounts:
        #     count += 1

        # print("count is " + str(count))

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        prev = " "
        for token in sentence:
            # print(token)
            # length = len(self.bigramCounts)
            bigram = "%s %s" % (prev, token)
            # bigram = prev + " " + token
            if bigram in self.bigramCounts:
                # print("a")
                score += math.log(self.bigramCounts[bigram])
                if prev in self.unigramCounts:
                    score -= math.log(self.unigramCounts[prev])
                else:
                    score -= math.log(self.unigramCounts["UNK"])
            else:
                # discount = 0.3
                discount = 0.4
                # print("b")
                if token in self.unigramCounts:
                    score += math.log(self.unigramCounts[token])
                else:
                    score += math.log(self.unigramCounts["UNK"])
                score += math.log(discount) - \
                    math.log(self.total + len(self.unigramCounts))
            prev = token
            # if length != len(self.bigramCounts):
            #     print("Alert!")
        # print("Size is " + str(len(self.unigramCounts)))
        return score
