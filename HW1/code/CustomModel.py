import math
import collections


# Kneser-Ney Smoothing model
class CustomModel:

    def __init__(self, corpus):
        """Initial custom language model and structures needed by this mode"""
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.asFirstTypeCounts = collections.defaultdict(lambda: 0)
        self.asSecondTypeCounts = collections.defaultdict(lambda: 0)
        self.d = 0.75
        self.total = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
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

        # count for continuation
        for w in self.unigramCounts:
            # calculate w as first word
            for bigram in self.bigramCounts:
                if bigram.startswith(w):
                    self.asFirstTypeCounts[w] += 1
                if bigram.endswith(w):
                    self.asSecondTypeCounts[w] += 1
            # calculate w as second word

        self.bigramCounts["UNK"] = 0
        self.unigramCounts["UNK"] = 1
        self.asFirstTypeCounts["UNK"] = 0
        self.asSecondTypeCounts["UNK"] = 0

    def score(self, sentence):
        """ With list of strings, return the log-probability of the sentence with language model. Use
            information generated from train.
        """
        score = 0.0
        prev = " "
        for token in sentence:
            firstTerm = 0.0
            d = self.d
            bigram = "%s %s" % (prev, token)
            countBigram = self.bigramCounts["UNK"]
            countPrev = self.unigramCounts["UNK"]
            asFirst = self.asFirstTypeCounts["UNK"]
            asSecond = self.asSecondTypeCounts["UNK"]
            if bigram in self.bigramCounts:
                countBigram = self.bigramCounts[bigram]

            if countBigram == 0:
                d = -0.000027
            elif countBigram == 1:
                d = -0.448

            if prev in self.unigramCounts:
                countPrev = self.unigramCounts[prev]

            if prev in self.asFirstTypeCounts:
                asFirst = self.asFirstTypeCounts[prev]

            if token in self.asSecondTypeCounts:
                asSecond = self.asSecondTypeCounts[token]

            firstTerm = max(countBigram - d, 0)
            firstTerm = float(firstTerm) / countPrev

            lam = self.d * asFirst / countPrev

            continuation = float(asSecond) / len(self.bigramCounts)
            score += math.log(firstTerm + lam * continuation)
            prev = token
        return score
