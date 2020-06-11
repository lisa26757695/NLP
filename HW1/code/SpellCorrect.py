##
# Credit Zhou Yu, Dan Jurafsky, Peter Norvig
# Updated Kevin Jesse
# Open source code under MIT license
##

import math
from Datum import Datum
from Sentence import Sentence
from Corpus import Corpus
from UniformModel import UniformModel
from UnigramModel import UnigramModel
from BackoffModel import BackoffModel
from SmoothUnigramModel import SmoothUnigramModel
from SmoothBigramModel import SmoothBigramModel
from CustomModel import CustomModel
from EditModel import EditModel
from SpellingResult import SpellingResult
import types
import re
import collections
# import copy


class SpellCorrect:
    """Spelling corrector for sentences. Holds edit model, language model and the corpus."""

    def __init__(self, lm, corpus):
        self.languageModel = lm
        self.editModel = EditModel('data/count_1edit.txt', corpus)

    def correctSentence(self, sentence):
        """Assuming exactly one error per sentence, returns the most probable corrected sentence.
           Sentence is a list of words."""

        if len(sentence) == 0:
            return []

        bestSentence = sentence[:]  # copy of sentence
        bestScore = float('-inf')

        # TODO: select the maximum probability sentence here, according to the noisy channel model.
        # Tip: self.editModel.editProbabilities(word) gives edits and log-probabilities according to your edit model.
        #      You should iterate through these values instead of enumerating all edits.
        # Tip: self.languageModel.score(trialSentence) gives log-probability of a sentence

        for i in xrange(1, len(sentence) - 1):  # ignore <s> and </s>
            originalWord = sentence[i]
            allEdits = self.editModel.editProbabilities(originalWord)
            # iterate through all possible edits
            for edit in allEdits:
                curEditWord = edit[0]
                score = edit[1]
                # if the new word is not in the vocabulary, it is very unlikely it will be the best fit
                if curEditWord in self.editModel.vocabulary:
                    sentence[i] = curEditWord
                    score += self.languageModel.score(sentence)
                if bestScore < score:
                    bestScore = score
                    bestSentence = sentence[:]
            sentence[i] = originalWord
        return bestSentence

    def evaluate(self, corpus):
        """Tests this speller on a corpus, returns a SpellingResult"""
        numCorrect = 0
        numTotal = 0
        testData = corpus.generateTestCases()
        for sentence in testData:
            if sentence.isEmpty():
                continue
            errorSentence = sentence.getErrorSentence()
            hypothesis = self.correctSentence(errorSentence)
            if sentence.isCorrection(hypothesis):
                numCorrect += 1
            numTotal += 1
        return SpellingResult(numCorrect, numTotal)

    def correctCorpus(self, corpus):
        """Corrects a whole corpus, returns a JSON representation of the output."""
        string_list = []  # we will join these with commas,  bookended with []
        sentences = corpus.corpus
        for sentence in sentences:
            uncorrected = sentence.getErrorSentence()
            corrected = self.correctSentence(uncorrected)
            word_list = '["%s"]' % '","'.join(corrected)
            string_list.append(word_list)
        output = '[%s]' % ','.join(string_list)
        return output


def main():
    """Trains all of the language models and tests them on the dev data. Change devPath if you
       wish to do things like test on the training data."""

    trainPath = 'data/tagged-train.dat'
    trainingCorpus = Corpus(trainPath)

    devPath = 'data/tagged-dev.dat'
    devCorpus = Corpus(devPath)

    print 'Unigram Language Model: '
    unigramLM = UnigramModel(trainingCorpus)
    unigramSpell = SpellCorrect(unigramLM, trainingCorpus)
    unigramOutcome = unigramSpell.evaluate(devCorpus)
    print str(unigramOutcome)

    print 'Uniform Language Model: '
    uniformLM = UniformModel(trainingCorpus)
    uniformSpell = SpellCorrect(uniformLM, trainingCorpus)
    uniformOutcome = uniformSpell.evaluate(devCorpus)
    print str(uniformOutcome)

    print 'Smooth Unigram Language Model: '
    smoothUnigramLM = SmoothUnigramModel(trainingCorpus)
    smoothUnigramSpell = SpellCorrect(smoothUnigramLM, trainingCorpus)
    smoothUnigramOutcome = smoothUnigramSpell.evaluate(devCorpus)
    print str(smoothUnigramOutcome)

    print 'Smooth Bigram Language Model: '
    smoothBigramLM = SmoothBigramModel(trainingCorpus)
    smoothBigramSpell = SpellCorrect(smoothBigramLM, trainingCorpus)
    smoothBigramOutcome = smoothBigramSpell.evaluate(devCorpus)
    print str(smoothBigramOutcome)

    print 'Backoff Language Model: '
    backoffLM = BackoffModel(trainingCorpus)
    backoffSpell = SpellCorrect(backoffLM, trainingCorpus)
    backoffOutcome = backoffSpell.evaluate(devCorpus)
    print str(backoffOutcome)

    print 'Custom Language Model: '
    customLM = CustomModel(trainingCorpus)
    customSpell = SpellCorrect(customLM, trainingCorpus)
    customOutcome = customSpell.evaluate(devCorpus)
    print str(customOutcome)


if __name__ == "__main__":
    main()
