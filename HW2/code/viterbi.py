#!/usr/bin/python

# Noah A. Smith
# 2/21/08
# Runs the Viterbi algorithm (no tricks other than logmath!), given an
# HMM, on sentences, and outputs the best state path.

# Usage:  viterbi.pl hmm-file < text > tags

# The hmm-file should include two kinds of lines.  One is a transition:
# trans Q R P
# where Q and R are whitespace-free state names ("from" and "to,"
# respectively) and P is a probability.  The other kind of line is an
# emission:
# emit Q S P
# where Q is a whitespace-free state name, S is a whitespace-free
# emission symbol, and P is a probability.  It's up to you to make sure
# your HMM properly mentions the start state (named init by default),
# the final state (named final by default) and out-of-vocabulary
# symbols (named OOV by default).

# If the HMM fails to recognize a sequence, a blank line will be written.
# Change $verbose to 1 for more verbose output.

# special keywords:
#  $init_state   (an HMM state) is the single, silent start state
#  $final_state  (an HMM state) is the single, silent stop state
#  $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word

import sys
import re
import math

hmmFile = sys.argv[1]
txtFile = sys.argv[2]

INIT_STATE = 'init'
FINAL_STATE = 'final'
OOV_SYMBOL = 'OOV'


tags = set()
# TODO change to set
vocab = {}
emissions = {}
transitions = {}

# read in the HMM and store the probabilities as log probabilities
with open(hmmFile) as hmmFile:
    for line in hmmFile.read().splitlines():
        transMatch = re.match('trans\s+(\S+)\s+(\S+)\s+(\S+)', line)
        emitMatch = re.match('emit\s+(\S+)\s+(\S+)\s+(\S+)', line)
        if transMatch:
            qq, q, p = transMatch.groups()
            transitions[(qq, q)] = math.log(float(p))
            tags.add(qq)
            tags.add(q)
        elif emitMatch:
            q, w, p = emitMatch.groups()
            emissions[(q, w)] = math.log(float(p))
            tags.add(q)
            vocab[w] = 1


with open(txtFile) as txtFile:
    for line in txtFile.read().splitlines():
        line = line.rstrip()
        words = line.split()
        n = len(words)
        words.insert(0, "")
        V = {(0, INIT_STATE): 0.0}  # base case of the recurisve equations!
        bp = {}  # backpointers

        for i in range(1, n+1):
            # if a word isn't in the vocabulary, rename it with the OOV symbol
            if words[i] not in vocab:
                words[i] = OOV_SYMBOL
            # consider each possible tags
            for q in tags:
                for qq in tags:
                    # only consider "non-zeros"
                    if (qq, q) in transitions and (q, words[i]) in emissions and (i-1, qq) in V:
                        v = V[(i-1, qq)] + transitions[(qq, q)] + \
                            emissions[(q, words[i])]
                        if (i, q) not in V or v > V[(i, q)]:
                            # if we found a better previous state, take note!
                            # Viterbi probability
                            V[(i, q)] = v
                            # best previous state
                            bp[(i, q)] = qq

        # this handles the last of the Viterbi equations, the one that brings in the final state.
        foundgoal = False
        goal = float('-inf')
        tag = ""
        # for each possible state for the last word
        for qq in tags:
            if (qq, FINAL_STATE) in transitions and (n, qq) in V:
                p = V[(n, qq)] + transitions[(qq, FINAL_STATE)]
                if (not foundgoal) or p > goal:
                    # we found a better path; remember it
                    goal = p
                    foundgoal = True
                    tag = qq

        if foundgoal:
            allTags = []
            for i in xrange(n, 0, -1):
                allTags.append(tag)
                tag = bp[(i, tag)]

            allTags.reverse()
            print ' '.join(allTags)
        else:
            print ''
