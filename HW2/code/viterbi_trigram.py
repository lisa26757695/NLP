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
states = {}
emissions = {}
transitions = {}

# read in the HMM and store the probabilities as log probabilities
with open(hmmFile) as hmmFile:
    for line in hmmFile.read().splitlines():
        # adjust the regular expression for trigram
        transMatch = re.match('trans\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
        emitMatch = re.match('emit\s+(\S+)\s+(\S+)\s+(\S+)', line)

        if transMatch:
            qqq, qq, q, p = transMatch.groups()
            transitions[(qqq, qq, q)] = math.log(float(p))
            tags.add(qqq)
            tags.add(qq)
            tags.add(q)
        elif emitMatch:
            q, w, p = emitMatch.groups()
            emissions[(q, w)] = math.log(float(p))
            tags.add(q)
            states[w] = 1

with open(txtFile) as txtFile:
    for line in txtFile.read().splitlines():
        line = line.rstrip()
        words = line.split()
        n = len(words)
        words.insert(0, "")
        # base case of the recurisve equations!
        V = {(0, INIT_STATE, INIT_STATE): 0.0}
        Backtrace = {}  # backpointers

        for i in range(1, n+1):
            # if a word isn't in the vocabulary, rename it with the OOV symbol
            if words[i] not in states:
                words[i] = OOV_SYMBOL
            # consider each possible tags
            for q in tags:
                for qq in tags:
                    for qqq in tags:
                        # smoothing
                        if (q, words[i]) not in emissions:
                            emissions[(q, words[i])] = math.log(0.0000001)
                        if (qqq, qq, q) in transitions and (q, words[i]) in emissions and (i-1, qqq, qq) in V:
                            v = V[(i-1, qqq, qq)] + transitions[(qqq, qq, q)] + \
                                emissions[(q, words[i])]
                            if (i, qq, q) not in V or v > V[(i, qq, q)]:
                                # if we found a better previous state, take note!
                                # Viterbi probability
                                V[(i, qq, q)] = v
                                # best previous state
                                Backtrace[(i, qq, q)] = qqq

        # To find and record the max probability of a tag to terminate the sentence
        for q in tags:
            for qq in tags:
                if (qq, q, FINAL_STATE) in transitions and (n, qq, q) in V:
                    v = V[(n, qq, q)] + transitions[(qq, q, FINAL_STATE)]
                    if (n + 1, q, FINAL_STATE) not in V or v > V[(n + 1, q, FINAL_STATE)]:
                        V[(n + 1, q, FINAL_STATE)] = v
                        Backtrace[(n + 1, q, FINAL_STATE)] = qq

        for q in tags:
            if (q, FINAL_STATE, FINAL_STATE) in transitions and (n + 1, q, FINAL_STATE) in V:
                v = V[(n + 1, q, FINAL_STATE)] + \
                    transitions[(q, FINAL_STATE, FINAL_STATE)]
                if (n + 2, FINAL_STATE, FINAL_STATE) not in V or v > V[(n + 2, FINAL_STATE, FINAL_STATE)]:
                    V[(n + 2, FINAL_STATE, FINAL_STATE)] = v
                    Backtrace[(n + 2, FINAL_STATE, FINAL_STATE)] = q

        # Actually find the best tag to terminate the sentence
        foundgoal = False
        goal = float('-inf')
        pretag = ""
        if (n + 2, FINAL_STATE, FINAL_STATE) in V:
            goal = V[(n + 2, FINAL_STATE, FINAL_STATE)]
            pretag = Backtrace[(n + 2, FINAL_STATE, FINAL_STATE)]
            foundgoal = True
        # find the best final tag

        # for each possible state for the last word
        for q in tags:
            if (q, FINAL_STATE, FINAL_STATE) in transitions and (n + 1, q, FINAL_STATE) in V:
                p = V[(n+1, q, FINAL_STATE)] + \
                    transitions[(q, FINAL_STATE, FINAL_STATE)]
                if p > goal:
                    # we found a better path; remember it
                    goal = p
                    foundgoal = True
                    pretag = q

        # back tracking to find the best path
        if foundgoal:
            allTags = []
            tag = FINAL_STATE
            for i in xrange(n + 1, 1, -1):
                allTags.append(pretag)
                temp = pretag
                pretag = Backtrace[(i, pretag, tag)]
                tag = temp

            # Because all the tags are appended from the end to start, we need to reverse it
            allTags.reverse()
            print ' '.join(allTags)
        else:
            print()
