#!/usr/bin/python

# custom model with trigram and smoothing (smoothing is done during viterbi algorithm)

import sys
import re
from itertools import izip
from collections import defaultdict

TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]

vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

emissions = {}
transitions = {}
transitionsTotal = defaultdict(int)
emissionsTotal = defaultdict(int)

with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
    for tagString, tokenString in izip(tagFile, tokenFile):

        tags = re.split("\s+", tagString.rstrip())
        tokens = re.split("\s+", tokenString.rstrip())
        pairs = zip(tags, tokens)

        words = tokenString.split(" ")
        prevprevtag = INIT_STATE
        prevtag = INIT_STATE

        for (tag, token) in pairs:
            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD

            if tag not in emissions:
                emissions[tag] = defaultdict(int)

            if prevprevtag not in transitions:
                transitions[prevprevtag] = defaultdict(dict)

            if prevtag not in transitions[prevprevtag]:
                transitions[prevprevtag][prevtag] = defaultdict(int)

            if prevprevtag not in transitionsTotal:
                transitionsTotal[prevprevtag] = defaultdict(int)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissionsTotal[tag] += 1

            transitions[prevprevtag][prevtag][tag] += 1
            transitionsTotal[prevprevtag][prevtag] += 1

            prevprevtag = prevtag
            prevtag = tag

        # don't forget the stop probability for each sentence

        # prevprevtag is the second last, prev is the last one
        if prevprevtag not in transitions:
            transitions[prevprevtag] = defaultdict(int)

        if prevtag not in transitions[prevprevtag]:
            transitions[prevprevtag][prevtag] = defaultdict(int)

        if prevprevtag not in transitionsTotal:
            transitionsTotal[prevprevtag] = defaultdict(int)

        transitions[prevprevtag][prevtag][FINAL_STATE] += 1
        transitionsTotal[prevprevtag][prevtag] += 1

        # prevprev as the last one
        if prevprevtag not in transitions:
            transitions[prevprevtag] = defaultdict(dict)

        if FINAL_STATE not in transitions[prevprevtag]:
            transitions[prevprevtag][FINAL_STATE] = defaultdict(int)

        if prevprevtag not in transitionsTotal:
            transitionsTotal[prevprevtag] = defaultdict(int)

        transitions[prevprevtag][FINAL_STATE][FINAL_STATE] += 1
        transitionsTotal[prevprevtag][FINAL_STATE] += 1

for prevprevtag in transitions:
    for prevtag in transitions[prevprevtag]:
        for tag in transitions[prevprevtag][prevtag]:
            print "trans %s %s %s %s" % (prevprevtag, prevtag, tag, float(
                transitions[prevprevtag][prevtag][tag]) / transitionsTotal[prevprevtag][prevtag])

for tag in emissions:
    for token in emissions[tag]:
        print "emit %s %s %s " % (tag, token, float(
            emissions[tag][token]) / emissionsTotal[tag])
