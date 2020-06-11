# train and plot the learning curve
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

x = [100, 1000, 2500, 5000,  10000,
     15000, 20000,
     25000, 30000,
     35000]

errWord = []
errSentence = []

for num in x:
    os.system("./task1.py ptb.2-21.tgs ptb.2-21.txt " +
              str(num) + " > task1.hmm")
    os.system("./viterbi.pl task1.hmm < ptb.22.txt > task1.out")
    output = subprocess.check_output(
        ["./tag_acc.pl", "ptb.22.tgs", "task1.out"])
    lines = output.splitlines()
    line1 = lines[0].split()
    line2 = lines[1].split()

    errWord.append(float(line1[4]))
    errSentence.append(float(line2[4]))

plt.xlabel('Training Dataset Size')
plt.ylabel('Error Rate')

plt.title('Learning Curve')

plt.plot(x, errWord, label="Error Rate for Words")
plt.plot(x, errSentence, label="Error Rate for Sentences")
plt.legend(loc='upper right')

plt.savefig('task1.png')
plt.show()
