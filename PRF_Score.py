#-*- coding: utf-8 -*-
#coding=utf-8

# Perceptron word segment for Chinese sentences
# 
# File usage:
# Results score. PRF
# 
# Author:
# Meteor Yee

from __future__ import division
import sys

e = 0 # wrong words number
c = 0 # correct words number
N = 0 # gold words number

infile = sys.argv[1]
goldfile = sys.argv[2]

inpt1 = open(infile, 'rU')
inpt2 = open(goldfile, 'rU')

test_raw = []

for ind, line in enumerate(inpt1):
	if ind > 5000:
		break

	sent = []

	for word in line.decode("utf-8").split():
		sent.append(word)

	test_raw.append(sent)

gold_raw = []

for ind, line in enumerate(inpt2):
	if ind > 5000:
		break

	sent = []

	for word in line.decode("utf-8").split():
		sent.append(word)
		N += 1

	gold_raw.append(sent)

# test
# You can see the data.
'''output = open('aaa_result.txt', 'a')
output.write(str(test_raw))
output.write('\n\n\n')
output.write(str(gold_raw))'''

for i in xrange(len(gold_raw)):
	test_sent = test_raw[i]
	gold_sent = gold_raw[i]

	count = 0

	for seg in gold_sent:
		if seg in test_sent:
			c += 1
			count += 1

	e += len(test_sent) - count
	# print e, c

precision = c / (c + e)
recall = c / N
F = 2 * precision * recall / (precision + recall)
error_rate = e / N

print "Correct words: %d"%c
print "Error words: %d"%e
print "Gold words: %d"%N
print
print "precision: %f"%precision
print "recall: %f"%recall
print "F-Value: %f"%F
print "error_rate: %f"%error_rate