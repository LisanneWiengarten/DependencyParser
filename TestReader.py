"""
Lisanne Wiengarten
Matriculation no. 3249897
Statistical Dependency Parsing
IMS, SuSe 17
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Sentence import Sentence
from Sentence import Token

# Used to read in a blind conll06 file #
# Stores the file as instances of the Sentence class
class TestReader:
	def __init__(self, infile):
		self.filey = open(infile, "r")
		# List of all the found sentences
		self.sentlist = list()
		# Artificial root token
		self.root = Token(0, "ROOT", "ROOT", "root_pos", "_", "_", -1, "_", "_", "_")
		
		self.read_in()
		
	
	# Actual process of reading the file #
	def read_in(self):
		tokenlist = ([Token(0, "ROOT", "ROOT", "root_pos", "_", "_", -1, "_", "_", "_")])
	
		# Get every sentence in input file
		for line in self.filey:
			splitted = line.split("\t")
			
			# If we have found a 'normal' line, add a new token with the info found
			if len(splitted) == 10:
				token = Token(splitted[0], splitted[1],splitted[2],splitted[3],splitted[4],splitted[5],splitted[6],splitted[7],splitted[8],splitted[9])
				# And add the new token to the list of tokens
				tokenlist.append(token)
			
			# Otherwise, we have reached the end of a sentence (blank line)
			else:
				# Create a new sentence from the info we have collected
				sentence = Sentence(tokenlist)
				self.sentlist.append(sentence)
				
				# Clean everything for the next sentence
				tokenlist = ([Token(0, "ROOT", "ROOT", "root_pos", "_", "_", -1, "_", "_", "_")])	
		
	
	# Prints out all sentences found during training #
	def print_sentences(self):
		for s in self.sentlist:
			print s.write()
			
			
	# Writes all the found sentences to a given file #
	def write_to_file(self, outfile):
		for s in self.sentlist:
			outfile.write(s.write()+"\n")
