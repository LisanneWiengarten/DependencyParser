"""
Lisanne Wiengarten
Matriculation no. 3249897
Statistical Dependency Parsing
IMS, SuSe 17
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Class for Tokens #
# One token consists of id, form, lemma, pos, morph1, morph2, head, relation, empty1, empty2, leftmost- and rightmost dependent
class Token:
	def __init__(self, id_in, form_in, lemma_in, pos_in, morph1_in, morph2_in, head_in, rel_in, empty1_in, empty2_in):
		self.id = int(id_in)
		self.form = form_in
		self.lemma = lemma_in
		self.pos = pos_in
		self.morph1 = morph1_in
		self.morph2 = morph2_in
		self.head = head_in
		self.rel = rel_in
		self.empty1 = empty1_in
		self.empty2 = empty2_in
		self.ld = -1
		self.rd = -1

	
	# Prettily write a token to string
	def write(self):
		return str(self.id) + "\t" + self.form + "\t" + self.lemma + "\t" + self.pos + "\t" + self.morph1 + "\t" + self.morph2 + "\t" + str(self.head) + "\t" + self.rel + "\t" + self.empty1 + "\t" + self.empty2



		
# Class for Sentences # 
# A sentence is basically a list of sentences, plus sets for leftarcs and rightarcs
class Sentence:
	def __init__(self, words):
		self.tokenlist = words
		self.rightarcs = ()
		self.leftarcs = ()
		
	def set_rightarcs(self, ras):
		self.rightarcs = ras
		
	def set_leftarcs(self, las):
		self.leftarcs = las
		
	# While reading in the goldstandard, the leftmost- and rightmost dependents are set
	# because they are needed for training
	def set_dependents(self):
		self.leftmost_dependents()
		self.rightmost_dependents()
	
	# Searches and sets the leftmost dependent for each token in the sentence #
	# If there is none, the parameter is set to -1
	def leftmost_dependents(self):
		# My leftmost dependent is the smallest number that has me as head
		for current in self.tokenlist:
			ld = len(self.tokenlist)+2
			for other in self.tokenlist:
				if current.id == int(other.head) and other.id < ld:
					ld = other.id
			if ld < len(self.tokenlist)+2:
				current.ld = ld
			else:
				current.ld = -1
				
	# Searches and sets the rightmost dependent for each token in the sentence #
	# If there is none, the parameter is set to -1
	def rightmost_dependents(self):
		# My rightmost dependent is the biggest number that has me as head
		for current in self.tokenlist:
			rd = -2
			for other in self.tokenlist:
				if current.id == int(other.head) and other.id > rd:
					rd = other.id
			if rd > -2:
				current.rd = rd
			else:
				current.rd = -1
			
	# Returns the token with the given id #
	def get_token_by_id(self, id):
		for token in self.tokenlist:
			if token.id == id:
				return token
				
		return Token(-1, "NAN", "NAN", "nan_pos", "_", "_", -2, "_", "_", "_\n")
	
	# Prettily writes a sentence to string #
	# Omits the artificial root token
	def write(self):
		out = str()
		for token in self.tokenlist[1:]:
			out += token.write()
		return out

