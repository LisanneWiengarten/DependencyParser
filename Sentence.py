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
		self.ld = 9999
		self.rd = -2

	
	# Prettily write a token to string
	def write(self):
		return str(self.id) + "\t" + self.form + "\t" + self.lemma + "\t" + self.pos + "\t" + self.morph1 + "\t" + self.morph2 + "\t" + str(self.head) + "\t" + self.rel + "\t" + self.empty1 + "\t" + self.empty2



		
# Class for Sentences # 
# A sentence is basically a list of words, plus sets for leftarcs and rightarcs
class Sentence:
	def __init__(self, words):
		self.tokenlist = words
		self.rightarcs = ()
		self.leftarcs = ()
		
	def set_rightarcs(self, ras):
		self.rightarcs = ras
		
	def set_leftarcs(self, las):
		self.leftarcs = las
			
	# Returns the token with the given id #
	def get_token_by_id(self, id):
		for token in self.tokenlist:
			if str(token.id) == str(id):
				return token
		
		return Token(-1, "NAN", "NAN", "NAN", "_", "_", -2, "_", "_", "_")
	
	# Prettily writes a sentence to string #
	# Omits the artificial root token
	def write(self):
		out = str()
		for token in self.tokenlist[1:]:
			out += token.write()
		return out

