# class for tokens 
class Token:
	def __init__(self, id_in, form_in, lemma_in, pos_in, morph1_in, morph2_in, head_in, rel_in, empty1_in, empty2_in):
		self.id = int(id_in)
		self.form = form_in
		self.lemma = lemma_in
		self.pos = pos_in
		self.morph1 = morph1_in
		self.morph2 = morph2_in
		self.head = int(head_in)
		self.rel = rel_in
		self.empty1 = empty1_in
		self.empty2 = empty2_in

	def write(self):
		return str(self.id) + "\t" + self.form + "\t" + self.lemma + "\t" + self.pos + "\t" + self.morph1 + "\t" + self.morph2 + "\t" + str(self.head) + "\t" + self.rel + "\t" + self.empty1 + "\t" + self.empty2



# class for sentences (list of tokens)
class Sentence:
	def __init__(self, words):
		self.tokenlist = words
		self.rightarcs = ()
		self.leftarcs = ()
		
	def set_rightarcs(self, ras):
		self.rightarcs = ras
		
	def set_leftarcs(self, las):
		self.leftarcs = las

	def write(self):
		out = str()
		for token in self.tokenlist:
			out += token.write()
		return out

