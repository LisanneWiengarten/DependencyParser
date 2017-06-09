from Sentence import Sentence
from Sentence import Token

# Used to read in a file in conll06 format that contains all dependency relations
class Reader:
	def __init__(self, infile):
		self.filey = open(infile, "r")
		# List of all the sentences in gold standard
		self.goldlist = list()
		# Artificial root token
		self.root = Token(0, "ROOT", "ROOT", "root_pos", "_", "_", -1, "_", "_", "_\n")
		
		self.read_in()
		
	
	def read_in(self):
		tokenlist = ([self.root])
		rightarcs = set()
		leftarcs = set()
	
		# Get every sentence in input file
		for line in self.filey:
			splitted = line.split("\t")
			# If we have found a 'normal' line, add a new token with the info found
			if len(splitted) == 10:
				token = Token(splitted[0], splitted[1],splitted[2],splitted[3],splitted[4],splitted[5],splitted[6],splitted[7],splitted[8],splitted[9])
				# If the id (pos 0) is left from its head (pos 6), then add an LA between them
				if splitted[0] < splitted[6]:
					leftarcs.add((int(splitted[6]), int(splitted[0])))
				else:
				# Otherwise, add an RA
					rightarcs.add((int(splitted[6]), int(splitted[0])))
					
				# If the current token has 0 as its head, add the artificial root arc to it
				if splitted[6] == 0:
					rightarcs.add((int(splitted[6]), 0))
				# And add the new token to the list of tokens
				tokenlist.append(token)
			
			# Otherwise, we have reached the end of a sentence (blank line)
			else:
				# Create a new sentence from the info we have collected
				sentence = Sentence(tokenlist)
				sentence.set_rightarcs(rightarcs)
				sentence.set_leftarcs(leftarcs)
				self.goldlist.append(sentence)
				# Clean everything for the next sentence
				tokenlist = ([self.root])
				rightarcs = set()
				leftarcs = set()
		
	
	# Prints out all sentences found during training
	def print_goldstandard(self):
		for s in self.goldlist:
			print s.write()
			
	# Writes all the sentences in the goldstandard to a given file
	def write_to_file(self, outfile):
		for s in self.goldlist:
			outfile.write(s.write()+"\n")
