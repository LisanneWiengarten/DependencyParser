from Sentence import Sentence
from Sentence import Token

# Used to read in a blind conll06 file
class TestReader:
	def __init__(self, infile):
		self.filey = open(infile, "r")
		# List of all the sentences in gold standard
		self.sentlist = list()
		# Artificial root token
		self.root = Token(0, "ROOT", "ROOT", "root_pos", "_", "_", -1, "_", "_", "_\n")
		
		self.read_in()
		
	
	def read_in(self):
		tokenlist = ([Token(0, "ROOT", "ROOT", "root_pos", "_", "_", -1, "_", "_", "_\n")])
		rightarcs = set()
		leftarcs = set()
	
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
				tokenlist = ([Token(0, "ROOT", "ROOT", "root_pos", "_", "_", -1, "_", "_", "_\n")])
				rightarcs = set()
				leftarcs = set()	
		
	
	# Prints out all sentences found during training
	def print_sentences(self):
		for s in self.sentlist:
			print s.write()
			
	# Writes all the sentences in the goldstandard to a given file
	def write_to_file(self, outfile):
		for s in self.sentlist:
			outfile.write(s.write()+"\n")
