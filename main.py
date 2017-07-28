"""
Lisanne Wiengarten
Matriculation no. 3249897
Statistical Dependency Parsing
IMS, SuSe 17
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import timeit
import pickle
import getopt

from OracleParser import OracleParser
from GoldReader import GoldReader
from Classifier import Classifier
from TestReader import TestReader
from GuideParser import GuideParser

# Usage function
def usage():
	print "Usage of " +sys.argv[0]+":"
	print "python "+sys.argv[0]+" --help shows this message"
	print "For saving a trained model: python "+sys.argv[0]+" --train trainfile.conll06 --save modelname.pik"
	print "Without the --save option, the model will be automatically saved to model.pik"
	print "For training a model: python "+sys.argv[0]+" --train trainfile.conll06"
	print "For loading a given model: python "+sys.argv[0]+" --model modelname.pik"
	print "For specifying output file name: python "+sys.argv[0]+" --output outputname.conll06"
	print "Without the --output option, the parsing results will be automatically saved to parser_output.conll06"
	print "For training & testing: python "+sys.argv[0]+" --train trainfile.conll06 --test testfile.conll06.blind"
	print "For loading & testing: python "+sys.argv[0]+" --model modelname.pik --test testfile.conll06.blind"
	print "For omitting status updates: python "+sys.argv[0]+"--quiet [other parameters]"
	

# Load a model from a pickle file
def load_model(model_name):
	with open(model_name, 'rb') as f:
		return pickle.load(f)

		
### MAIN FUNCTION ###
def main(argv):	
		
	### DEFAULT VALUES ###
	model_name = "model.pik"
	output_name = "parser_output.conll06"
	log = True

	# Parse command line arguments and parameters
	try:
		options, remainder = getopt.getopt(argv[1:], 'o:t', ['quiet','output=','model=','save=','train=','test=','help'])
	except getopt.GetoptError as err:
		# print help information and exit:
		print str(err)
		usage()
		sys.exit(2)
	
	if len(options) == 0:
		usage()
		sys.exit(2)

	for opt, arg in options:
	
		# DO NOT SHOW LOGGING INFO #
		if opt in ('-q', '--q', '-quiet', '--quiet'):
			log = False
	
		# SAVE PARSE RESULTS #
		if opt in ('-o', '--output', '--o', '-output'):
			output_name = arg
			
		# SAVE MODEL #
		elif opt in ('-s', '-save', '--s', '--save'):
			model_name = arg
			
		# LOAD MODEL #
		elif opt in ('-model', '--model'):
			classifier = load_model(arg)
			oracleparser = OracleParser()
			oracleparser.unique_feats = classifier.unique_feats
			
		### TRAINING ###
		elif opt in ('--train', '-train'):
			start = timeit.default_timer()
			sents = 0
			
			# Read in train data
			goldstandard = GoldReader(arg)
	
			# Parse all sentences in goldstandard
			oracleparser = OracleParser()
			for s in goldstandard.goldlist:
				oracleparser.parse_sentence(s)
				if log:
					stop = timeit.default_timer()
					sents += 1
					print "Trained on another sentence", sents, stop - start
		
			# Train the classifier
			classifier = Classifier(oracleparser.extracted_feats, oracleparser.unique_feats, 10)
			classifier.train()
			if log:
				stop = timeit.default_timer()
				print "Training completed ", stop - start
			
			classifier.save_model(model_name)
			if log:
				stop = timeit.default_timer()
				print "Model saved ", stop - start
				
			
		### TESTING ###
		elif opt in ('-test', '--test'):
			start = timeit.default_timer()
			sents = 0
			
			# Use the classifier to test new data
			testsents = TestReader(arg)
			parser = GuideParser(classifier, oracleparser)
			
			# Parse all sentences and write results to file
			with open(output_name, 'w') as f:
				for s in testsents.sentlist:
					parsed = parser.parse_sentence(s)
					f.write(parsed.write()+"\n")
					if log:
						stop = timeit.default_timer()
						sents += 1
						print "Parsed another sentence", sents, stop - start
						print "LAs: ", parsed.leftarcs
						print "RAs: ", parsed.rightarcs
				
		# HELP FUNCTION #
		elif opt in ('-help', '--help', '-h', '--h', '-usage', '--usage'):
			usage()
			sys.exit(2)



		
if __name__ == "__main__":
	main(sys.argv)