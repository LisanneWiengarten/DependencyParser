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
	print "For training a model: python "+sys.argv[0]+" --train trainfile.conll06"
	print "For saving a trained model: python "+sys.argv[0]+" --train trainfile.conll06 --save modelname.pik"
	print "Without the --save option, the model will be automatically saved to model.pik"
	print "For loading a given model: python "+sys.argv[0]+" --model modelname.pik"
	print "For training & testing: python "+sys.argv[0]+" --train trainfile.conll06 --test testfile.conll06.blind"
	print "For loading & testing: python "+sys.argv[0]+" --model modelname.pik --test testfile.conll06.blind"
	print "For specifying output file name: python "+sys.argv[0]+" --output outputname.conll06"
	print "Without the --output option, the parsing results will be automatically saved to parser_output.conll06"
	

# Load a model from a pickle file
def load_model(model_name):
	with open(model_name, 'rb') as f:
		return pickle.load(f)

def main(argv):
	start = timeit.default_timer()	
		
	### DEFAULT VALUES ###
	model_name = "model.pik"
	output_name = "parser_output.conll06"

	# Parse command line arguments and parameters
	try:
		options, remainder = getopt.getopt(argv[1:], 'o:t', ['output=','model=','save=','train=','test=','help'])
	except getopt.GetoptError as err:
		# print help information and exit:
		print str(err)  # will print something like "option -a not recognized"
		usage()
		sys.exit(2)
	
	if len(options) == 0:
		usage()
		sys.exit(2)

	for opt, arg in options:
	
		# SAVE PARSE RESULTS #
		if opt in ('-o', '--output', '--o', '-output'):
			output_name = arg
			
		# SAVE MODEL #
		elif opt in ('-s', '-save', '--s', '--save'):
			model_name = arg
			
		# LOAD MODEL #
		elif opt in ('-model', '--model'):
			classifier = load_model(arg)
			
		### TRAINING ###
		elif opt in ('--train', '-train'):
			# Read in train data
			goldstandard = GoldReader(arg)
	
			# Parse all sentences in golstandard
			oracleparser = OracleParser()
			for s in goldstandard.goldlist:
				oracleparser.parse_sentence(s)
		
			# Train the classifier
			classifier = Classifier(oracleparser.raw_feats, oracleparser.unique_feats, 300, 7)
			classifier.train()
			
			classifier.save_model(model_name)
			
		### TESTING ###
		elif opt in ('-test', '--test'):
			# Use the classifier to test new data
			testsents = TestReader(arg)
			parser = GuideParser(classifier)
			# Parse all sentences and write results to file
			with open(output_name, 'w') as f:
				for s in testsents.sentlist:
					parsed = parser.parse_sentence(s)
					f.write(parsed.write()+"\n")
					f.write("LAs: "+str(parsed.leftarcs)+"\n")
					f.write("RAs: "+str(parsed.rightarcs)+"\n")
				
		# HELP FUNCTION #
		elif opt in ('-help', '--help', '-h', '--h', '-usage', '--usage'):
			usage()
			sys.exit(2)

	
	
	stop = timeit.default_timer()
	print stop - start


		
if __name__ == "__main__":
	main(sys.argv)