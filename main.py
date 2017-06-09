import sys
from Oracleparser import Oracleparser
from Reader import Reader
from Classifier import Classifier

def main(infile):
	goldstandard = Reader(infile)
	goldstandard.print_goldstandard()
	#outfile = open("train_out.conll06", "w")
	#goldstandard.write_to_file(outfile)
	
	parser = Oracleparser()
	for s in goldstandard.goldlist:
		parser.parse_sentence(s)
		
	classifier = Classifier(parser.raw_feats, parser.unique_feats)


		
if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Error: Train file needed"
	else:
		main(sys.argv[1])