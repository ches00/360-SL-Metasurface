import sys

from dataset.Generator import Generator
from utils.ArgParser import Argument

if __name__ == "__main__":

	parser = Argument()
	args = parser.parse()

	generator = Generator(args)
	generator.gen()

	
