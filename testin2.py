import argparse
parser = argparse.ArgumentParser(description='Example of libmelee in action')

parser.add_argument('--logging','-l', action='store_true',
                    help='Logging of Gamestates')
parser.add_argument('--twoagents','-2',action ='store_true')
args = parser.parse_args()
print(args)