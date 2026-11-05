import sys

from classification_model import main


if __name__ == '__main__':
    main(['--model', 'neural_network', '--device', 'gpu', *sys.argv[1:]])
