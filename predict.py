import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
import math
from modelRNN import NMT
import sys
import os
#from example_module import NMT

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Decoding")
parser.add_argument("--data_file", default="data/hw5",
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="de",
                    help="Source Language. (default = de)")
parser.add_argument("--trg_lang", default="en",
                    help="Target Language. (default = en)")
parser.add_argument("--model", required=True,
                    help="Location to load the models.")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")

def main(options):
    use_cuda = (len(options.gpuid) >= 1)

    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
    trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

    trg_vocab_size = len(trg_vocab)
    src_vocab_size = len(src_vocab)

    nmt = torch.load(options.model)

    if use_cuda > 0:
        nmt.cuda()
    else:
        nmt.cpu()
    
    nmt.eval()
  
    os.system("rm -f {}".format(options.model + '.out'))
    for src_sent_ in src_test:
        src_sent = Variable(src_sent_[:, None]) 
        if use_cuda > 0:
            src_sent = src_sent.cuda()
        trans_sent = nmt.decode(src_sent, trg_vocab)
    
        with open(options.model+ '.out', 'a+') as f:
            f.write(trans_sent + '\n')

        sys.stderr.write(trans_sent + '\n')
        


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
