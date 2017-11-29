import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import NMT
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
parser.add_argument("--data_file", default="data/hw5",
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="de",
                    help="Source Language. (default = de)")
parser.add_argument("--trg_lang", default="en",
                    help="Target Language. (default = en)")
parser.add_argument("--model_file", default="model.py",
                    help="Location to dump the models.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=SGD)")
parser.add_argument("--learning_rate", "-lr", default=0.1, type=float,
                    help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the testelopment set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu testice to use. Empty implies cpu usage.")
parser.add_argument("--modelname", default="model.py.nll_0.68.epoch_18")
parser.add_argument("--output_file", default="output.txt")
# feel free to add more arguments as you need


def to_var(input, volatile=True):
    x = Variable(input, volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def main(options):

  _, _, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  _, _, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  src_vocab_size = len(src_vocab)
  trg_vocab_size = len(trg_vocab)

  nmt = NMT(src_vocab_size, trg_vocab_size)
  nmt = torch.load(open(options.modelname, 'rb'))
  
  nmt.eval()

  if torch.cuda.is_available():
    nmt.cuda()
  else:
    nmt.cpu()

  with open(options.output_file, 'w') as f_write:
    for i in range(len(src_test)):
      src = to_var(torch.unsqueeze(src_test[i],1), volatile=True)
      trg = to_var(torch.unsqueeze(trg_test[i],1), volatile=True)

      output = nmt(src)
      s = ""
      for ix in output[1:-1]:
        idx = ix.data[0]
        #if idx == 2: # if <s>, don't write it
        #  continue
        #if idx == 3: # if </s>, end the loop
        #  break
        s += trg_vocab.itos[idx] + " "
        
      s += '\n'
      f_write.write(s.encode('utf-8'))

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
