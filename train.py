import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import NMT

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
parser.add_argument("--optimizer", default="Adadelta", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adadelta)")
parser.add_argument("--learning_rate", "-lr", default=1.0, type=float,
                    help="Learning rate of the optimization. (default=1.0)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the testelopment set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu testice to use. Empty implies cpu usage.")
# feel free to add more arguments as you need


def to_var(input, volatile=False):
    x = Variable(input, volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def main(options):

  if torch.cuda.is_available():
    cuda.set_device(1)

  src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  batched_train_src, batched_train_src_mask, sort_index = utils.tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
  batched_train_trg, batched_train_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
  batched_dev_src, batched_dev_src_mask, sort_index = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])
  batched_dev_trg, batched_dev_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_dev, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)

  src_vocab_size = len(src_vocab)
  trg_vocab_size = len(trg_vocab)

  nmt = NMT(src_vocab_size, trg_vocab_size) # TODO: add more arguments as necessary
  if torch.cuda.is_available():
    nmt.cuda()
  else:
    nmt.cpu()

  criterion = torch.nn.NLLLoss()
  optimizer = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)

  # main training loop
  last_dev_avg_loss = float("inf")
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))
    # srange generates a lazy sequence of shuffled range

    for i, batch_i in enumerate(utils.rand.srange(len(batched_train_src))):
      train_src_batch = to_var(batched_train_src[batch_i])  # of size (src_seq_len, batch_size)
      train_trg_batch = to_var(batched_train_trg[batch_i])  # of size (src_seq_len, batch_size)
      train_src_mask = to_var(batched_train_src_mask[batch_i])
      train_trg_mask = to_var(batched_train_trg_mask[batch_i])

      sys_out_batch = nmt(train_src_batch, train_trg_batch, training=True)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary
      train_trg_mask = train_trg_mask.view(-1)
      train_trg_batch = train_trg_batch.view(-1)
      train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
      train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size)
      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size)
      loss = criterion(sys_out_batch, train_trg_batch)
      if i % 1000 == 0:
        logging.debug("loss at batch {0}: {1}".format(i, loss.data[0]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0

    for batch_i in range(len(batched_dev_src)):
      dev_src_batch = to_var(batched_dev_src[batch_i], volatile=True)
      dev_trg_batch = to_var(batched_dev_trg[batch_i], volatile=True)
      dev_src_mask = to_var(batched_dev_src_mask[batch_i], volatile=True)
      dev_trg_mask = to_var(batched_dev_trg_mask[batch_i], volatile=True)

      sys_out_batch = nmt(dev_src_batch, dev_trg_batch)  # (trg_seq_len, batch_size, trg_vocab_size) # TODO: add more arguments as necessary
      dev_trg_mask = dev_trg_mask.view(-1)
      dev_trg_batch = dev_trg_batch.view(-1)
      dev_trg_batch = dev_trg_batch.masked_select(dev_trg_mask)
      dev_trg_mask = dev_trg_mask.unsqueeze(1).expand(len(dev_trg_mask), trg_vocab_size)
      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size)
      #sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size)
      loss = criterion(sys_out_batch, dev_trg_batch)
      if batch_i % 1000 == 0:
        logging.debug("dev loss at batch {0}: {1}".format(batch_i, loss.data[0]))
      dev_loss += loss
    dev_avg_loss = dev_loss / len(batched_dev_src)
    logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))

    # if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
      # logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
      # break
    torch.save(nmt, open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
