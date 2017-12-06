import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
import math
from modelCNN import NMT

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
parser.add_argument("--model_file", default="dump",
                    help="Location to dump the models.")
parser.add_argument("--load_dump", required=False,
                    help="Location to load pretrained models.")
parser.add_argument("--batch_size", default=50, type=int,
                    help="Batch size for training. (default=50)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--learning_rate", "-lr", default=0.001, type=float,
                    help="Learning rate of the optimization. (default=0.001)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")

# feel free to add more arguments as you need

def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  if options.gpuid:
    cuda.set_device(options.gpuid[0])

  src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))
  new_src_train = []
  new_trg_train = []
  for src_sent, trg_sent in zip(src_train, trg_train):
      new_src_train.append(src_sent)
      new_trg_train.append(trg_sent)
  src_train = new_src_train
  trg_train = new_trg_train

  batched_train_src, batched_train_src_mask, sort_index = utils.tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
  batched_train_trg, batched_train_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
  batched_dev_src, batched_dev_src_mask, sort_index = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])
  batched_dev_trg, batched_dev_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_dev, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
  
  trg_vocab_size = len(trg_vocab)
  src_vocab_size = len(src_vocab)

  opts = {'src_voc_size' : len(src_vocab), 'trg_voc_size' : len(trg_vocab), 'dim_rnn' : 512, 'dim_emb' : 300, 'dropout' : 0.2}
  nmt = NMT(opts)

  if options.load_dump:
    nmt.load_param(options.load_dump)
  torch.save(nmt, open(options.model_file + ".init", 'wb'), pickle_module=dill)
  
  if use_cuda:
    nmt.cuda()
  else:
    nmt.cpu()

  criterion = torch.nn.NLLLoss()
  if options.optimizer == 'Adam':
    optimizer = torch.optim.Adam(params=nmt.parameters(), lr=options.learning_rate, eps=1e-3)
  else:
    optimizer = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)

  src_sent = Variable(src_test[0][:, None]) 
  last_dev_avg_loss = float("inf")
  if use_cuda:
    src_sent = src_sent.cuda()
    
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))
    
    # srange generates a lazy sequence of shuffled range
    nmt.train()
    max_train_batch = len(batched_train_src)
    for i, batch_i in enumerate(utils.rand.srange(len(batched_train_src))):
      #print(i)
      train_src_batch = Variable(batched_train_src[batch_i])  # of size (src_seq_len, batch_size)
      train_trg_batch = Variable(batched_train_trg[batch_i])  # of size (src_seq_len, batch_size)
      train_src_mask = Variable(batched_train_src_mask[batch_i])
      train_trg_mask = Variable(batched_train_trg_mask[batch_i])

      if use_cuda:
        train_src_batch = train_src_batch.cuda()
        train_trg_batch = train_trg_batch.cuda()
        train_src_mask = train_src_mask.cuda()
        train_trg_mask = train_trg_mask.cuda()

      sys_out_batch = nmt.forward(train_src_batch, train_trg_batch, train_src_mask, train_trg_mask)  # (trg_seq_len, batch_size, trg_vocab_size)

      #print(sys_out_batch.size())
      #print(train_trg_mask.size())
      #print(sys_out_batch)

      train_trg_mask = train_trg_mask[1:].view(-1)
      train_trg_batch = train_trg_batch[1:].view(-1)
      train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
      train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size)
      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size)
      loss = criterion(sys_out_batch, train_trg_batch)
      if i % 100 == 0:
          logging.debug("loss at batch {0} / {3} ({2}): {1}".format(i, loss.data[0], batch_i, max_train_batch))
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm(nmt.parameters(), 2)
      optimizer.step()
    
    nmt.eval()
    # validation -- this is a crude esitmation because there might be some paddings at the end
    dev_loss = 0.0
    for batch_i in range(len(batched_dev_src)):
      dev_src_batch = Variable(batched_dev_src[batch_i], volatile=True)
      dev_trg_batch = Variable(batched_dev_trg[batch_i], volatile=True)
      dev_src_mask = Variable(batched_dev_src_mask[batch_i], volatile=True)
      dev_trg_mask = Variable(batched_dev_trg_mask[batch_i], volatile=True)
      if use_cuda:
        dev_src_batch = dev_src_batch.cuda()
        dev_trg_batch = dev_trg_batch.cuda()
        dev_src_mask = dev_src_mask.cuda()
        dev_trg_mask = dev_trg_mask.cuda()

      sys_out_batch = nmt.forward(dev_src_batch, dev_trg_batch, dev_src_mask, dev_trg_mask)  # (trg_seq_len, batch_size, trg_vocab_size) 
      dev_trg_mask = dev_trg_mask[1:].view(-1)
      dev_trg_batch = dev_trg_batch[1:].view(-1)
      dev_trg_batch = dev_trg_batch.masked_select(dev_trg_mask)
      dev_trg_mask = dev_trg_mask.unsqueeze(1).expand(len(dev_trg_mask), trg_vocab_size)
      sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
      sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size)
      loss = criterion(sys_out_batch, dev_trg_batch)
      #logging.debug("dev loss at batch {0}: {1}".format(batch_i, loss.data[0]))
      dev_loss += loss
    dev_avg_loss = dev_loss / len(batched_dev_src)
    logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss.data[0], epoch_i))
    

    #if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
    #  logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
    #  break
    torch.save(nmt, open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'), pickle_module=dill)
    last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
