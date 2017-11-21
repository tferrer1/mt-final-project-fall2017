import torch
import torch.nn as nn
import dill
from torch.autograd import Variable

def to_var(input, volatile=False):
    x = Variable(input, volatile=volatile)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class ATTN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ATTN, self).__init__()
        self.Wi = nn.Linear(in_dim, in_dim, bias=False)
        self.Wo = nn.Linear(out_dim, in_dim, bias=False)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
    def forward(self, input, h):
        semi = torch.unsqueeze(self.Wi(h), 0).expand_as(input)
        score = torch.t(self.softmax(torch.t(torch.sum(input * semi, dim=2)))) # erased torch.t() after softmax
        score = torch.unsqueeze(score.contiguous(),2)
        s_tilde = torch.sum(score * input, dim=0)
        c_t = self.tanh(self.Wo(torch.cat([s_tilde, h], dim=1)))
        return c_t

class NMT(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super(NMT, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.ENC_esize = 30
        self.DEC_esize = 30
        self.ENC_hsize = 64
        self.DEC_hsize = 128
        # encoding embedding
        self.EEMB = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=self.ENC_esize)
        # encoding
        self.ENC = nn.LSTM(input_size=self.ENC_esize, hidden_size=self.ENC_hsize, bidirectional=True)
        # attention
        self.ATTN = ATTN(2*self.ENC_hsize, 2*self.DEC_hsize)
        # decoding
        #self.DEC = nn.LSTM(input_size=1324, hidden_size=1024)
        self.DEC = nn.LSTMCell(input_size=(self.DEC_hsize+self.DEC_esize), hidden_size=self.DEC_hsize)
        # decoding embedding
        self.DEMB = nn.Embedding(num_embeddings=trg_vocab_size, embedding_dim=self.DEC_esize)
        # generator
        self.GEN = nn.Linear(in_features=self.DEC_hsize, out_features=trg_vocab_size)
        # miscellaneous
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input_src_batch, input_trg_batch, training=False):
        sent_len = input_trg_batch.size()[0]

        encoder_input = self.EEMB(input_src_batch)
        encoder_output, (hidden, context) = self.ENC(encoder_input)

        batch_size = encoder_output.size()[1]

        hidden = hidden.permute(1,2,0).contiguous().view(batch_size, self.ENC_hsize*2)
        context = context.permute(1,2,0).contiguous().view(batch_size, self.ENC_hsize*2)

        output = to_var(torch.zeros(sent_len, batch_size, self.trg_vocab_size).fill_(-1))
        output[0,:,2] = 0

        word = to_var(torch.LongTensor(batch_size).fill_(2)) #

        for i in xrange(1, sent_len):
            c_t = self.ATTN(encoder_output, hidden)

            if training:
                decoder_input = torch.cat([c_t, self.DEMB(input_trg_batch[i-1])], dim=1)
            else:
                decoder_input = torch.cat([c_t, self.DEMB(word)], dim=1)

            (hidden, context) = self.DEC(decoder_input, (hidden, context))


            word = self.logsoftmax(self.GEN(hidden))
            output[i] = word
            if not training:
                _, word = torch.max(word, dim=1)

        return output
