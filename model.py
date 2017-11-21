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

class EnsembleNMT(nn.Module):
    def __init__(self, models):
        super(EnsembleNMT, self).__init__()
        self.models = models

    def forward(self, input_src_batch, input_trg_batch):
        encoded = []
        hiddens = []
        for model in self.models:
            encoder_input = model.EEMB(input_src_batch)
            sys_out_batch, (h,c) = model.ENC(encoder_input)
            batch_size = sys_out_batch.size()[1]
            h = h.permute(1,2,0).contiguous().view(batch_size, 2*model.ENC_hsize)
            c = c.permute(1,2,0).contiguous().view(batch_size, 2*model.ENC_hsize)
            encoded.append(sys_out_batch)
            hiddens.append((h,c))
        sent_len = input_trg_batch.size()[0]
        batch_size = input_trg_batch.size()[1]

        results = to_var(torch.LongTensor(sent_len, batch_size))
        word = to_var(torch.LongTensor(batch_size).fill_(2))
        results[0] = word

        for i in range(1, sent_len - 1):
            total_word = []
            next_h = []
            for j, model in enumerate(self.models):
                sys_out_batch = encoded[j]
                h, c = hiddens[j]

                # TODO
                seq_len = sys_out_batch.size()[0]
                batch_size_ = sys_out_batch.size()[1]
                c_t = model.ATTN(sys_out_batch, h)
                decoder_input = torch.cat([c_t, model.DEMB(word)], dim=1)
                h, c = model.DEC(decoder_input, (h, c))
                w2 = model.logsoftmax(model.GEN(h))
                next_h.append((h,c))
                total_word.append(w2)

            w = total_word
            hidden = next_h
            total = 0
            for prob in w:
                total += prob
            total[:, 3] = torch.min(total).data[0]
            _, word = torch.max(total, dim=1)
            results[i] = word

        word = to_var(torch.LongTensor(batch_size).fill_(3))
        results[sent_len - 1] = word
        return results

