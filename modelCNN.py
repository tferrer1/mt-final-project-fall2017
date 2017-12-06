import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

class WeightNormConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, init_scale=1.,
                 polyak_decay=0.9995):
        super(WeightNormConv2d, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding,
                                               dilation, groups)

        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            # out_channels, in_channels // groups, * kernel_size
            self.V.data.copy_(torch.randn(self.V.data.size()
                                          ).type_as(self.V.data) * 0.05)
            V_norm = self.V.data / self.V.data.view(self.out_channels, -1)\
                .norm(2, 1).view(self.out_channels, *(
                    [1] * (len(self.kernel_size) + 1))).expand_as(self.V.data)
            x_init = F.conv2d(x, Variable(V_norm), None, self.stride,
                              self.padding, self.dilation, self.groups).data
            t_x_init = x_init.transpose(0, 1).contiguous().view(
                self.out_channels, -1)
            m_init, v_init = t_x_init.mean(1).squeeze(
                1), t_x_init.var(1).squeeze(1)
            # out_features
            scale_init = self.init_scale / \
                torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            m_init_shape = m_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            x_init = scale_init_shape.expand_as(
                x_init) * (x_init - m_init_shape.expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            V, g, b = get_vars_maybe_avg(
                self, ['V', 'g', 'b'], self.training,
                polyak_decay=self.polyak_decay)

            scalar = torch.norm(V.view(self.out_channels, -1), 2, 1)
            if len(scalar.size()) == 2:
                scalar = g / scalar.squeeze(1)
            else:
                scalar = g / scalar

            W = scalar.view(self.out_channels, *
                            ([1] * (len(V.size()) - 1))).expand_as(V) * V

            x = F.conv2d(x, W, b, self.stride,
                         self.padding, self.dilation, self.groups)
            return x

SCALE_WEIGHT = 0.5 ** 0.5


def shape_transform(x):
    """ Tranform the size of the tensors to fit for conv input. """
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class GatedConv(nn.Module):
    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WeightNormConv2d(input_size, 2 * input_size,
                                     kernel_size=(width, 1), stride=(1, 1),
                                     padding=(width // 2 * (1 - nopad), 0))
        init.xavier_uniform(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var, hidden=None):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * F.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    def __init__(self, num_layers, input_size, cnn_kernel_width=3,
                 dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x, hidden=None):
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x

class EncoderBase(nn.Module):
    """
    EncoderBase class for sharing code among various encoder.
    """
    def _check_args(self, input, lengths=None, hidden=None):
        print(input)
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        raise NotImplementedError

class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))
            

class CNNEncoder(EncoderBase):
    """
    Encoder built on CNN.
    """
    def __init__(self, num_layers, hidden_size,
                 cnn_kernel_width, dropout, embeddings):
        super(CNNEncoder, self).__init__()

        self.embeddings = embeddings
        input_size = embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size,
                              cnn_kernel_width, dropout)

    def forward(self, input_batch):
        """ See EncoderBase.forward() for description of args and returns."""
        #self._check_args(input)

        emb = self.embeddings(input_batch)
        s_len, batch, emb_dim = emb.size()

        emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return emb_remap.squeeze(3).transpose(0, 1).contiguous(),\
            out.squeeze(3).transpose(0, 1).contiguous()

def seq_linear(linear, x):
    # linear transform for 3-d tensor
    batch, hidden_size, length, _ = x.size()
    h = linear(torch.transpose(x, 1, 2).contiguous().view(
        batch * length, hidden_size))
    return torch.transpose(h.view(batch, length, hidden_size, 1), 1, 2)


class ConvMultiStepAttention(nn.Module):
    def __init__(self, input_size):
        super(ConvMultiStepAttention, self).__init__()
        self.linear_in = nn.Linear(input_size, input_size)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, base_target_emb, input, encoder_out_top,
                encoder_out_combine):
        """
        It's like Luong Attetion.
        Conv attention takes a key matrix, a value matrix and a query vector.
        Attention weight is calculated by key matrix with the query vector
        and sum on the value matrix. And the same operation is applied
        in each decode conv layer.
        Args:
            base_target_emb: target emb tensor
            input: output of decode conv
            encoder_out_t: the key matrix for calculation of attetion weight,
                which is the top output of encode conv
            encoder_out_c: the value matrix for the attention-weighted sum,
                which is the combination of base emb and top output of encode
        """
        # checks
        batch, channel, height, width = base_target_emb.size()
        batch_, channel_, height_, width_ = input.size()
        aeq(batch, batch_)
        aeq(height, height_)

        enc_batch, enc_channel, enc_height = encoder_out_top.size()
        enc_batch_, enc_channel_, enc_height_ = encoder_out_combine.size()

        aeq(enc_batch, enc_batch_)
        aeq(enc_height, enc_height_)

        preatt = seq_linear(self.linear_in, input)
        target = (base_target_emb + preatt) * SCALE_WEIGHT
        target = torch.squeeze(target, 3)
        target = torch.transpose(target, 1, 2)
        pre_attn = torch.bmm(target, encoder_out_top)

        if self.mask is not None:
            pre_attn.data.masked_fill_(self.mask, -float('inf'))

        pre_attn = pre_attn.transpose(0, 2)
        attn = F.softmax(pre_attn)
        attn = attn.transpose(0, 2).contiguous()
        context_output = torch.bmm(
            attn, torch.transpose(encoder_out_combine, 1, 2))
        context_output = torch.transpose(
            torch.unsqueeze(context_output, 3), 1, 2)
        return context_output, attn

class CNNDecoder(nn.Module):
    """
    Decoder built on CNN, which consists of resduial convolutional layers,
    with ConvMultiStepAttention.
    """
    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, cnn_kernel_width, dropout, embeddings):
        super(CNNDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'cnn'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn_kernel_width = cnn_kernel_width
        self.embeddings = embeddings
        self.dropout = dropout

        # Build the CNN.
        input_size = self.embeddings.embedding_size
        self.linear = nn.Linear(input_size, self.hidden_size)
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv_layers.append(
                GatedConv(self.hidden_size, self.cnn_kernel_width,
                          self.dropout, True))

        self.attn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.attn_layers.append(
                ConvMultiStepAttention(self.hidden_size))

        self._copy = False

    def forward(self, input, context, state):
        """
        Forward through the CNNDecoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        CNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder CNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # CHECKS
        assert isinstance(state, CNNDecoderState)
        input_len, input_batch = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END CHECKS

        if state.previous_input is not None:
            input = torch.cat([state.previous_input, input], 0)

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        assert not self._copy, "Copy mechanism not yet tested in conv2conv"
        if self._copy:
            attns["copy"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        tgt_emb = emb.transpose(0, 1).contiguous()
        # The output of CNNEncoder.
        src_context_t = context.transpose(0, 1).contiguous()
        # The combination of output of CNNEncoder and source embeddings.
        src_context_c = state.init_src.transpose(0, 1).contiguous()

        # Run the forward pass of the CNNDecoder.
        emb_reshape = tgt_emb.contiguous().view(
            tgt_emb.size(0) * tgt_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(tgt_emb.size(0), tgt_emb.size(1), -1)
        x = shape_transform(x)

        pad = Variable(torch.zeros(x.size(0), x.size(1),
                                   self.cnn_kernel_width - 1, 1))
        pad = pad.type_as(x)
        base_target_emb = x

        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out,
                                src_context_t, src_context_c)
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT
        output = x.squeeze(3).transpose(1, 2)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        if state.previous_input is not None:
            outputs = outputs[state.previous_input.size(0):]
            attn = attn[:, state.previous_input.size(0):].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state.update_state(input)

        return outputs, state, attns

    def init_decoder_state(self, src, context, enc_hidden):
        return CNNDecoderState(context, enc_hidden)


class CNNDecoderState(DecoderState):
    def __init__(self, context, enc_hidden):
        self.init_src = (context + enc_hidden) * SCALE_WEIGHT
        self.previous_input = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input,)

    def update_state(self, input):
        """ Called for every decoder forward pass. """
        self.previous_input = input

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.init_src = Variable(
            self.init_src.data.repeat(1, beam_size, 1), volatile=True)
            
    
class EMB(nn.Module):
    def __init__(self, voc_size, dim_emb, dropout):
        super(EMB, self).__init__()
        self.look_up = nn.Embedding(num_embeddings=voc_size, embedding_dim=dim_emb)
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_size = dim_emb
    
    def forward(self, batch_seq):
        return self.dropout(self.look_up(batch_seq))
        #return self.look_up(batch_seq)

class ENC(nn.Module):
    def __init__(self, opts):
        super(ENC, self).__init__()
        self.embedding = EMB(opts['src_voc_size'], opts['dim_emb'], opts['dropout'])
        self.bi_lstm = nn.LSTM(input_size=opts['dim_emb'], hidden_size=opts['dim_rnn'], dropout=opts['dropout'], bidirectional=True)

    def forward(self, src_batch):
        seq_emb = self.embedding(src_batch)       
        seq_context, final_states = self.bi_lstm(seq_emb)
        return seq_context, final_states

class ATTN(nn.Module):
    def __init__(self, opts):
        super(ATTN, self).__init__()
        self.dim_rnn = opts['dim_rnn']

        self.linear_in = nn.Linear(2 * self.dim_rnn, 2 * self.dim_rnn, bias=False)
        self.linear_out = nn.Linear(4 * self.dim_rnn, 2 * self.dim_rnn, bias=False)
        self.softmax= nn.Softmax()
        self.tanh = nn.Tanh()
    
    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)

        return torch.bmm(h_t, h_s_)

    def forward(self, trg_hidden,context, src_mask):
        context = context.transpose(0, 1)
        trg_hidden = trg_hidden.unsqueeze(0).transpose(0, 1)
        
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = trg_hidden.size()

        align = self.score(trg_hidden, context)
        
        if src_mask is not None:
            src_mask = src_mask.transpose(0, 1)
            mask_ = src_mask.contiguous().view(batch, 1, sourceL)  # make it broardcastable
            align.data.masked_fill_(1 - mask_.data, - 1.0 * 1e6)
            
        align_vectors = self.softmax(align.view(batch * targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)
        c = torch.bmm(align_vectors, context)
        concat_c = torch.cat([c, trg_hidden], dim=2).view(batch * targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        attn_h = self.tanh(attn_h)
        return attn_h.squeeze(1)

class GEN(nn.Module):
    def __init__(self, opts):
        super(GEN, self).__init__()
        self.generate_linear = nn.Linear(2 * opts['dim_rnn'], opts['trg_voc_size'])

    def forward(self, input_tensor):
        assert len(input_tensor.size()) == 2
        return nn.functional.log_softmax(self.generate_linear(input_tensor))

class DEC(nn.Module):
    def __init__(self, opts):
        super(DEC, self).__init__()
        
        self.embedding = EMB(opts['trg_voc_size'],opts['dim_emb'],opts['dropout'])
        self.attn_layer = ATTN(opts)
        self.dim_rnn = opts['dim_rnn']
        self.lstm_cell = nn.LSTMCell(input_size=2 * opts['dim_rnn'] + opts['dim_emb'],hidden_size= 2 * opts['dim_rnn'])
        self.dropout = nn.Dropout(p=opts['dropout']) 
        
    def forward(self, seq_context, src_mask, seq_trg, final_states_encoder):
        max_len_trg = seq_trg.size(0)
        batch_size = seq_trg.size(1)

        seq_trg_emb = self.embedding(seq_trg)
        
        prev_h , prev_c = final_states_encoder
        prev_h = torch.cat(prev_h, 1)
        prev_c = torch.cat(prev_c, 1)
        output = prev_h
        decoder_output_list = []
        
        for i in range(max_len_trg - 1):
            lstm_input = torch.cat([seq_trg_emb[i],output], dim=1)
            prev_h = Variable(prev_h.data)
            prev_c = Variable(prev_c.data)
            prev_h, prev_c = self.lstm_cell(lstm_input, (prev_h, prev_c))
            
            output = self.attn_layer(prev_h, seq_context, src_mask)
            output = self.dropout(output)
            decoder_output_list.append(output.unsqueeze(0))
            
        decoder_output = torch.cat(decoder_output_list, dim=0)
        return decoder_output


def get_vars_maybe_avg(namespace, var_names, training, polyak_decay):
    # utility for retrieving polyak averaged params
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(
            namespace, vn, training, polyak_decay))
    return vars

def get_var_maybe_avg(namespace, var_name, training, polyak_decay):
    # utility for retrieving polyak averaged params
    # Update average
    v = getattr(namespace, var_name)
    v_avg = getattr(namespace, var_name + '_avg')
    v_avg -= (1 - polyak_decay) * (v_avg - v.data)

    if training:
        return v
    else:
        return Variable(v_avg)
    
class NMT(nn.Module):
    def __init__(self, opts):
        super(NMT, self).__init__()
        emb_enc = EMB(opts['src_voc_size'], opts['dim_emb'], opts['dropout'])
        emb_dec = EMB(opts['trg_voc_size'], opts['dim_emb'], opts['dropout'])
        self.dim_rnn = opts['dim_rnn']
        #self.encoder = ENC(opts)
        self.encoder = CNNEncoder(2, 1024, 5, 0.2, emb_enc)
        #self.decoder = DEC(opts)
        self.decoder = CNNDecoder(2, 1024, None, None, 5, 0.2, emb_dec)
        self.generator = GEN(opts)
    
    def forward(self, src_batch, trg_batch, src_mask, trg_mask):
        #print(src_batch)
        #print(trg_batch)
        final_states, seq_context = self.encoder(src_batch)
        #print("finished encoding")
        #print(final_states)
        #print(seq_context)
        state = self.decoder.init_decoder_state(src_batch, seq_context, final_states)
        #print("initialized decoder state")
        decoder_output, dec_states, attns = self.decoder(trg_batch, seq_context, state)
        #print("finished decoding")
	decoder_output = decoder_output[1:,:,:]
        trg_len, batch_size, decoder_dim = decoder_output.size()
        #print(decoder_output)
        seq_trg_log_prob = self.generator(decoder_output.view(trg_len * batch_size, -1)).view(trg_len, batch_size, -1)
        #print("finished generating")
        return seq_trg_log_prob
    
    def load_param(self, path):
        with open(path, 'rb') as f:
            params = torch.load(f)
        for key, value in params.iteritems():
            print(key, value.size())
            
        # encoder param
        self.encoder.embedding.look_up.weight.data = params['encoder.embeddings.emb_luts.0.weight']

        self.encoder.bi_lstm.weight_hh_l0.data = params['encoder.rnn.weight_hh_l0']
        self.encoder.bi_lstm.bias_hh_l0.data = params['encoder.rnn.bias_hh_l0']        
        self.encoder.bi_lstm.weight_ih_l0.data = params['encoder.rnn.weight_ih_l0']
        self.encoder.bi_lstm.bias_ih_l0.data = params['encoder.rnn.bias_ih_l0'] 

        self.encoder.bi_lstm.weight_hh_l0_reverse.data = params['encoder.rnn.weight_hh_l0_reverse']
        self.encoder.bi_lstm.bias_hh_l0_reverse.data = params['encoder.rnn.bias_hh_l0_reverse']        
        self.encoder.bi_lstm.weight_ih_l0_reverse.data = params['encoder.rnn.weight_ih_l0_reverse']
        self.encoder.bi_lstm.bias_ih_l0_reverse.data = params['encoder.rnn.bias_ih_l0_reverse'] 
        
        # decoder param
        self.decoder.embedding.look_up.weight.data = params['decoder.embeddings.emb_luts.0.weight']

        self.decoder.lstm_cell.weight_hh.data = params['decoder.rnn.layers.0.weight_hh']
        self.decoder.lstm_cell.bias_hh.data = params['decoder.rnn.layers.0.bias_hh']
        self.decoder.lstm_cell.weight_ih.data = params['decoder.rnn.layers.0.weight_ih']
        self.decoder.lstm_cell.bias_ih.data = params['decoder.rnn.layers.0.bias_ih']

        #self.decoder.linear_in.weight.data = params['decoder.attn.linear_in.weight'].transpose(0, 1)
        self.decoder.attn_layer.linear_in.weight.data = params['decoder.attn.linear_in.weight']
        self.decoder.attn_layer.linear_out.weight.data = params['decoder.attn.linear_out.weight']

        self.generator.generate_linear.weight.data = params['0.weight']
        self.generator.generate_linear.bias.data = params['0.bias']
    
    def decode(self, src_sent, trg_vocab, gpu=True):
        seq_context, final_states = self.encoder(src_sent)
        prev_h , prev_c = final_states
        
        # initial states
        prev_h = torch.cat([prev_h[0:prev_h.size(0):2], prev_h[1:prev_h.size(0):2]], dim=2)[0]
        prev_c = torch.cat([prev_c[0:prev_c.size(0):2], prev_c[1:prev_c.size(0):2]], dim=2)[0]

        output = Variable(seq_context.data.new(1, self.dim_rnn * 2))

        decoder_output_list = []
        log_prob_list = []

        w = Variable(torch.LongTensor(1))
        if gpu:
            w = w.cuda()
        w[0] = trg_vocab.stoi['<s>']
        w_list = []
        i = 0
        while i < 80 :
            i += 1
            emb_w = self.decoder.embedding(w)
            lstm_input = torch.cat([emb_w,output],dim=1)
            prev_h, prev_c = self.decoder.lstm_cell(lstm_input, (prev_h, prev_c))
            output = self.decoder.attn_layer(prev_h,seq_context,None)
            output = self.decoder.dropout(output)
        
            log_prob = self.generator(output)
            _, w = torch.max(log_prob, dim=1)
            if w.data[0] == trg_vocab.stoi['</s>']:
                break
            w_list.append(trg_vocab.itos[w.data[0]])
            
        return u' '.join(w_list).encode('utf-8')

