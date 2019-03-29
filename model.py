import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
                 vocab_size=1,
                 embedding_dim=100,
                 hidden_size=1,
                 num_layers=1,
                 dropout=0.5
                 ):
        super(Model, self).__init__()

        # ========= attribute =========

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ============== net =========

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=self.hidden_size*2,
            num_layers=num_layers,
            dropout=dropout
        )

        self.hidden_layer = nn.Linear(
            in_features=self.hidden_size*2,
            out_features=self.hidden_size
        )

        self.output_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=vocab_size
        )

    def forward(self, x, hidden):

        seq_len, batch_size = x.size()  # input (seq_len,batch_size)

        '''
        Embedding layer
        in: (seq_len,batch_size)  -> [25,32]
        out: (seq_len,batch_size,embeding_dim) -> [25,32,emb_dim]
        '''

        embedded = self.embedding_layer(x)

        '''
        GRU layer
        in shape: (seq_len, batch, input_size[embedding_size])
        out shape: (seq_len, batch, hidden_dim)  -> [25,32,128*2]
        '''

        out, hidden = self.gru(embedded, hidden)
        out = nn.functional.relu(out)

        '''
        hidden layer
        '''
        out = out.view(seq_len*batch_size, -1)
        out = self.hidden_layer(out)
        out = nn.functional.relu(out)

        '''
        output layer
        in: (seq_len*batch_size,hidden_dim)
        out: (seq_len*batch_size,vocab_size)
        '''

        out = self.output_layer(out)

        return out, hidden

    def init_hidden(self, batch_size):
        '''
        h_0 of shape (num_layers * num_directions, batch, hidden_size)
        '''

        return torch.zeros(self.num_layers, batch_size, self.hidden_size*2)
