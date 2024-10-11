import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

'''

class ShapeEmbeddingRNN(nn.Module):
    def __init__(self, shape_vocab_size, color_vocab_size, embedding_size=16, hidden_size=16):
        super(ShapeEmbeddingRNN, self).__init__()
        
        self.shape_embedding = nn.Embedding(shape_vocab_size, hidden_size // 2)
        self.color_embedding = nn.Embedding(color_vocab_size, hidden_size // 2)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, embedding_size)
    def forward(self, x, lengths):        
        shape_indices = torch.arange(0, x.size(1), 2, device=x.device, dtype=torch.long)  
        color_indices = torch.arange(1, x.size(1), 2, device=x.device, dtype=torch.long) 

        print('shape_indices: ', shape_indices)
        print('color_indices: ', color_indices)
        print('lengths: ', lengths)
        print('x: ', x)
        
        shape_embeds = self.shape_embedding(x[:, shape_indices])  
        color_embeds = self.color_embedding(x[:, color_indices]) 

        print('shape_embeds: ', shape_embeds.shape) 
        print('color_embeds: ', color_embeds.shape) 
        
        combined_embeds = torch.stack((shape_embeds, color_embeds), dim=2).view(x.size(0), x.size(1) // 2, -1)

        print('combined_embeds: ', combined_embeds) 

        packed_embeds = rnn_utils.pack_padded_sequence(combined_embeds, lengths, batch_first=True, enforce_sorted=False)

        print()

        _, (hn, _) = self.lstm(packed_embeds) 
        
        final_embedding = self.fc(hn[-1])  

        return final_embedding
'''


class ShapeEmbeddingRNN(nn.Module):
    def __init__(self, hidden_size=16, rnn_type='gru'):
        super(ShapeEmbeddingRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Since the input is a flat sequence, the input size is 1 per number in the sequence
        self.input_size = 1
        
        # Define the type of RNN (GRU, LSTM, or simple RNN)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(self.input_size, hidden_size, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(self.input_size, hidden_size, batch_first=True)
        
        # Linear layer to map the final hidden state to the desired output size (16)
        self.fc = nn.Linear(hidden_size, 16)
    
    def forward(self, x):
        # Reshape the flat tensor from (batch_size, seq_length) to (batch_size, seq_length, 1)
        x = x.unsqueeze(-1)
        
        # Pass through the RNN
        _, hidden = self.rnn(x)  # Only interested in the final hidden state
        
        # For LSTM, the hidden state is a tuple (hidden, cell_state)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        
        # hidden is of shape (num_layers * num_directions, batch_size, hidden_size)
        hidden = hidden[-1]  # Get the last layer's hidden state
        
        # Pass the hidden state through a fully connected layer
        output = self.fc(hidden)
        
        return output


if __name__ == '__main__':
    shape_vocab_size = 4  # 4 shapes: square, circle, triangle, hexagon
    color_vocab_size = 4  # 4 colors: red, blue, green, yellow
    embedding_size = 16

    model = ShapeEmbeddingRNN(shape_vocab_size, color_vocab_size, embedding_size)

    input_data = torch.tensor([
        [0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1], 
        [1, 2, 0, 3, 2, 1, 3, 0, 1, 2, 1, 0]
    ])

    sequence_lengths = torch.tensor([6, 4])

    output_embedding = model(input_data, sequence_lengths)
    print(output_embedding.shape)
