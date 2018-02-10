hidden_size = 300
input_size = vocabularySize
output_size = vocabularySize


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        

    def forward(self, inputs, hidden,state):
        output,(hidden,state) = self.lstm(inputs,(hidden,state))
        output = self.out(output)
        return output,hidden,state

    def initHidden(self):
        h = Variable(torch.zeros(1, 1, self.hidden_size))
        c = Variable(torch.zeros(1, 1, self.hidden_size))
        return h.cuda(), c.cuda()


decoder = DecoderLSTM(hidden_size, input_size, output_size) 
decoder.cuda()
decoder.train()

decoder