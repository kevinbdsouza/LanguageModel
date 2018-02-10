hidden_size = 300
input_size = vocabularySize

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, inputs,hidden,state):
        output,(hidden,state) = self.lstm(inputs,(hidden,state))
        return output,hidden,state

    def initHidden(self):
        h = Variable(torch.zeros(1, 1, self.hidden_size))
        c = Variable(torch.zeros(1, 1, self.hidden_size))
        return h.cuda(), c.cuda()
        

encoder = EncoderLSTM(input_size, hidden_size)
encoder.cuda()
encoder.train()
encoder
