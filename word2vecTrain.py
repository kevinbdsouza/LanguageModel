sentences = train_sentences

sentences = [["<SOS>"] + word_tokenize(sentence.lower()) + ["<EOS>"] for sentence in sentences]

def word2vec(sentence):
    numberized = numberize(sentence)
    
    w2v_embedded = w2v_embeddings[numberized]
    
    return w2v_embedded

print(word2vec(sentences[1]))

wordEncodingSize = 300
hidden_size = 300
input_size = wordEncodingSize
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


def train(decoder,decoder_optimizer,criterion,input_embeddings,target_embeddings): 
    
    loss = 0
    decoder_optimizer.zero_grad()
    decoder_hidden, decoder_state = decoder.initHidden()
    
    #use embeddings as target variable 
    target_variable = target_embeddings
    input_variable = input_embeddings
    nWords, VocSize = input_variable.shape
    
    decoder_input = torch.FloatTensor(input_variable[1]) 
    decoder_input = decoder_input.unsqueeze(0).unsqueeze(0)  
    decoder_input = Variable(decoder_input).cuda()
    
    #Without teacher forcing #ignore <SOS> #teaching it to break at <EOS>
    for di in range(2,nWords):
        
        decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,decoder_state)
        
        temp_target = torch.FloatTensor(target_variable[di]) 
        temp_target = temp_target.unsqueeze(0).unsqueeze(0)  
        decoder_target = Variable(temp_target).cuda()
        decoder_target = decoder_target.long()
        decoder_target = decoder_target.squeeze(0)
        label = torch.max(decoder_target, 1)[1]
        
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        next_input = w2v_embeddings[ni.cpu().numpy()]
        decoder_input = Variable(torch.FloatTensor(next_input).unsqueeze(0)) 
        decoder_input = decoder_input.cuda() 
        
        loss += criterion(decoder_output.squeeze(0), label)
          

    loss.backward()

    decoder_optimizer.step()

    return loss.data[0] / (nWords - 1)
    

# Train the model and monitor the loss
def trainIters(decoder, epochs, learning_rate):
    
    plot_loss_total = 0  # Reset every plot_every

    criterion = nn.CrossEntropyLoss()
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    
    for epoch in range(epochs):
        count = 0
        for sentence in sentences[:100000]:
            
            input_embeddings = word2vec(sentence)
            target_embeddings = one_hot(sentence)
            loss = train(decoder, decoder_optimizer, criterion, input_embeddings,target_embeddings)
            plot_loss_total += loss
            
            if (count % 5000 == 0):
                print(plot_loss_total/5000)
                plot_loss_total = 0
            
            count = count + 1
         
    
    
epochs = 1
learning_rate = 0.00001

trainIters(decoder,epochs,learning_rate)  

torch.save(decoder.state_dict(), './decoderW1.pth')
print('training done')

