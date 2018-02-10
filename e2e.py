
def train(embeddings, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, criterion,hidden_states,max_length=maxSequenceLength):
    
    encoder_hidden,encoder_state = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = embeddings
    target_variable = embeddings
    nWords, VocSize = target_variable.shape
    
    encoder_outputs = Variable(torch.zeros(nWords,1, hidden_size))
    encoder_outputs = encoder_outputs.cuda() 

    loss = 0

    for ei in range(1,nWords):
        
        encoder_input = torch.FloatTensor(input_variable[ei]) 
        encoder_input = encoder_input.unsqueeze(0).unsqueeze(0)  
        encoder_input = Variable(encoder_input).cuda()
        
        encoder_output, encoder_hidden,encoder_state = encoder(encoder_input,encoder_hidden,encoder_state)
        encoder_outputs[ei] = encoder_output[0][0]
    
    
    ind = word2index.get("<SOS>", 0)
    one_hot_vec = one_hot_embeddings[ind]
    decoder_input = torch.FloatTensor(one_hot_vec)
    decoder_input = decoder_input.unsqueeze(0).unsqueeze(0) 
    decoder_input = Variable(decoder_input).cuda() 
    
    hidden_states.append(encoder_hidden.squeeze().cpu().data.numpy()) 
    decoder_hidden = encoder_hidden

    #Without teacher forcing
    for di in range(1,nWords):
        
        decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,
                                                              encoder_outputs[di].unsqueeze(0))
        
        temp_target = torch.FloatTensor(target_variable[di]) 
        temp_target = temp_target.unsqueeze(0).unsqueeze(0)  
        decoder_target = Variable(temp_target).cuda()
        decoder_target = decoder_target.long()
        decoder_target = decoder_target.squeeze(0)
        label = torch.max(decoder_target, 1)[1]
        
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        next_input = one_hot_embeddings[ni.cpu().numpy()]
        decoder_input = Variable(torch.FloatTensor(next_input).unsqueeze(0)) 
        decoder_input = decoder_input.cuda() 
        
        loss += criterion(decoder_output.squeeze(0), label)
            
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / nWords,hidden_states


def trainIters(encoder, decoder, epochs, learning_rate):
    
    hidden_states = []
    plot_loss_total = 0  
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        count = 0
        for sentence in sentences[:200000]:
            
            embeddings = one_hot(sentence)
            loss,hidden_states = train(embeddings, encoder, decoder, encoder_optimizer,
                                       decoder_optimizer,criterion,hidden_states)
            plot_loss_total += loss
            
            if (count % 5000 == 0):
                print(plot_loss_total/5000)
                plot_loss_total = 0
            
            count = count + 1
            
        
        np.save(open('outputs/hidden_states_train'+str(epoch), 'wb+'), hidden_states)    
           

        
epochs = 1
learning_rate = 0.00001

trainIters(encoder,decoder,epochs,learning_rate)

torch.save(encoder.state_dict(), './encoder.pth')
torch.save(decoder.state_dict(), './decoder2.pth')
print('training done')   