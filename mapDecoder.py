decoder.load_state_dict(torch.load('./decoder.pth'))
softmax = nn.Softmax()
#decoder.eval()

index2word = {index:word for index,word in enumerate(vocabulary)}

def inference(decoder, init_word, embeddings=one_hot_embeddings, max_length=maxSequenceLength):
    
    decoder_hidden, decoder_state = decoder.initHidden()
    decoded_words = [init_word]
    
    ind = word2index.get(init_word, 0)
    one_hot_vec = embeddings[ind]
    decoder_input = torch.FloatTensor(one_hot_vec)
    decoder_input = decoder_input.unsqueeze(0).unsqueeze(0) 
    decoder_input = Variable(decoder_input).cuda() 
    
    for di in range(max_length):
        
        decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,decoder_state)
        decoder_output = softmax(decoder_output.squeeze(0))
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        #detect <EOS> 
        if (ni == 2):
            break
        else:
            decoded_words.append(index2word.get(ni,0))

        next_input = one_hot_embeddings[ni]
        decoder_input = Variable(torch.FloatTensor(next_input).unsqueeze(0).unsqueeze(0)) 
        decoder_input = decoder_input.cuda() 

    return decoded_words

s = " "
print(s.join(inference(decoder, init_word="the")))
print(s.join(inference(decoder, init_word="man")))
print(s.join(inference(decoder, init_word="woman")))
print(s.join(inference(decoder, init_word="dog")))