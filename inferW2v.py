from nltk.translate.bleu_score import SmoothingFunction
cc = SmoothingFunction()

sentences = val_sentences

# Lower-case the sentence, tokenize them and add <SOS> and <EOS> tokens
sentences = [["<SOS>"] + word_tokenize(sentence.lower()) + ["<EOS>"] for sentence in sentences]

#bleu function with reweighting
def bleu(reference_sentence, predicted_sentence):
    return sentence_bleu([reference_sentence], predicted_sentence,smoothing_function=cc.method4)

print(sentences[1])
print(numberize(sentences[1]))
print(word2vec(sentences[1]))


def seq2seq_inference(sentence,encoder,decoder,hidden_states,max_length=maxSequenceLength):
    
    encoder_hidden,encoder_state = encoder.initHidden()
    
    input_variable = word2vec(sentence)
    #target_variable = one_hot(sentence)
    nWords, VocSize = input_variable.shape
    
    encoder_outputs = Variable(torch.zeros(nWords,1, hidden_size))
    encoder_outputs = encoder_outputs.cuda() 
    
    for ei in range(1,nWords):
        
        encoder_input = torch.FloatTensor(input_variable[ei]) 
        encoder_input = encoder_input.unsqueeze(0).unsqueeze(0)  
        encoder_input = Variable(encoder_input).cuda()
        
        encoder_output, encoder_hidden,encoder_state = encoder(encoder_input,encoder_hidden,encoder_state)
        encoder_outputs[ei] = encoder_output[0][0]
        
    
    ind = word2index.get("<SOS>", 0)
    one_hot_vec = w2v_embeddings[ind]
    decoder_input = torch.FloatTensor(one_hot_vec)
    decoder_input = decoder_input.unsqueeze(0).unsqueeze(0) 
    decoder_input = Variable(decoder_input).cuda() 
    
    hidden_states.append(encoder_hidden.squeeze().cpu().data.numpy()) 
    decoder_hidden = encoder_hidden
    decoded_words = ["<SOS>"]
    
    for di in range(max_length):
        
        if (di<nWords):
            decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,
                                                              encoder_outputs[di].unsqueeze(0))
        else: 
            decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,
                                                              encoder_outputs[nWords-1].unsqueeze(0))
            
        decoder_output = softmax(decoder_output.squeeze(0))
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        #detect <EOS> 
        if (ni == 3):
            decoded_words.append("<EOS>")
            break
        elif (ni != 0):
            #print(ni)
            decoded_words.append(index2word.get(ni,0))

        next_input = w2v_embeddings[ni]
        decoder_input = Variable(torch.FloatTensor(next_input).unsqueeze(0).unsqueeze(0)) 
        decoder_input = decoder_input.cuda() 
        
    s = " "
    predicted = decoded_words
    #print(s.join(sentence))
    #print(s.join(predicted))
    bleu_score = bleu(sentence,predicted)
    return bleu_score,hidden_states

    # Perform inference for all validation sequences and report the average BLEU score
hidden_states = []
softmax = nn.Softmax()
index2word = {index:word for index,word in enumerate(vocabulary)}

encoder.load_state_dict(torch.load('./encoderW.pth'))

decoder.load_state_dict(torch.load('./decoderW2.pth'))

Total_bleu = 0 
for sentence in sentences[:25]:
    
    blue_score,hidden_states = seq2seq_inference(sentence,encoder,decoder,hidden_states)
    Total_bleu += blue_score


#np.save(open('outputs/hidden_states_val_W', 'wb+'), hidden_states)     
#print("Average bleu score:",Total_bleu/500)    