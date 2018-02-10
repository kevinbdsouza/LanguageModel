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
print(one_hot(sentences[1]))

def seq2seq_inference(sentence,encoder,decoder,hidden_states,max_length=maxSequenceLength):
    
    encoder_hidden,encoder_state = encoder.initHidden()
    
    embeddings = one_hot(sentence)
    input_variable = embeddings
    target_variable = embeddings
    nWords, VocSize = target_variable.shape
    
    encoder_outputs = Variable(torch.zeros(nWords,1, hidden_size))
    encoder_outputs = encoder_outputs.cuda() 
    
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
    decoded_words = ["<SOS>"]
    
    for di in range(max_length):
        
        decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,
                                                              encoder_outputs[di].unsqueeze(0))
        decoder_output = softmax(decoder_output.squeeze(0))
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        
        #detect <EOS> 
        if (ni == 2):
            decoded_words.append("<EOS>")
            break
        else:
            decoded_words.append(index2word.get(ni,0))

        next_input = one_hot_embeddings[ni]
        decoder_input = Variable(torch.FloatTensor(next_input).unsqueeze(0).unsqueeze(0)) 
        decoder_input = decoder_input.cuda() 
        
    
    predicted = decoded_words
    bleu_score = bleu(sentence,predicted)
    return bleu_score,hidden_states