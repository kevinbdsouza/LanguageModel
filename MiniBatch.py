sentences = train_sentences

# Lower-case the sentence, tokenize them and add <SOS> and <EOS> tokens
sentences = [["<SOS>"] + word_tokenize(sentence.lower()) + ["<EOS>"] for sentence in sentences]

# Create the vocabulary. Note that we add an <UNK> token to represent words not in our vocabulary.
vocabularySize = 1000
word_counts = Counter([word for sentence in sentences for word in sentence])
vocabulary = ["<pad>"] + [e[0] for e in word_counts.most_common(vocabularySize-1)]
word2index = {word:index for index,word in enumerate(vocabulary)}
one_hot_embeddings = np.eye(vocabularySize)

# Define the max sequence length to be the longest sentence in the training data. 
maxSequenceLength = max([len(sentence) for sentence in sentences])

def numberize(sentence):
    numberized = [word2index.get(word, 0) for word in sentence]
    return numberized

def one_hot(sentence):
    numberized = numberize(sentence)
    # Represent each word as it's one-hot embedding
    one_hot_embedded = one_hot_embeddings[numberized]
    
    return one_hot_embedded

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools


def trainIters(encoder, decoder, epochs, learning_rate):
    
    hidden_states = []
    plot_loss_total = 0  
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        count = 0
        for bch in range(2):
            
            vectorized_sen = [[one_hot(sen)]for sen in sentences[(bch*batch_size)+1:(bch+1)*batch_size]]

            seq_lengths = []
            for i in range(len(vectorized_sen)):
                seq_lengths.append(len(vectorized_sen[i][0]))
    

            seq_tensor = Variable(torch.zeros((len(vectorized_sen),maxSequenceLength,1000))).long().cuda()
            for idx, (seq, seqlen) in enumerate(zip(vectorized_sen, seq_lengths)):
                seq_tensor[idx,:seqlen,:1000] = torch.FloatTensor(seq)
    
            seq_lengths = torch.cuda.LongTensor(seq_lengths)
            seq_tensor = seq_tensor.long()
            # SORT YOUR TENSORS BY LENGTH!
            seq_lengths, perm_idx = seq_lengths.sort(0,descending=True)
            seq_tensor = seq_tensor[perm_idx]
            seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)

            #pack them up 
            packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())

            encoder_hidden,encoder_state = encoder.initHidden()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_outputs = Variable(torch.zeros(maxSequenceLength,batch_size, hidden_size))
            encoder_outputs = encoder_outputs.cuda() 

            #ERROR HERE     
            encoder_output, encoder_hidden,encoder_state = encoder(packed_input,encoder_hidden,encoder_state)
            
    
                                 
            decoder_input = packed_input
            hidden_states.append(encoder_hidden.squeeze().cpu().data.numpy()) 
            decoder_hidden = encoder_hidden

                    
            decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,
                                                                               encoder_output)
            
            #change this 
            '''         
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            ''' 
                    
            loss += criterion(decoder_output.squeeze(0), label)
            
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

                        
            plot_loss_total += loss.data[0] / nWords
            
            if (count % 50 == 0):
                print(plot_loss_total/50)
                plot_loss_total = 0
            
            count = count + 1
            
        
        np.save(open('outputs/hidden_states_train_BW'+str(epoch), 'wb+'), hidden_states)    
           

        
epochs = 1
learning_rate = 0.00001

trainIters(encoder,decoder,epochs,learning_rate)

torch.save(encoder.state_dict(), './encoderBW.pth')
torch.save(decoder.state_dict(), './decoderBW.pth')
print('training done')   


# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output,ht,ct = decoder(packed_input,ht,ct)

