# Perform inference for all validation sequences and report the average BLEU score
hidden_states = []
softmax = nn.Softmax()

encoder.load_state_dict(torch.load('./encoder.pth'))

decoder.load_state_dict(torch.load('./decoder2.pth'))

Total_bleu = 0 
for sentence in sentences[:500]:
    
    blue_score,hidden_states = seq2seq_inference(sentence,encoder,decoder,hidden_states)
    Total_bleu += blue_score


np.save(open('outputs/hidden_states_val', 'wb+'), hidden_states)     
print("Average bleu score:",Total_bleu/500)    