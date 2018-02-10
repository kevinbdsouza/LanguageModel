# Now get nearest neighbors and print
import math

epoch = 0
f_train = np.load(open('outputs/hidden_states_train'+str(epoch), 'rb'))
f_val = np.load(open('outputs/hidden_states_val', 'rb'))


for val_id in range(10):
       
    val_vec = f_val[val_id]
    min_id = 0
    min_dist = math.inf
    for train_id in range(200000):
        train_vec = f_train[train_id]
        
        dist = np.square(np.linalg.norm(val_vec-train_vec))
        if (dist < min_dist):
            min_dist = dist
            min_id = train_id
            
           
    
    print(val_sentences[val_id])
    print(train_sentences[min_id])
    print("")
    
