# -*- coding: utf-8 -*-


import random
import torch
import json

class DataHandler():
    def __init__(self):
       pass;
    
    def get_gold_file(self,filename):
        with open(filename) as f:
                return json.load(f)
    
    def get_train_embedding(self,gold_train,config,tokenizer):
#sample batch size random entries from the gold file        
        index = random.randint(int(config['batch_size']),len(gold_train))
        batch_list = [gold_train[i] for i in range(index - int(config['batch_size']),index)]

        encoded_output_list = []
        mask_list = []
        max_length = 0
# convert each templated string from batch_list to encoded form using 
#GPT2 encoding. It will convert each token to a GPT2 token and use the vocabulary
#to convert them into numbers

        for entry in batch_list:
            if int(config['stage']) == 1:
                encoded_output = tokenizer.encode(entry[3])
                encoded_output_list.append(encoded_output)
                max_length = max(max_length , len(encoded_output))
                mask_list.append(len(encoded_output)*[1])
            elif int(config['stage']) == 2:
                pass;

# append eos token id at the end of longest encoding,
# and pad all encodings with eos token id to match length of longest string
        max_length += 1
        for entry in encoded_output_list:
            index = encoded_output_list.index(entry)
            eos_list = (max_length - len(entry))* [tokenizer.eos_token_id]
            encoded_output_list[index] = encoded_output_list[index] + eos_list
            mask_list[index] = mask_list[index] + [1] + (len(eos_list)-1)*[0]



        
        encoded_input_list = []
        max_length = 0



#Append each title(i.e e[2]) with the table description(i.e e[4]) and encode
#with the GPT2 tokenizer
        for entry in batch_list:
            encoded_title = tokenizer.encode(entry[2])
            encoded_table = tokenizer.encode(entry[4])
            encoded_input_list.append(encoded_title + encoded_table)
            encoded_input_list[-1] = encoded_input_list[-1][:int(config['max_length'])-1]
            max_length = max(max_length , len(encoded_input_list[-1]))
            
        
#Append eos tokens before each encoded input and pad using the eos tokens
#to make all inputs of equal length       
        max_length += 1
        for entry in encoded_input_list:
            index = encoded_input_list.index(entry)
            eos_list = (max_length - len(entry))* [tokenizer.eos_token_id]
            encoded_input_list[index] = eos_list + encoded_input_list[index]
        
        
        output_tensor = torch.LongTensor(encoded_output_list)
        mask_tensor = torch.Tensor(mask_list)
        input_tensor = torch.LongTensor(encoded_input_list)    
            
        
        return input_tensor, mask_tensor, output_tensor
            
            
            
            
            
            
            