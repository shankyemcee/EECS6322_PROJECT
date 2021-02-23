# -*- coding: utf-8 -*-


import random
import torch
import json

class DataHandler():
    def __init__(self):
       pass;
    
    def get_gold_file(self,config):
        if type(config) != dict:
            raise TypeError("must provide config file as type dict") 
        elif type(config['gold_train']) != str:
            raise TypeError("must provide filename of gold file in string format") 
        elif config['gold_train'][-5:] != ".json": 
            raise Exception("gold file must be a .json file") 
        else:
            with open(config['gold_train']) as f:
                gold_file = json.load(f)
        
        
        if len(gold_file) < int(config['batch_size']):
            raise Exception("gold file must have atleast batch size no. of entries")
        
        
        return gold_file
    
    
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
            try:
                if int(config['stage']) == 1:
                    encoded_output = tokenizer.encode(entry[3])
                    encoded_output_list.append(encoded_output)
                    max_length = max(max_length , len(encoded_output))
                    mask_list.append(len(encoded_output)*[1])
                elif int(config['stage']) == 2:
                    encoded_actual = tokenizer.encode(entry[0])
                    encoded_sep = tokenizer.encode(' [SEP] ')
                    encoded_template = tokenizer.encode(entry[3])
                    encoded_output = encoded_actual + encoded_sep + encoded_template
                    encoded_output_list.append(encoded_output)
                    max_length = max(max_length , len(encoded_output))
                    mask_list.append(len(encoded_output)*[1])
            except IndexError:
                raise IndexError("Encountered incomplete entry: ",entry)

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
            try:
                encoded_title = tokenizer.encode(entry[2])
                encoded_table = tokenizer.encode(entry[4])
                encoded_input_list.append(encoded_title + encoded_table)
                encoded_input_list[-1] = encoded_input_list[-1][:int(config['max_length'])-1]
                max_length = max(max_length , len(encoded_input_list[-1]))
            except IndexError:
                raise IndexError("Encountered incomplete entry: ",entry)    
        
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
            
            
            
            
            
            
            