# -*- coding: utf-8 -*-


import random
import torch as th
import json
import pandas as pd

class DataHandler():
    def __init__(self):
       pass;
    
    def get_gold_train(self,config):
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
    
    def get_gold_test(self,config):
        if type(config) != dict:
            raise TypeError("must provide config file as type dict") 
        elif type(config['gold_test']) != str:
            raise TypeError("must provide filename of gold file in string format") 
        elif config['gold_test'][-5:] != ".json": 
            raise Exception("gold file must be a .json file") 
        else:
            with open(config['gold_test']) as f:
                gold_file = json.load(f)
        
        
        return gold_file
    
    def get_references(self,table_entries):
        
        ref_list = [entry[0].lower().split() for entry in table_entries]
        
        return ref_list
        
        
    
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
                tokenized_table = tokenizer.tokenize(entry[4])   
                tokenized_title = tokenizer.tokenize(entry[2])
                tab_n_title = tokenized_title + tokenized_table
                tab_n_title = tab_n_title[:int(config['max_length'])-1] 
                encoded_input_list.append(tokenizer.convert_tokens_to_ids(tab_n_title))
                
                
                
                # encoded_title = tokenizer.encode(entry[2])
                # encoded_table = tokenizer.encode(entry[4])
                # encoded_input_list.append(encoded_title + encoded_table)
                # encoded_input_list[-1] = encoded_input_list[-1][:int(config['max_length'])-1]
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
        
        
        output_tensor = th.LongTensor(encoded_output_list)
        mask_tensor = th.Tensor(mask_list)
        input_tensor = th.LongTensor(encoded_input_list)    
            
        
        return input_tensor, mask_tensor, output_tensor
            
            
            
            
    def get_test_embedding(self,gold_test,table_id,config,tokenizer):
        
        '''      
       
        with open("data/test_lm.json") as f:
            gold_test = json.load(f)
   
        '''
        data = pd.read_csv(config['data_tables'] + table_id, '#')
#each table entry has several references and each reference has the
#original string, templated string, title and linked columns
        ref_list = gold_test[table_id]
        
        encoded_output_list = []
        encoded_input_list = []
        output_max_length = 0
        input_max_length = 0
# convert each templated string from ref_list to encoded form using 
#GPT2 encoding. It will convert each token to a GPT2 token and use the vocabulary
#to convert them into numbers

        for entry in ref_list:
            try:
                    encoded_actual = tokenizer.encode(entry[0])
                    encoded_sep = tokenizer.encode(' [SEP] ')
                    encoded_template = tokenizer.encode(entry[3])
                    encoded_output = encoded_actual + encoded_sep + encoded_template
                    encoded_output_list.append(encoded_output)
                    output_max_length = max(output_max_length , len(encoded_output))
            except IndexError:
                raise IndexError("Encountered incomplete entry: ",entry)
            
            row_string = ""
            
            for index, row in data.iterrows():
                
                try:    
                    row_string += " In row " + str(index + 1) + " , "  
                    col_string = ""
                    for linked_col in entry[1]:
                        data_string = str(data.columns[linked_col])
                        col_string += "the " + data_string + " is "
                        data_string = str(row[linked_col]).title()
                        col_string += data_string + " , " 

                    col_string = col_string[:-3] + " . "
                    row_string += col_string
                 
                except IndexError:
                    raise IndexError("Encountered incomplete entry: ",entry)  
            
            tokenized_table = tokenizer.tokenize(row_string)   
            tokenized_title = tokenizer.tokenize(entry[2])
            tab_n_title = tokenized_title + tokenized_table
            # if len(tab_n_title) > 1024: print(table_id,len(tokenized_title + tokenized_table))
            tab_n_title = tab_n_title[:int(config['max_length'])-1] 
            encoded_input_list.append(tokenizer.convert_tokens_to_ids(tab_n_title))
            # print(table_id,len(encoded_input_list[-1]))
            # encoded_table = tokenizer.encode(row_string)   
            # encoded_title = tokenizer.encode(entry[2])
            # encoded_input_list.append(encoded_title + encoded_table)
            # print(table_id,len(encoded_input_list[-1]))
            # encoded_input_list[-1] = encoded_input_list[-1][:int(config['max_length'])-1]     
            input_max_length = max(input_max_length , len(encoded_input_list[-1]))        
                    
                   
                    
                
            
            
            
# append eos token id at the end of longest encoding,
# and pad all encodings with eos token id to match length of longest string
        output_max_length += 1
        for entry in encoded_output_list:
            index = encoded_output_list.index(entry)
            eos_list = (output_max_length - len(entry))* [tokenizer.eos_token_id]
            encoded_output_list[index] = encoded_output_list[index] + eos_list



        
#Append eos tokens before each encoded input and pad using the eos tokens
#to make all inputs of equal length       
        input_max_length += 1
        for entry in encoded_input_list:
            index = encoded_input_list.index(entry)
            eos_list = (input_max_length - len(entry))* [tokenizer.eos_token_id]
            encoded_input_list[index] = eos_list + encoded_input_list[index]
        
        
        output_tensor = th.LongTensor(encoded_output_list)
        input_tensor = th.LongTensor(encoded_input_list)    
            
        
        return input_tensor, output_tensor     
        
    
    
    
         