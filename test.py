# -*- coding: utf-8 -*-

import argparse
from Configs.ConfigHandler import ConfigHandler
from DataHandler.DataHandler import DataHandler
#from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch as th
#from tensorboardX import SummaryWriter
# import datetime
# import tensorflow as tf
from NucleusSampling.NucleusSampling import top_k_top_p_filtering
import torch.nn.functional as F
import nltk
import json





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config.ini',
                        help="the .ini file containing all the model and program settings")
    parser.add_argument("--section", type=str, default='DEFAULT',
                        help="the section of config file")
    args = parser.parse_args()    
    
#load all the model configuration settings from the config file
    config=ConfigHandler.get_configs(filename=args.config_file,section=args.section)
    print(config)
    
    
    tokenizer = GPT2Tokenizer.from_pretrained(config['model'])

#add these tokens to the dictionary otherwise model considers [ENT] as 
#3 seperate tokens([,ENT,])

    tokenizer.add_tokens(['[ENT]', '[SEP]'])

#load the gpt2 model from transformers library
    
    model = GPT2LMHeadModel.from_pretrained(config['model'])

#resize the token embeddings since the model has two extra tokens added
    model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(th.load(config['checkpoint_dir'] + config['model_test_file']))

                      
    device = th.device(config['device'])
#load the model to the default gpu/cpu device specified in config    
    model.to(device)
# set the model to test mode    
    model.eval()
    
    dataHandler = DataHandler()
    
#load the gold references file of the model for training    
    gold_test = dataHandler.get_gold_test(config)
    
    
    sent_bleus_1 = []
    sent_bleus_2 = []
    sent_bleus_3 = []

    seq_list = {}
    with th.no_grad():
        for table_id in gold_test:
            input_tensor,output_tensor = dataHandler.get_test_embedding(gold_test,table_id,config,tokenizer)
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)
            input_dim = input_tensor.shape[1]
            seq_list[table_id] = []


            finished_template = [False for _ in range(len(input_tensor))]
            finished_sentence = [False for _ in range(len(input_tensor))]

            for tok in range(int(config['max_decoding_length'])):
                

                model_output = model(input_tensor)[0]

                
                modeloutput_tail = model_output[:, -1, :]

#apply nuclus filtering. This will set most components of each vector to -inf.    
                filtered_tail = top_k_top_p_filtering(modeloutput_tail,
                                                        top_k=int(config['top_k']),
                                                        top_p=float(config['top_p']))
    

                predicted_tokens = th.multinomial(F.softmax(filtered_tail, dim=-1), num_samples=1)
                

                for token in range(len(predicted_tokens)):
                    if predicted_tokens[token].item() == tokenizer.convert_tokens_to_ids('[SEP]'):
                        finished_template[token] = True
                    if predicted_tokens[token].item() == tokenizer.eos_token_id:
                        finished_sentence[token] = True
    
                input_tensor = th.cat((input_tensor, predicted_tokens), dim=1)
    
                if all(finished_sentence):
                    break;            
            
            predicted_tensor = input_tensor[:,input_dim:]
   
            
            
            for seq in predicted_tensor:
                    decoded_seq = tokenizer.decode(seq, clean_up_tokenization_spaces=True)
                    decoded_seq = decoded_seq[decoded_seq.find('[SEP]') + 6: decoded_seq.find(tokenizer.eos_token)].strip()
                    seq_list[table_id].append(decoded_seq)

            
            references = dataHandler.get_references(gold_test[table_id])
            #get references from the table entry and convert to list of lists
            
            for seq in seq_list[table_id]:
                    
                    seq = seq.lower().split()
                    sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(references, seq, weights=(1, 0, 0)))
                    sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu( references, seq, weights=(0.5, 0.5, 0)))
                    sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(references, seq, weights=(0.33, 0.33, 0.33)))


            bleu_1 = format((sum(sent_bleus_1) / len(sent_bleus_1) * 100), '.2f')
            bleu_2 = format((sum(sent_bleus_2) / len(sent_bleus_2) * 100), '.2f')
            bleu_3 = format((sum(sent_bleus_3) / len(sent_bleus_3) * 100), '.2f')
    
            print("table: {}, bleu-1: {}, bleu-2: {}, bleu-3: {}".format(table_id,bleu_1,bleu_2,bleu_3) )
   

    
        print("total corpus BLEU score = {}/{}/{}".format(bleu_1, bleu_2, bleu_3))
        
        with open( config['test_output_dir'] + 'GPT_C2F_output.json', 'w') as f:
            json.dump(seq_list, f)




            