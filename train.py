# -*- coding: utf-8 -*-

import argparse
from Configs.ConfigHandler import ConfigHandler
from DataHandler.DataHandler import DataHandler
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='train_config.ini',
                        help="the .ini file containing all the model and program settings")
    parser.add_argument("--section", type=str, default='DEFAULT',
                        help="the section of config file")
    args = parser.parse_args()    
    
#load all the model configuration settings from the config file
    config=ConfigHandler.get_configs(filename=args.config_file,section=args.section)
    print(config)
    
    
    tokenizer = GPT2Tokenizer.from_pretrained(config['model'])

    tokenizer.add_tokens(['[ENT]', '[SEP]'])

    model = GPT2LMHeadModel.from_pretrained(config['model'])
    model.resize_token_embeddings(len(tokenizer))
    
    dataHandler = DataHandler()
    
#load the gold references file of the model for training    
    gold_train = dataHandler.get_gold_file(config)
    

    # for epoch in range(int(config['epochs'])):
    #     for row in range(len(gold_train)):
    batch_embedding = dataHandler.get_train_embedding(gold_train,config,tokenizer)        
    input_tensor, mask_tensor, output_tensor = batch_embedding
    
    
    
    
    
    
    