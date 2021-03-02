# -*- coding: utf-8 -*-

import argparse
from Configs.ConfigHandler import ConfigHandler
from DataHandler.DataHandler import DataHandler
#from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch as th
#from tensorboardX import SummaryWriter
import datetime
import tensorflow as tf




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
    
    if config['stage'] == 2:
            model.load_state_dict(th.load(config['checkpoint_dir'] + config['model_checkpoint_file']))
    
    device = th.device(config['device'])
#load the model to the default gpu/cpu device specified in config    
    model.to(device)
# set the model to train mode    
    model.train()
    
    dataHandler = DataHandler()
    
#load the gold references file of the model for training    
    gold_train = dataHandler.get_gold_train(config)
    
    loss_function = th.nn.CrossEntropyLoss(reduction='none')

    optimizer = th.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    writer = tf.summary.create_file_writer(config['log_dir'] + current_time)
    #writer.add_scalar('datetime of starting: ', current_time)
    
    for epoch in range(int(config['epochs'])):
        for row in range(len(gold_train)):
            batch_embedding = dataHandler.get_train_embedding(gold_train,config,tokenizer)        
            input_tensor, mask_tensor, output_tensor = batch_embedding
            input_tensor = input_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            output_tensor = output_tensor.to(device)
            
            optimizer.zero_grad()
            
#the input tensor can sometimes be smaller than the output tensor 
#or sometimes greater, but the model output will be of dimension equal
#to the model input and should match the size of output tensor before
#feeding to the softmax. Thus we concatenate and truncate by max sequence
#length to prevent the size of the tensor going over the max length
            model_input = th.cat([input_tensor,output_tensor],1)
            model_input = model_input[:,:int(config['max_length'])]

        # extract the predicted tensor and store in model_output
            model_output = model(model_input)[0]
            model_output = model_output[:,-output_tensor.shape[1]:,:].contiguous()
            loss_tensor = loss_function(model_output.view(-1,model_output.shape[2]),output_tensor.view(-1))
            
        #we multiply back the mask tensor to set all the extra pad tokens from the
        #input and caption tensors to 0 so they do no contribute towards the loss    
            loss_tensor = loss_tensor * mask_tensor.view(-1)
            loss_tensor = loss_tensor.sum()/loss_tensor.shape[0]
            loss_tensor.backward()
            optimizer.step()
            
            with writer.as_default():
                tf.summary.scalar('loss', loss_tensor.tolist(), step=epoch)
        print("epoch: " + str(epoch) + "  loss: " + loss_tensor.tolist())    
        th.save(model.state_dict(), config['checkpoint_dir'] + 'C2F_stage{}_epoch{}.pt'.format(config['stage'], epoch))

    writer.close()       
                