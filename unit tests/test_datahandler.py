# -*- coding: utf-8 -*-

import sys 
sys.path.append('..')

import unittest
from DataHandler.DataHandler import DataHandler
from Configs.ConfigHandler import ConfigHandler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch as th

class TestDataHandler(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.dataHandler = DataHandler()
    
    def test_getgoldfile(self):
        # file = dataHandler.get_gold_file('aaa.ini')
        # self.assertEquals(file,25)
        self.assertRaises(TypeError,self.dataHandler.get_gold_file,'aaa.ini')
        config = {'gold_train':5}
        self.assertRaises(TypeError,self.dataHandler.get_gold_file,config)
        config = {'gold_train':'aaa.ini'}
        self.assertRaises(Exception,self.dataHandler.get_gold_file,config)
        config = {'gold_train':'../data/train_lm_preprocessed.json',
                  'batch_size':100000}
        self.assertRaises(Exception,self.dataHandler.get_gold_file,config)
        config = {'gold_train':'../data/train_lm_preprocessed.json',
                  'batch_size':6}
        self.assertIsInstance(self.dataHandler.get_gold_file(config),list)


    def test_gettrainembedding(self):
        config=ConfigHandler.get_configs(filename="../train_config.ini",section="DEFAULT")
        config['gold_train'] = '../data/train_lm_preprocessed.json'
        tokenizer = GPT2Tokenizer.from_pretrained(config['model'])
        tokenizer.add_tokens(['[ENT]', '[SEP]'])
        gold_train = [[1,2,3,4],[5,6,7]]
        self.assertRaises(Exception,self.dataHandler.get_train_embedding,
                          gold_train,config,tokenizer)
        gold_train = self.dataHandler.get_gold_file(config)
        batch_embedding = self.dataHandler.get_train_embedding(gold_train,config,tokenizer)
        tens1,tens2,tens3=batch_embedding
        self.assertIsInstance(tens1, th.Tensor)
        self.assertIsInstance(tens2, th.Tensor)
        self.assertIsInstance(tens3, th.Tensor)




if __name__ == "__main__":
    unittest.main()