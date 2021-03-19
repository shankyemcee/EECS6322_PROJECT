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
    
    def test_getconfigs(self):
        self.assertRaises(Exception,ConfigHandler.get_configs,"../configg.ini","DEFAULT")
        self.assertRaises(Exception,ConfigHandler.get_configs,"../config.ini","DEFULT")

    
    
    def test_getgoldtrain(self):
        self.assertRaises(TypeError,self.dataHandler.get_gold_train,'aaa.ini')
        config = {'gold_train':5}
        self.assertRaises(TypeError,self.dataHandler.get_gold_train,config)
        config = {'gold_train':'aaa.ini'}
        self.assertRaises(Exception,self.dataHandler.get_gold_train,config)
        config = {'gold_train':'../data/train_lm_preprocessed.json',
                  'batch_size':100000}
        self.assertRaises(Exception,self.dataHandler.get_gold_train,config)
        config = {'gold_train':'../data/train_lm_preprocessed.json',
                  'batch_size':6}
        self.assertIsInstance(self.dataHandler.get_gold_train(config),list)


    def test_gettrainembedding(self):
        config=ConfigHandler.get_configs(filename="../config.ini",section="DEFAULT")
        config['gold_train'] = '../data/train_lm_preprocessed.json'
        tokenizer = GPT2Tokenizer.from_pretrained(config['model'])
        tokenizer.add_tokens(['[ENT]', '[SEP]'])
        gold_train = [[1,2,3,4],[5,6,7]]
        self.assertRaises(Exception,self.dataHandler.get_train_embedding,
                          gold_train,config,tokenizer)
        gold_train = self.dataHandler.get_gold_train(config)
        batch_embedding = self.dataHandler.get_train_embedding(gold_train,config,tokenizer)
        tens1,tens2,tens3=batch_embedding
        self.assertIsInstance(tens1, th.Tensor)
        self.assertIsInstance(tens2, th.Tensor)
        self.assertIsInstance(tens3, th.Tensor)

    def test_getgoldtest(self):
        self.assertRaises(TypeError,self.dataHandler.get_gold_test,'aaa.ini')
        config = {'gold_train':5}
        self.assertRaises(TypeError,self.dataHandler.get_gold_train,config)
        config = {'gold_train':'aaa.ini'}
        self.assertRaises(Exception,self.dataHandler.get_gold_train,config)
        config = {'gold_train':'../data/train_lm_preprocessed.json',
                  'batch_size':6}
        self.assertIsInstance(self.dataHandler.get_gold_train(config),list)
        
    def test_getreferences(self):
        self.assertRaises(TypeError,self.dataHandler.get_references,{})
        self.assertRaises(IndexError,self.dataHandler.get_references,[])
        self.assertRaises(IndexError,self.dataHandler.get_references,["","aaa"])
        
    def test_gettestembedding(self):
        config=ConfigHandler.get_configs(filename="../config.ini",section="DEFAULT")
        config['gold_test'] = '../data/test_lm.json'
        config['data_tables'] = '../data/tables/'
        tokenizer = GPT2Tokenizer.from_pretrained(config['model'])
        tokenizer.add_tokens(['[ENT]', '[SEP]'])
        gold_test = [[1,2,3,4],[5,6,7]]
        self.assertRaises(Exception,self.dataHandler.get_test_embedding,
                          "2-18424778-6.html.csv",
                          gold_test,config,tokenizer)
        gold_test = self.dataHandler.get_gold_test(config)
        self.assertRaises(IndexError,self.dataHandler.get_test_embedding,gold_test,
                                                              "2-1842477-6.html.csv",
                                                              config,
                                                              tokenizer)
        batch_embedding = self.dataHandler.get_test_embedding(gold_test,
                                                              "2-18424778-6.html.csv",
                                                              config,
                                                              tokenizer)
        tens1,tens2=batch_embedding
        self.assertIsInstance(tens1, th.Tensor)
        self.assertIsInstance(tens2, th.Tensor)
     
        
        
        
        
        

if __name__ == "__main__":
    unittest.main()