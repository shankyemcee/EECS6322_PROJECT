# -*- coding: utf-8 -*-



import configparser
 
 
class ConfigHandler():
    def __init__(self):
       pass; 
 
    @staticmethod
    def get_configs(filename,section):
        config = configparser.ConfigParser()
        config.read(filename)
        return dict(config.items(section))
