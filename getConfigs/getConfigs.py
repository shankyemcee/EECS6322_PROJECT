# -*- coding: utf-8 -*-



import configparser
 
 
 
def get_configs(filename='config',section='DEFAULT'):
    config = configparser.ConfigParser()
    config.read(filename)
    return dict(config.items(section))
