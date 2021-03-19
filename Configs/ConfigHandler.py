# -*- coding: utf-8 -*-



import configparser
from pathlib import Path
 
class ConfigHandler():
    def __init__(self):
       pass; 
 
    @staticmethod
    def get_configs(filename,section):
        if not Path(filename).is_file():
            raise Exception("This config file does not exist in the filepath.") 
        config = configparser.ConfigParser()
        config.read(filename)
        try:
            config_dict = dict(config.items(section))
        except:
            raise Exception("No section: ", section) 
        
        return config_dict

















