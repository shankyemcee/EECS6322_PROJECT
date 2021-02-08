# -*- coding: utf-8 -*-

from getConfigs.getConfigs import get_configs




if __name__ == "__main__":
    config=get_configs(filename='train_config.ini',section='DEFAULT')
    print(config)
    
    