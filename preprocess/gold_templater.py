


import json
import re

with open('../data/train_lm_preprocessed.json','r') as f:
    gold_train = json.load(f)


#split in row, column, token_count format
def templatize_colname(string,col_no):
    template = ""
    count = 0
    for tok in string.split():
        template += "template[0][" + str(col_no) + "][" + str(count) + "] "
        count+=1
    return template   

def templatize_colvalue(string,row_no,col_no):
    template = ""
    count = 0
    for tok in string.split():
        template += "template[" + str(row_no) + "][" + str(col_no) + "][" + str(count) + "] "
        count+=1
    return template   


for entry in gold_train:
    
    # gold_train.index(entry)
    # entry = gold_train[2015]

    col_list = entry[1]
    data_string = entry[4]
    
    templated_data_string = ""
    
    while len(data_string) > 0:
        row = re.search('In row (.+?) , ', data_string)
        row_no = row.group(1)
        data_string = data_string[len(row.group(0)):]
        templated_data_string += row.group(0)
        col_count = 1
        for col_no in col_list:

            if len(col_list) == col_count:
                col = re.search('the (.+?) is (.+?) (\.) ', data_string)
            else:
                col = re.search('the (.+?) is (.+?) (,) the', data_string)
            col_count+=1
            templ_colname = templatize_colname(col.group(1),col_no)
            templ_colvalue = templatize_colvalue(col.group(2),row_no,col_no)
            templated_data_string+= "the " + templ_colname + "is " + templ_colvalue + col.group(3) + " "
            data_string = data_string[col.group(0).rfind(" ")+1:]

    gold_train[gold_train.index(entry)][4] = templated_data_string
    
        
        
        
with open('../data/train_lm_preprocessed_templt.json','w') as f:
    json.dump(gold_train,f)
        

















       