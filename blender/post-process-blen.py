import pandas as pd 
import numpy as np 
import os
import json

base_folder_met = "/workspace/data/ensemble-llm/llm-blen/infer-res-met/"

files = os.listdir(base_folder_met)

print()
print()
print()
print(files)
print()
print()
print()



p_rl = []
p_b = []
p_sb = []
p_s = []
p_br = []
p_mt = []
p_rls = []




name_ = []

for f in files: 
    
    if(".csv" in f):
        
        continue
    
    
    
    with open(os.path.join(base_folder_met,f), 'r') as file:
        data = json.load(file)
    
    
    name_.append(f)
    

    p_rl.append(data['proposed']['Rouge-l'])
    p_b.append(data['proposed']['Blue'])
    p_sb.append(data['proposed']['Sac_Blue'])
    p_s.append(data['proposed']['Sari'])
    p_br.append(data['proposed']['Bert_f1'])
    p_mt.append(data['proposed']['Meteor'])
    p_rls.append(data['proposed']['Rouge-ls'])

                                

d = {
    
    "exp" : name_, 

    
    "pro_rl" :  p_rl, 
    "pro_blue" :  p_b, 
    "pro_sac_blue" : p_sb,
    "pro_sari" : p_s, 
    "pro_bf" : p_br,
    "pro_mt" : p_mt,
    "pro_rls" :  p_rls, 
    
}


df = pd.DataFrame(d)

df.to_csv(os.path.join(base_folder_met,"all-results.csv"),index= False)
