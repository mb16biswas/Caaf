import pandas as pd 
import numpy as np 
import os
import json

base_folder_met = "/workspace/data/Momojit/ensemble-llm/pre-train-model/infer-res-met/"

files = os.listdir(base_folder_met)

print()
print()
print()
print(files)
print()
print()
print()


mis_2_rl = []
mis_2_b = []
mis_2_sb = []
mis_2_s = []
mis_2_br = []
mis_2_mt = []
mis_2_rls = []


mis_3_rl = []
mis_3_b = []
mis_3_sb = []
mis_3_s = []
mis_3_br = []
mis_3_mt = []
mis_3_rls = []

ll_rl = []
ll_b = []
ll_sb = []
ll_s = []
ll_br = []
ll_mt = []
ll_rls = []


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
    
    mis_3_rl.append(data['mistral-3']['Rouge-l'])
    mis_3_b.append(data['mistral-3']['Blue'])
    mis_3_sb.append(data['mistral-3']['Sac_Blue'])
    mis_3_s.append(data['mistral-3']['Sari'])
    mis_3_br.append(data['mistral-3']['Bert_f1'])
    mis_3_mt.append(data['mistral-3']['Meteor'])
    mis_3_rls.append(data['mistral-3']['Rouge-ls'])

    mis_2_rl.append(data['mistral-2']['Rouge-l'])
    mis_2_b.append(data['mistral-2']['Blue'])
    mis_2_sb.append(data['mistral-2']['Sac_Blue'])
    mis_2_s.append(data['mistral-2']['Sari'])
    mis_2_br.append(data['mistral-2']['Bert_f1'])
    mis_2_mt.append(data['mistral-2']['Meteor'])
    mis_2_rls.append(data['mistral-2']['Rouge-ls'])

    ll_rl.append(data['llama-3']['Rouge-l'])
    ll_b.append(data['llama-3']['Blue'])
    ll_sb.append(data['llama-3']['Sac_Blue'])
    ll_s.append(data['llama-3']['Sari'])
    ll_br.append(data['llama-3']['Bert_f1'])
    ll_mt.append(data['llama-3']['Meteor'])
    ll_rls.append(data['llama-3']['Rouge-ls'])
                                 
    p_rl.append(data['proposed']['Rouge-l'])
    p_b.append(data['proposed']['Blue'])
    p_sb.append(data['proposed']['Sac_Blue'])
    p_s.append(data['proposed']['Sari'])
    p_br.append(data['proposed']['Bert_f1'])
    p_mt.append(data['proposed']['Meteor'])
    p_rls.append(data['proposed']['Rouge-ls'])

                                

d = {
    
    "exp" : name_, 
    "mis_2_rl" :  mis_2_rl, 
    "mis_2_blue" :  mis_2_b, 
    "mis_2_sac_blue" : mis_2_sb,
    "mis_2_sari" : mis_2_s, 
    "mis_2_bf" : mis_2_br, 
    "mis_2_mt": mis_2_mt,
    "mis_2_rls" :  mis_2_rls, 
    
    
    "mis_3_rl" :  mis_3_rl, 
    "mis_3_blue" :  mis_3_b, 
    "mis_3_sac_blue" : mis_3_sb,
    "mis_3_sari" : mis_3_s, 
    "mis_3_bf" : mis_3_br, 
    "mis_3_mt": mis_3_mt,
    "mis_3_rls" :  mis_3_rls, 
    
    "ll_3_rl" :  ll_rl, 
    "ll_3_blue" :  ll_b, 
    "ll_3_sac_blue" : ll_sb,
    "ll_3_sari" : ll_s, 
    "ll_3_bf" : ll_br,
    "ll_3_mt" : ll_mt,
    "ll_3_rls" :  ll_rls, 
    
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
