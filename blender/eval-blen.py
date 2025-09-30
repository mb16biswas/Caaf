from datasets import load_metric
import evaluate
from evaluate import load
import pandas as pd
import os
import nltk
import numpy as np
import json
nltk.download('punkt_tab')

meteor = evaluate.load("meteor")
sari = load("sari")
rouge = load_metric("rouge",trust_remote_code=True)
bleu = load_metric("bleu",trust_remote_code=True)
sacrebleu = load_metric("sacrebleu",trust_remote_code=True)
bertscore = load("bertscore")



base_folder_infer = "/workspace/data/ensemble-llm/llm-blen/infer-res/"
base_folder_met = "/workspace/data/ensemble-llm/llm-blen/infer-res-met/"


# paths = ["t5-small-pred-Con_Model_All-covidqa.csv",
# "t5-small-pred-Con_Model_All-delucionqa.csv",
# "t5-small-pred-Con_Model_All-finqa.csv" , 
# "t5-small-pred-Con_Model_All-hagrid.csv",
# "t5-small-pred-Con_Model_All-hotpotqa.csv"]

paths = os.listdir(base_folder_infer)

print()
print()
print(paths)
print()
print()

def rouge_score(pred,truth):
    
    print()
    print("rouge_score")
    print()


    FmeasureL = []
    FmeasureLs = []


    for i,j in zip(pred,truth):

        res = rouge.compute(predictions=[i], references=[j])


        FmeasureL.append(res["rougeL"].mid.fmeasure)
        FmeasureLs.append(res["rougeLsum"].mid.fmeasure)


    return np.mean(FmeasureL), np.mean(FmeasureLs)


def bleu_score(pred,truth):
    
    print()
    print("bleu_score")
    print()


    Blue = []

    for i,j in zip(pred,truth):


        i = [i.split(" ")]
        j = [[j.split(" ")]]

        res = bleu.compute(predictions=i, references=j)['bleu']

        Blue.append(res)

    return np.mean(Blue)


def sacrebleu_score(pred,truth):

    print()
    print("sacrebleu_score")
    print()
    

    Blue = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [[j]]

        res = sacrebleu.compute(predictions=i, references=j)['score']
        Blue.append(res)

    return np.mean(Blue)


def meteor_score(pred,truth):
    
    
    print()
    print("meteor_score")
    print()

    Meteor = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = meteor.compute(predictions=i, references=j)['meteor']
        Meteor.append(res)

    return np.mean(Meteor)


def sari_score(pred,truth):
    
    print()
    print("sari_score")
    print()

    Sari = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = sari.compute(sources = i , predictions=j, references=[j])['sari']
        Sari.append(res)

    return np.mean(Sari)


def bert_score(pred,truth):

    print()
    print("bert_score")
    print()
    
    Bert_f1 = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = bertscore.compute(predictions = i , references=j, model_type = "distilbert-base-uncased")


        Bert_f1.append(res['f1'][0])

    return np.mean(Bert_f1)

print()
print("*"*100)
print("*"*100)
print()
print(paths)
print()
print("*"*100)
print("*"*100)
print()

for f in paths:
    
    print()
    print("*"*100)
    print("*"*100)
    print()
    print(f)
    print()
    print("*"*100)
    print("*"*100)
    print()

    d = {}
    
    d_path = os.path.join(base_folder_infer,f)
    
    df = pd.read_csv(d_path)
    

    pred = list(df["Pred"])
    gt = list(df["GT"])


    pred = [i if type(i) == str else "The Answer is not mentioned in the context" for i in pred] 
    gt = [i if type(i) == str else "The Answer is not mentioned in the context" for i in gt] 

    

    rl,rs = rouge_score(pred,gt)    

    Blue = bleu_score(pred,gt)

    Sac_Blue = sacrebleu_score(pred,gt)

    Meteor = meteor_score(pred,gt)

    Sari = sari_score(pred,gt)

    Bert_f1 = bert_score(pred,gt)

    print()
    print()
    print("*"*100)
    
    print("propossed")
    print(f"Rouge: {rl}")
    print(f"Rouge ls: {rs}")
    print(f"Blue: {Blue}")
    print(f"Sac_Blue: {Sac_Blue}")
    print(f"Meteor: {Meteor}")
    print(f"Sari: {Sari}")
    print(f"Bert_f1: {Bert_f1}")
    print()
    print("*"*100)
    print()
    print()

    d1 = {
        "Rouge-l" : rl,
        "Rouge-ls" : rs,
        "Blue" : Blue , 
        "Sac_Blue" : Sac_Blue, 
        "Meteor" : Meteor, 
        "Sari" : Sari, 
        "Bert_f1" : Bert_f1
    }


    d["proposed"] = d1     


    print()
    print()
    print(d)
    print()
    print()
    
    
    with open(os.path.join(base_folder_met,f[:-4] + ".json"), "w") as f_:
        json.dump(d, f_, indent=4)  
    

    


    
