# import subprocess



# commands = [
# "pip install --upgrade pip",
# "pip install torch==2.0.1",
# "pip install langchain",
# "pip install langchain-community",
# "pip install langchainhub",
# "pip install chromadb",
# "pip install bs4",
# "pip install sentence_transformers",
# "pip install pypdf",
# "pip install langchain-huggingface",
# "pip install huggingface_hub",
# "pip install ragatouille",
# "pip install openpyxl",
# "pip install lxml",
# "pip install pandas",
# "pip install transformers==4.38.2",
# "pip install accelerate",
# "pip uninstall -y apex",
# "pip install FlagEmbedding==1.3.2",

# ]









# for cmd in commands:
#     subprocess.run(cmd, shell=True)



import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import pandas as pd
import string
from langchain.docstore.document import Document
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from langchain_community.vectorstores import Chroma
from transformers import set_seed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, pipeline, AutoModel,AutoModelForMaskedLM
from huggingface_hub import login
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report
import json
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--f1', type=int, default = 3)
parser.add_argument('--f2', type=int, default = 1)
parser.add_argument('--f3', type=int, default = 2)
parser.add_argument('--f4', type=int, default = 1)
parser.add_argument('--s', type=int, default = 42)


args, unknown_args = parser.parse_known_args()




base = "/workspace/data/Momojit/ensemble-llm/pre-train/"
datasets = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']



f1 = args.f1
f2 = args.f2
f3 = args.f3
f4 = args.f4
seed = args.s

temperature=0.1 
if(f2 >= len(datasets)):
    
    f2 = 1 

# base = "/workspace/data/Momojit/misinter-policy/pdfs/"
# db_path = "/workspace/data/Momojit/misinter-policy/db/"
# cache_path = "/workspace/data/Momojit/misinter-policy/cache-qa3/"
# Chuck_Size = 1200
# Chunk_Overlap = 200
# K_ = 5


if unknown_args:
    print(f"Unrecognized arguments: {unknown_args}")

#sec_key = "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"  #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
sec_key = "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"  #for base_res2.py
sec_key = "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"  #for base_res1.py
os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key
login(token = sec_key)

set_seed(seed)



if(f1==1):

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    folder = "/workspace/data/Momojit/ensemble-llm/base-res-test/mistralai-2/"

elif(f1==2):

    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    folder = "/workspace/data/Momojit/ensemble-llm/base-res-test/llama-3-hf/"

elif(f1==3):


    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    folder = "/workspace/data/Momojit/ensemble-llm/base-res-test/mistralai-3/"


elif(f1==4):

    repo_id = "dfurman/Llama-2-13B-Instruct-v0.2"
    folder = "/workspace/data/Momojit/ensemble-llm/base-res-test/llama-2-13b/"


elif(f1==5):

    repo_id = "Equall/Saul-7B-Instruct-v1"
    folder = "/workspace/data/Momojit/ensemble-llm/base-res-test/Saul/"

else:

    repo_id = "Qwen/Qwen2-7B-Instruct"
    folder = "/workspace/data/Momojit/ensemble-llm/base-res-test/quen-2/"
    temperature=2.0


if(f4 == 1): 

    si_ = 0 
    ei_ = 350

if(f4 == 2): 

    si_ = 500
    ei_ = 1000 

if(f4 == 3): 

    si_ = 1000
    ei_ = 1500 

if(f4 == 4): 

    si_ = 1500
    ei_ = 2000 


if(f4 == 5): 

    si_ = 2000
    ei_ = 2500 

if(f4 == 6): 

    si_ = 2500
    ei_ = 3000 

dataset_name = datasets[f2]

if(f2 == 1 or f2 == 11 or f2 == 4 ):
    
    max_length = 2048
    
else: 
    
    max_length = 2048*2
    
    

f_path = os.path.join(base ,f"test-{dataset_name}.csv")
df = pd.read_csv(f_path)


if(f3 == 1):

    t1 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(repo_id, device_map= "auto")

    if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
    
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=2048*2,temperature=0.1,token=sec_key,task="text-generation")
    
    t2 = time.time()
    
    
    print()
    print("*"*100)
    print("*"*100)
    print()
    print("time taken to download the  model: ", (t2-t1)/60)
    print()
    print("*"*100)
    print("*"*100)
    print()  

elif(f3 == 2): 
    
    t1 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, device_map= "auto")
    model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="auto", torch_dtype=torch.float16)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, return_full_text=False, temperature=temperature)  
    llm = HuggingFacePipeline(pipeline=pipe, batch_size=1)

    # llm = HuggingFacePipeline.from_model_id(
    #         model_id="daryl149/llama-2-7b-chat-hf",
    #         task="text-generation",
    #         model_kwargs={"temperature":0, "max_length": 2048},
    #     )

    t2 = time.time()


    print()
    print("*"*100)
    print("*"*100)
    print()
    print(f"total time to download the model: {t2-t1}")
    print()
    print("*"*100)
    print("*"*100)
    print()  

else:

    t1 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(repo_id, device_map= "auto")

    if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    llm = HuggingFacePipeline.from_model_id(
            model_id=repo_id,
            task="text-generation",
            model_kwargs={"temperature":0, "max_new_tokens" : 2048},
        )

    t2 = time.time()


    print()
    print("*"*100)
    print("*"*100)
    print()
    print(f"total time to download the model: {t2-t1}")
    print()
    print("*"*100)
    print("*"*100)
    print()  




model_name = folder.split("/")[-2]
f_name_curr = f"res-{model_name}-{dataset_name}-v3.csv"  

print()
print()
print("*"*100)
print("*"*100)
print()
print()
print()
print()
print()
print()
print("*"*100)
print("*"*100)
print()
print()
print("base_res4.py")
print(f"repo_id: {repo_id}")
print(f"f1: {f1}")
print(f"f2: {f2}")
print(f"f3: {f3}")
print(f"f4: {f4}")
print(f"seed: {seed}")
print(f"dataset_name: {dataset_name}")
print()
print()
print("*"*100)
print("*"*100)
print()
print()
print()
print()
print()
print("*"*100)
print("*"*100)
print()
print()
print()




def create_folder(folder_path):

    if not os.path.exists(folder_path):
        # Create the directory and all necessary parent directories
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


create_folder(folder)





def generate_prompt(context, question):
    prompt = f"""
You are a helpful assistant. Your task is to answer a question based on a given context.

You must return two things:
1. A concise answer to the question.
2. A confidence score between 0.0 and 1.0 that reflects how confident you are in your answer being correct — whether the answer is present in the context or not.

### Guidelines:
- If the answer is **clearly stated in the context**, answer it directly and return a **high confidence score (e.g., 0.9–1.0)**.
- If the answer is **not present**, but you are confident it is not present, respond accordingly and still return a **high confidence score** (e.g., 0.9–1.0).
- If you are **unsure** or the information is ambiguous, return an appropriate answer (e.g., "I don't know") and a **lower confidence score** (e.g., 0.0–0.6).

Always respond in the following format:
```python
{{
  "answer": "<your generated answer>",
  "score": <confidence_score>
}}


### Examples:

Example 1
Question: What animal is thought to have originated the 2003 SARS outbreak?
Context: The 2003 SARS outbreak was caused by a coronavirus known as SARS-CoV. It was believed to have originated in civet cats.
Output:
{{
"answer": "civet cats",
"score": 0.95
}}

Example 2
Question: Is Favipiravir mentioned as a treatment in the passage?
Context: The passage discusses treatments for COVID-19 including Remdesivir and dexamethasone. It does not mention Favipiravir.
Output:
{{
"answer": "The answer is not present in the context.",
"score": 0.98
}}

Example 3
Question: How many countries does the Nile flow through?
Context: The Nile River is over 6,600 kilometers long and flows through multiple countries including Egypt and Sudan.
Output:
{{
"answer": "I don't know",
"score": 0.3
}}
---

### Now Answer This:

Question: {question}
Context: {context}


"""
    return prompt





# def extract_answers_and_scores(responses):
    
#     responses = [responses]

#     for item in responses:
#         item = item.strip()

#         # Try extracting JSON-style first
#         try:
#             json_match = re.search(r'\{.*"answer"\s*:\s*".*?".*"score"\s*:\s*[\d.]+.*\}', item, re.DOTALL)
#             if json_match:
#                 parsed = json.loads(json_match.group().replace('\n', ' '))
                
#                 return {
#                     "answer": parsed.get("answer", "").strip(),
#                     "score": float(parsed.get("score", 0.0))
#                 }
#                 continue
#         except Exception:
#             pass

#         # Regex fallback: catch malformed dict-like string
#         answer_match = re.search(r"'?answer'?\s*[:=]\s*['\"]?(.*?)['\"]?(,|$)", item)
#         score_match = re.search(r"score\s*[:=]\s*([0-9.]+)", item)

#         answer = answer_match.group(1).strip() if answer_match else item
#         score = float(score_match.group(1)) if score_match else 0.0



#     return {
#             "answer": answer,
#             "score": score
#         }

def extract_answers_and_scores(responses_text):
    extracted_data = []

    qa_blocks = re.findall(
        r'Question:.*?Answer:\s*(.*?)\nScore:\s*(\d+\.?\d+)',
        responses_text,
        re.DOTALL
    )

    if qa_blocks:
        for answer, score_str in qa_blocks:
            extracted_data.append({
                "answer": answer.strip(),
                "score": float(score_str)
            })
    else:
        # Fallback to the previous logic if no such blocks are found,
        # treating the entire 'responses_text' as a single unit if needed.
        # This part should ideally be reviewed based on what kind of responses_text
        # you expect that DON'T fit the "Question:...Answer:...Score:..." format.

        # Try extracting JSON-style first (if the entire text is a JSON object)
        try:
            json_match = re.search(r'\{(?:[^{}]|(?R))*"answer"\s*:\s*".*?"(?:,\n?)"score"\s*:\s*[\d.]+.*?\}', responses_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group().replace('\n', ' '))
                    extracted_data.append({
                        "answer": parsed.get("answer", "").strip(),
                        "score": float(parsed.get("score", 0.0))
                    })
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError in fallback: {e} - Could not parse as single JSON.")
                    pass # Continue to next fallback
        except Exception as e:
            print(f"An unexpected error occurred during JSON fallback matching: {e}")
            pass

        # Specific Regex Fallback for single "answer" and "score" if it's not a block of Q/A
        if not extracted_data: # Only try this if no data was extracted above
            answer_score_match = re.search(r'"answer":\s*"(.*?)"(?:,\n?)"score":\s*(\d+\.?\d+)', responses_text, re.DOTALL)
            if answer_score_match:
                answer = answer_score_match.group(1).strip()
                score = float(answer_score_match.group(2))
                extracted_data.append({
                    "answer": answer,
                    "score": score
                })
            else:
                # Less specific fallback if nothing else works
                answer_match = re.search(r"'?answer'?\s*[:=]\s*['\"]?(.*?)['\"]?(,|$)", responses_text)
                score_match = re.search(r"score\s*[:=]\s*([0-9.]+)", responses_text)

                answer = answer_match.group(1).strip() if answer_match else ""
                score = float(score_match.group(1)) if score_match else 0.0
                if answer or score: # Only add if we actually found something
                    extracted_data.append({
                        "answer": answer,
                        "score": score
                    })

    return extracted_data


question = list(df["question"])[si_:ei_]
docs = list(df["documents"])[si_:ei_]
res = list(df["response"])[si_:ei_]
n_ = len(question)

Index = []
Q = []
D = []
R = []
A = []
T = []
P = []
E = []
S = []

for i in tqdm(range(n_)):


    print()
    print()
    print()
    print("*"*100)
    print("*"*100)
    print("*"*100)
    print("*"*100)
    print()
    print()
    print()
    print(i)
    print()
    print()
    print()
    print("*"*100)
    print("*"*100)
    print("*"*100)
    print("*"*100)
    print()
    print()
    print()
    # try:

    t1_ = time.time()        

    q = question[i]
    d = docs[i]
    r = res[i]

    t2 = generate_prompt(d, q)
    
    tokenized_input = tokenizer(t2, max_length=max_length, truncation=True, return_tensors="pt")
    
    t2 = tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)
    

    ans = llm.invoke(t2)

    ans = ans.replace(t2, '')
    
    d = extract_answers_and_scores(ans)
    
    if(len(d) == 0):
        
        continue 
    
    t2_ = time.time()    
    
    Index.append(i)
    Q.append(q)
    D.append(d)
    R.append(r)
    P.append(t2)
    A.append(ans)
    E.append(d[0]["answer"])
    S.append(d[0]["score"])
    T.append(t2_ - t1_)
    
    




df_ = pd.DataFrame({
    "Index" : Index, 
    "Question" : Q, 
    "Document" : D, 
    "Response" : R, 
    "Prompt"   : P, 
    "Answer"   : A, 
    "Extracted_Answer" : E, 
    "Score" : S, 
    "Time"     : T, 
    })





df_.to_csv(os.path.join(folder,f_name_curr ), index=False)



