import os
import json
import argparse
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from tqdm import tqdm 
import re 
from typing import Tuple
import requests
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=int, default = 1)
parser.add_argument('--l', type=str, default = "")

args, unknown_args = parser.parse_known_args()

if unknown_args:
    print(f"Unrecognized arguments: {unknown_args}")





dataset_name = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']
models = ["mistralai-2", "llama-3-hf", "mistralai-3", "llama-2-hf" , "Saul" , "quen-2" ]
base_paths = ["/home/coder/ensemble-llm/mistralai-2", 
             "/home/coder/ensemble-llm/llama-3-hf" , 
             "/home/coder/ensemble-llm/mistralai-3", ]

main_path = "/home/coder/ensemble-llm"

i_ = args.i
link = args.l



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
#                     print(f"JSONDecodeError in fallback: {e} - Could not parse as single JSON.")
                    pass # Continue to next fallback
        except Exception as e:
#             print(f"An unexpected error occurred during JSON fallback matching: {e}")
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



all_inputs_mis2= []


all_inputs_llama3= []



all_inputs_mis3 = []


all_targets = []

inputs = []

candidates_texts = []

insts = []

print()
print("*"*100)
print()
print()


print(dataset_name[i_])
print()
print()
print()
print("*"*100)
print()

f_name_1 = f"res-{models[0]}-{dataset_name[i_]}-v3.csv"  
f_name_2 = f"res-{models[1]}-{dataset_name[i_]}-v3.csv"  
f_name_3 = f"res-{models[2]}-{dataset_name[i_]}-v3.csv"

folder_path1 = os.path.join(base_paths[0],f_name_1)
folder_path2 =  os.path.join(base_paths[1],f_name_2)
folder_path3 =  os.path.join(base_paths[2],f_name_3)

df1 = pd.read_csv(folder_path1)
df2 = pd.read_csv(folder_path2)
df3 = pd.read_csv(folder_path3)


df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()

print(len(df1),len(df2),len(df3))

ans1 = list(df1["Answer"])
ans2 = list(df2["Answer"])
ans3 = list(df3["Answer"])

ea_1 = []
s_1 = []
A = []
for a in ans1: 

    a_ = extract_answers_and_scores(a)

    if(len(a_) == 0):

        ea_1.append([])
        s_1.append([])

    else:

        ea_1.append(a_[0]["answer"])
        s_1.append(a_[0]["score"])


ea_2 = []
s_2 = []
A = []
for a in ans2: 

    a_ = extract_answers_and_scores(a)

    if(len(a_) == 0):

        ea_2.append([])
        s_2.append([])

    else: 
        ea_2.append(a_[0]["answer"])
        s_2.append(a_[0]["score"])


ea_3 = []
s_3 = []
A = []
for a in ans3: 

    a_ = extract_answers_and_scores(a)

    if(len(a_) == 0):

        ea_3.append([])
        s_3.append([])

    else:
        ea_3.append(a_[0]["answer"])
        s_3.append(a_[0]["score"])



df1["Extracted_Answer"] = ea_1
df1["Score"] = s_1

df2["Extracted_Answer"] = ea_2
df2["Score"] = s_2

df3["Extracted_Answer"] = ea_3
df3["Score"] = s_3


df1 = df1[df1['Extracted_Answer'].apply(lambda x: len(x) > 0)]
df2 = df2[df2['Extracted_Answer'].apply(lambda x: len(x) > 0)]
df3 = df3[df3['Extracted_Answer'].apply(lambda x: len(x) > 0)]


df1_index = set(list(df1["Index"]))
df2_index = set(list(df2["Index"]))
df3_index = set(list(df3["Index"]))

print(len(df1_index),len(df2_index),len(df3_index))






def build_aggregation_prompt(a1: str, c1: float,
                             a2: str, c2: float,
                             a3: str, c3: float) -> str:
 
    # Normalize/trim answers
    a1 = a1.strip()
    a2 = a2.strip()
    a3 = a3.strip()

    # Format confidences to a consistent string (e.g., 0.73)
    def fmt_conf(x: float) -> str:
        return f"{float(x):.2f}".rstrip('0').rstrip('.')  # e.g., 0.7 or 0.73

    c1_s, c2_s, c3_s = map(fmt_conf, (c1, c2, c3))

    # Build prompt, preserving {{QUESTION}} verbatim by doubling braces in f-string
    prompt = (
        "You are a helpful assistant.\n\n"
        "You will receive three answers to the same question, each with a confidence score (0–1). "
        "Your task is to produce a single, best possible answer by combining the most accurate and "
        "relevant parts of the inputs. Use confidence scores as guidance, but do not copy any answer "
        "verbatim. Resolve conflicts and ensure clarity.\n\n"
        "### Inputs\n"
        f"Answer 1 (Confidence Scores: C1={c1}): {a1}\n"
        f"Answer 2 (Confidence Scores: C2={c2}): {a2}\n"
        f"Answer 3 (Confidence Scores: C3={c3}): {a3}\n\n"
        "### Output\n"
        "Provide only the improved final answer. Do not include reasoning or any extra text."
    )

    return prompt



all_outputs_mis2= []
all_confidences_mis2= []

all_outputs_llama3= []
all_confidences_llama3 = []


all_outputs_mis3 = []
all_confidences_mis3= []

all_targets = []

for i in range(0,1000):

    if(i in df1_index and i in df2_index and i in df3_index):

        df1_ = df1[df1['Index'] == i]
        df2_ = df2[df2['Index'] == i]
        df3_ = df3[df3['Index'] == i]

        df1_ea = list(df1_["Extracted_Answer"])[0]
        df2_ea = list(df2_["Extracted_Answer"])[0]
        df3_ea = list(df3_["Extracted_Answer"])[0]


        df1_s = list(df1_["Score"])[0]
        df2_s = list(df2_["Score"])[0]
        df3_s = list(df3_["Score"])[0]

        df1_gt = list(df1_["Response"])[0]




        all_targets.append(df1_gt)

        all_outputs_mis2.append(df1_ea)
        all_outputs_llama3.append(df2_ea)
        all_outputs_mis3.append(df3_ea)

        all_confidences_mis2.append(df1_s)
        all_confidences_llama3.append(df2_s)
        all_confidences_mis3.append(df3_s)


model_input = []

for i in range(0,len(all_outputs_mis2)): 

    a = build_aggregation_prompt(all_outputs_mis2[i], all_confidences_mis2[i],
                        all_outputs_llama3[i], all_confidences_llama3[i],
                        all_outputs_mis3[i], all_confidences_mis3[i])


    model_input.append(a)



def send_model_inputs_one_by_one(
    prompt: str,
    url: str = f"http://{link}", 
    model: str = "gpt-4",
    max_tokens: int = 128,
    temperature: float = 0.1,
    presence_penalty: float = 1.0,
    system_message: str = "Output only the improved final answer.",
    headers: dict = {"Content-Type": "application/json"},
    timeout: int = 60
) -> dict:
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": prompt}]

    payload = {
        "chat_message": messages,
        "max_tokens": max_tokens,
               "model": model,
        "temperature": temperature,
        "presence_penalty": presence_penalty
    }

    try:
        
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        return resp.json()['response'] if resp.status_code == 200 else {"error": resp.text, "status": resp.status_code}
    except Exception as e:

        print(e)



print()
print(f"{dataset_name[i_]}.csv")
print()


file_path = os.path.join(main_path ,f"{dataset_name[i_]}.csv")

if os.path.exists(file_path): 

   

    df =  pd.read_csv(file_path)
    start_index = len(df)

    print(f"file exists: start index {start_index}")

else: 


    df = pd.DataFrame(columns=["pred", "gt"])

    df.to_csv(file_path, index=False)

    start_index = 0 

    print(f"file does not exists")

    


gt = []
pred = []
for i in tqdm(range(start_index, len(model_input))):   #len(model_input) start_index + 50

    time.sleep(30)

    a_ = send_model_inputs_one_by_one(model_input[i])


    pred.append(a_)
    gt.append(all_targets[i])

    if isinstance(a_, dict):

        print()
        print(f"Step {i} : hit the rate limit")
        print()
        break 

    print(f"\rStep {i} | target: {all_targets[i]} | pred: {a_}", end="", flush=True)


df_ = {"pred" : pred ,
    "gt" : gt }


df_ = pd.DataFrame(df_)

df_combined = pd.concat([df, df_], ignore_index=True)

df_combined.to_csv(file_path, index=False)


print(len(df_combined))
