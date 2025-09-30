import pandas as pd 
import os 
import re 
import json
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import argparse
import datasets
import time

parser = argparse.ArgumentParser()
parser.add_argument('--f1', type=int, default = 1)
parser.add_argument('--i', type=int, default = 0)


args, unknown_args = parser.parse_known_args()


flag = args.f1
i_ = args.i




if(flag == 1): 
    
    model_name = "llm-blender/gen_fuser_700m"

else: 
    
    model_name = "llm-blender/gen_fuser_3b"


    
print("*"*100)
print()
print()
print("*"*100)
print()
print("blen-test-5.py : rag bench")
print(f"flag: {flag}")
print(f"i_ : {i_}")
print(f"model_name: {model_name}")
print()
print("*"*100)
print()
print()
print("*"*100)
print()


fuser_config = llm_blender.GenFuserConfig()
fuser_config.torch_dtype = "float32"
fuser_config.model_name = model_name

print()
print(fuser_config)
print()

blender = llm_blender.Blender(fuser_config = fuser_config)
blender.loadranker("llm-blender/PairRM")


s_ = """You are a helpful assistant. Your task is to answer a question based on a given context.

You must return two things:
1. A concise answer to the question.
2. A confidence score between 0.0 and 1.0 that reflects how confident you are in your answer being correct — whether the answer is present in the context or not.

### Guidelines:
- If the answer is **clearly stated in the context**, answer it directly and return a **high confidence score (e.g., 0.9–1.0)**.
- If the answer is **not present**, but you are confident it is not present, respond accordingly and still return a **high confidence score** (e.g., 0.9–1.0).
- If you are **unsure** or the information is ambiguous, return an appropriate answer (e.g., "I don't know") and a **lower confidence score** (e.g., 0.0–0.6).

Always respond in the following format:
```python
{
    "answer": "<your generated answer>",
    "score": <confidence_score>
}


### Examples:

Example 1
Question: What animal is thought to have originated the 2003 SARS outbreak?
Context: The 2003 SARS outbreak was caused by a coronavirus known as SARS-CoV. It was believed to have originated in civet cats.
Output:
{
"answer": "civet cats",
"score": 0.95
}

Example 2
Question: Is Favipiravir mentioned as a treatment in the passage?
Context: The passage discusses treatments for COVID-19 including Remdesivir and dexamethasone. It does not mention Favipiravir.
Output:
{
"answer": "The answer is not present in the context.",
"score": 0.98
}

Example 3
Question: How many countries does the Nile flow through?
Context: The Nile River is over 6,600 kilometers long and flows through multiple countries including Egypt and Sudan.
Output:
{
"answer": "I don't know",
"score": 0.3
}
---

### Now Answer This:


"""


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




dataset_name = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']
models = ["mistralai-2", "llama-3-hf", "mistralai-3", "llama-2-hf" , "Saul" , "quen-2" ]
base_paths = ["/workspace/data/ensemble-llm/base-res-test/mistralai-2/", 
             "/workspace/data/ensemble-llm/base-res-test/llama-3-hf/" , 
             "/workspace/data/ensemble-llm/base-res-test/mistralai-3/", 
             "/workspace/data/ensemble-llm/base-res-test/llama-2-hf/", 
             "/workspace/data/ensemble-llm/base-res-test/Saul/", 
             "/workspace/data/ensemble-llm/base-res-test/quen-2/"]




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


Ins = "You are a helpful assistant. Your task is to answer a question based on a given context.\n\n\n"

for i in range(0,1000):

    if(i in df1_index and i in df2_index and i in df3_index):

        df1_ = df1[df1['Index'] == i]
        df2_ = df2[df2['Index'] == i]
        df3_ = df3[df3['Index'] == i]

        df1_ea = list(df1_["Extracted_Answer"])[0]
        df2_ea = list(df2_["Extracted_Answer"])[0]
        df3_ea = list(df3_["Extracted_Answer"])[0]


        df1_q = list(df1_["Question"])[0]
        df2_q = list(df2_["Question"])[0]
        df3_q = list(df3_["Question"])[0]

        df1_p = list(df1_["Prompt"])[0]
        df2_p = list(df2_["Prompt"])[0]
        df3_p = list(df3_["Prompt"])[0]

        p1 = df1_p[len(s_):]
        df1_i = Ins + p1[p1.find("Context:"):] 

        p2 = df2_p[len(s_):]
        df2_i = Ins + p2[p2.find("Context:"):] 

        p3 = df3_p[len(s_):]
        df3_i = Ins + p3[p3.find("Context:"):] 

        df1_gt = list(df1_["Response"])[0]

        all_targets.append(df1_gt)

        all_inputs_mis2.append(df1_i)
        all_inputs_llama3.append(df2_i)
        all_inputs_mis3.append(df3_i)

        inputs.append(df1_q)
        insts.append(df1_i)

        candidates_texts.append([df1_ea,df2_ea,df3_ea])


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


# inputs = inputs[:10]
# candidates_texts = candidates_texts[:10]
# insts = insts[:10]
# all_targets = all_targets[:10]

print()
print("*"*100)
print()
print("Inputs")
print()
print(len(inputs))
print()
print("*"*100)
print()
print()
print("*"*100)
print()
print("candidates_texts")
print()
print(len(candidates_texts))
print()
print("*"*100)
print()


print()
print("Start: blender.rank")
print()

ranks = blender.rank(inputs, candidates_texts, instructions=insts, return_scores=False, batch_size=1)

print()
print(len(ranks))
print()
print("end: blender.rank")
print()



print()
print("start: topk_candidates")
print()
t_start = time.time()
topk_candidates = get_topk_candidates_from_ranks(ranks, candidates_texts, top_k=2)
print()
print(len(topk_candidates))
print()
print("end: topk_candidates")
print()

print()
print("start: fuse_generations")
print()
fuse_generations = blender.fuse(inputs, topk_candidates, instructions=insts, batch_size=1)
t_end = time.time()

total_time = t_end-t_start
print()
print("*"*100)
print()
print("total time")
print(total_time)
print()
print("*"*100)
print()

print()
print(len(fuse_generations))
print()
print("end: fuse_generations")
print()

print()
print("*"*100)
print()
print()
print()
print()
print()
print("*"*100)
print()
print("--------------------------------------Rank-----------------------------------")
print()
print("*"*100)
print()
print()
print()
print(ranks)
print()
print()
print(type(ranks))
print()
print()
print(len(ranks))
print()
print()
print()
print()
print()
print()
print("*"*100)
print()
print("--------------------------------------topk_candidates-----------------------------------")
print()
print("*"*100)
print()
print()
print()
print(topk_candidates)
print()
print()
print(type(topk_candidates))
print()
print()
print(len(topk_candidates))
print()
print()
print()
print()
print()
print()
print("*"*100)
print()
print("--------------------------------------fuse_generations-----------------------------------")
print()
print("*"*100)
print()
print()
print()
print(fuse_generations)
print()
print()
print(type(fuse_generations))
print()
print()
print(len(fuse_generations))
print()
print()
print()
print("*"*100)
print()


topk_candidates = [i[0] for i in topk_candidates]
# fuse_generations = [i[0] for i in fuse_generations]


print()
print("*"*100)
print()
print("after correction")
print()
print("*"*100)
print()


print("*"*100)
print()
print("--------------------------------------topk_candidates-----------------------------------")
print()
print("*"*100)
print()
print()
print()
print(topk_candidates)
print()
print()
print(type(topk_candidates))
print()
print()
print(len(topk_candidates))
print()
print()
print()
print()
print()
print()
print("*"*100)
print()

print("--------------------------------------fuse_generations-----------------------------------")
print()
print("*"*100)
print()
print()
print()
print(fuse_generations)
print()
print()
print(type(fuse_generations))
print()
print()
print(len(fuse_generations))
print()
print()
print()
print("*"*100)
print()



df = {

    "topk_candidates" : topk_candidates, 
    "Pred" : fuse_generations, 
    "GT" : all_targets, 
    "Total_time" : [total_time for i in range(len(topk_candidates))]
     }


print(df)

df = pd.DataFrame(df)

print(df.head(2))

model_name_ = model_name.split("/")[-1]

df.to_csv(f"/workspace/data/ensemble-llm/llm-blen/infer-res/llm-blen-{model_name_}-{dataset_name[i_]}.csv", index = False)
