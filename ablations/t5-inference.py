import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from datasets import Dataset
import pandas as pd 
import os
import glob
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--f1', type=int, default = 2)
parser.add_argument('--i', type=int, default = 0)
parser.add_argument('--ml', type=int, default = 2048)
parser.add_argument('--mtl', type=int, default = 512)


args, unknown_args = parser.parse_known_args()

flag1 = args.f1
i_ = args.i
max_input_length = args.ml 
max_target_length = args.mtl


if(flag1 == 1):
    
    MODEL_NAME = "google-t5/t5-small"

elif(flag1 == 2): 
    
    MODEL_NAME = "google-t5/t5-base"

elif(flag1 == 3): 
    
    MODEL_NAME = "facebook/bart-base"


print()
print("*"*100)
print("*"*100)
print()
print(f"MODEL_NAME: {MODEL_NAME}")
print()
print("*"*100)
print("*"*100)
print()

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



folder_name = MODEL_NAME.split("/")[-1]
output_dir = f"/workspace/data/ensemble-llm/base-line/models/{folder_name}"
base_folder_infer = "/workspace/data/ensemble-llm/pre-train-model/infer-res/"






def find_latest_checkpoint(output_dir):

    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {output_dir}")
        return None


    def extract_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else -1

    sorted_checkpoints = sorted(checkpoint_dirs, key=extract_step, reverse=True)
    
    latest_checkpoint = sorted_checkpoints[0]
    
    return latest_checkpoint

latest_checkpoint = find_latest_checkpoint(output_dir)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint )
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print()
print()
print(loaded_model)
print()
print()



def gen_prompt(LLM1_Output,LLM2_Output,LLM3_Output, 
              Confidence_Score_LLM1,Confidence_Score_LLM2,Confidence_Score_LLM3 ):

    s = f"""**Instructions:**

    1.  **Analyze the Inputs:** Carefully review the three LLM outputs provided below, paying close attention to the content, phrasing, and any subtle differences in their responses.
    2.  **Evaluate Confidence:** Consider the confidence score assigned to each LLM output. Higher scores indicate greater certainty in the response. Give more weight to outputs with higher confidence scores, but still consider the content of all outputs.
    3.  **Identify Agreements and Disagreements:**  Pinpoint the areas where the LLM outputs align and where they diverge. Note any significant contradictions or complementary details.
    4.  **Synthesize and Generate:** Based on your analysis and the confidence scores, generate a new output that:
        *   Integrates the most relevant and accurate information from the three LLM outputs.
        *   Prioritizes information from outputs with higher confidence scores.
        *   Resolves any discrepancies or contradictions in a logical and coherent manner.
        *   Maintains a consistent tone and style throughout the generated output.
        *   Provides a well-rounded and comprehensive response to the original query (implicitly represented by the combined input).
    5.  **Output Format:** Present your synthesized output clearly and concisely.

    **LLM Outputs with Confidence Scores:**

    **LLM 1 Output:**
    {LLM1_Output}
    **Confidence Score (LLM1):** {Confidence_Score_LLM1}

    **LLM 2 Output:**
    {LLM2_Output}
    **Confidence Score (LLM2):** {Confidence_Score_LLM2}

    **LLM 3 Output:**
    {LLM3_Output}
    **Confidence Score (LLM3):** {Confidence_Score_LLM3}

    **Your Synthesized Output:**

    """
    
    return s 


dataset_name = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']
models = ["mistralai-2", "llama-3-hf", "mistralai-3", "llama-2-hf" , "Saul" , "quen-2" ]
base_paths = ["/workspace/data/ensemble-llm/base-res-test/mistralai-2/", 
             "/workspace/data/ensemble-llm/base-res-test/llama-3-hf/" , 
             "/workspace/data/ensemble-llm/base-res-test/mistralai-3/", 
             "/workspace/data/ensemble-llm/base-res-test/llama-2-hf/", 
             "/workspace/data/ensemble-llm/base-res-test/Saul/", 
             "/workspace/data/ensemble-llm/base-res-test/quen-2/"]


all_outputs_mis2= []
all_confidences_mis2= []

all_outputs_llama3= []
all_confidences_llama3 = []


all_outputs_mis3 = []
all_confidences_mis3= []

all_targets = []



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


X = []


for i in range(0,len(all_outputs_mis2)): 
    
    s =  gen_prompt(all_outputs_mis2[i],
                    all_outputs_llama3[i],
                    all_outputs_mis3[i], 
                    all_confidences_mis2[i],
                    all_confidences_llama3[i],
                    all_confidences_mis3[i])
    X.append(s)
    
    

def summarize(input_text):

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_length).to(loaded_model.device)

    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model.generate(inputs["input_ids"], max_length=max_target_length)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


Pred_Ans = []


arr = []
for i in range(0,len(X)): 
    
    s = summarize(X[i])
    Pred_Ans.append(s)
    

df = {"mis2" : all_outputs_mis2, 
      "llama3" : all_outputs_llama3, 
      "mis3" : all_outputs_mis3, 
      "Pred" : Pred_Ans, 
     "GT" :  all_targets}

df = pd.DataFrame(df)

df.to_csv(os.path.join(base_folder_infer, f"baseline-method-{folder_name}-{dataset_name[i_]}.csv"), index = False)

print()
print()
print()
print(f" saved: baseline-method-{folder_name}-{dataset_name[i_]}.csv")
print()
print()
print()
