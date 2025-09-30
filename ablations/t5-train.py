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
parser.add_argument('--f1', type=int, default = 1)
parser.add_argument('--f2', type=int, default = 1)
parser.add_argument('--b', type=int, default = 4)
parser.add_argument('--e', type=int, default = 5)
parser.add_argument('--lr', type=float, default = 2*0.00001)
parser.add_argument('--ml', type=int, default = 2048)
parser.add_argument('--mtl', type=int, default = 512)

args, unknown_args = parser.parse_known_args()

flag1 = args.f1 
flag2 = args.f2 
batch_size = args.b 
max_input_length = args.ml 
max_target_length = args.mtl
lr = args.lr
epochs = args.e

if(flag1 == 1):
    
    MODEL_NAME = "google-t5/t5-small"

elif(flag1 == 2): 
    
    MODEL_NAME = "google-t5/t5-base"

elif(flag1 == 3): 
    
    MODEL_NAME = "facebook/bart-base"
    


print()
print("*"*100)
print()
print()
print("*"*100)
print()
print("baselines/train.py")
print(f"flag1: {flag1}")
print(f"flag2: {flag2}")
print(f"batch_size: {batch_size}")
print(f"max_input_length: {max_input_length}")
print(f"max_target_length: {max_target_length}")
print(f"lr: {lr}")
print(f"MODEL_NAME: {MODEL_NAME}")
print()
print("*"*100)
print()
print()
print("*"*100)
print()

def load_seq2seq_model(model_name):
    
    checkpoint = "/workspace/data/ensemble-llm/base-line/models/t5-base/checkpoint-400/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model

tokenizer, model = load_seq2seq_model(MODEL_NAME)
    
    
def preprocess_function(examples):
    inputs = [ doc for doc in examples["X"]]
    targets = examples["y"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



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


df = pd.read_csv("/workspace/data/ensemble-llm/pre-train-model/data/pre-data-4774.csv")
df = df.sample(frac=1, random_state=42)


a_mis2 = list(df["mis-2"])
a_llama3 = list(df["llama-3"])
a_mis3 = list(df["mis-3"])

c_mis2 = list(df["con-mis-2"])
c_llama3 = list(df["con-llama-3"])
c_mis3 = list(df["con-mis-3"])

all_targets = list(df["target"])



X = []
y = []
for i in range(0,len(a_mis2 )):
    
    s = gen_prompt(a_mis2[i],a_llama3[i],a_mis3[i], c_mis2[i],c_llama3[i],c_mis3[i])
    
    X.append(s)
    y.append(all_targets[i])
    

df = {
    "X" : X, 
    "y" : y
}

df = pd.DataFrame(df)

print(df.head(2))

df = df.sample(frac=1, random_state=42)

df = df.head(1500)

df = Dataset.from_pandas(df)
dataset = df.train_test_split(test_size=0.2, seed=42)



tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)




folder_name = MODEL_NAME.split("/")[-1]
output_dir = f"/workspace/data/ensemble-llm/base-line/models/{folder_name}"

training_args = TrainingArguments(

output_dir = output_dir,
# evaluation_strategy = "epoch",
save_strategy = "epoch",
learning_rate=lr,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
num_train_epochs=epochs,
weight_decay=0.01,
save_total_limit=2,
report_to="none"
# load_best_model_at_end=True,

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)



t = trainer.train()
e = trainer.evaluate()

trainer.log_metrics("train", t.metrics)
trainer.save_metrics("train", t.metrics)

trainer.log_metrics("eval", e)
trainer.save_metrics("eval", e)



