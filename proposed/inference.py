# import subprocess



# commands = [
# "pip install datasets==2.16.1", 
# "pip install scikit-learn numpy pandas", 
# "pip install transformers==4.30", 
# "pip install -q -U trl accelerate", 
# "pip uninstall -y apex", 
# "pip install -U sentence-transformers"
# ]

# for cmd in commands:
#     subprocess.run(cmd, shell=True)
    

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import  get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd 
import numpy as np
import re
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--f1', type=int, default = 1)
parser.add_argument('--f2', type=int, default = 1)
parser.add_argument('--i', type=int, default = 0)
parser.add_argument('--e', type=int, default = 15)




args, unknown_args = parser.parse_known_args()


flag = args.f1
flag2 = args.f2
i_ = args.i
epochs = args.e


base_folder_model = "/workspace/data/Momojit/ensemble-llm/pre-train-model/models/"
base_folder_graph = "/workspace/data/Momojit/ensemble-llm/pre-train-model/loss_curves/"
base_folder_infer = "/workspace/data/Momojit/ensemble-llm/pre-train-model/infer-res/"


if(flag == 1):
    
    base_model_name="google-t5/t5-small"

elif(flag == 2): 
    
    base_model_name="google-t5/t5-base"
    
elif(flag == 3): 
    
    base_model_name="google-t5/t5-large"


print()
print()
print()
print("*"*100)
print()
print(f"base_model_name: {base_model_name}")
print(f"flag {flag}")
print(f"flag2 {flag2}")
print()
print("*"*100)
print()
print()

if(flag2 == 0):
    
    class ConfidenceAwareFusionModel(nn.Module):
        def __init__(self, encoder_decoder_model_name=base_model_name, sentence_encoder_name="all-mpnet-base-v2"):
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_decoder_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(encoder_decoder_model_name)
            self.sentence_encoder = SentenceTransformer(sentence_encoder_name)
    
            hidden_size = self.model.config.hidden_size
            self.conf_proj = nn.Linear(1, hidden_size)
            self.bias_mlp = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
    
        def compute_semantic_agreement(self, answers: List[str]) -> torch.Tensor:
            embeddings = self.sentence_encoder.encode(answers, convert_to_tensor=True)  # (n, d)
            sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)  # (n, n)
            return sim_matrix
    
        # def compute_final_confidence_scores(self, raw_confidences: List[float], sem_sim_matrix: torch.Tensor, lambda_param=0.7) -> List[float]:
        #     n = len(raw_confidences)
        #     norm_conf = torch.tensor(raw_confidences) / max(raw_confidences)
        #     final_scores = []
    
        #     for i in range(n):
        #         sem_agree = (torch.sum(sem_sim_matrix[i]) - 1) / (n - 1)
        #         score = lambda_param * norm_conf[i] + (1 - lambda_param) * sem_agree
        #         final_scores.append(score.item())
    
        #     return final_scores  # List[float]
    
        def compute_final_confidence_scores(self, raw_confidences: List[float], sem_sim_matrix: torch.Tensor, lambda_param=0.7) -> List[float]:
            n = len(raw_confidences)
            norm_conf = torch.tensor(raw_confidences) / max(raw_confidences)
            final_scores = []
    
            for i in range(n):
                if n == 1:
                    sem_agree = 1.0  # or 0.0, or just use norm_conf[i]
                else:
                    sem_agree = (torch.sum(sem_sim_matrix[i]) - 1) / (n - 1)
                score = lambda_param * norm_conf[i] + (1 - lambda_param) * sem_agree
                final_scores.append(score.item())
    
            return final_scores
    
        # def encode_with_confidence(self, input_ids_list, attention_masks, scores):
        #     all_embeddings, all_masks = [], []
        #     for input_ids, attn_mask, ai in zip(input_ids_list, attention_masks, scores):
        #         token_embeddings = self.model.encoder.embed_tokens(input_ids)  # (1, seq_len, dim)
        #         conf_vec = self.conf_proj(torch.tensor([[ai]], device=token_embeddings.device))  # (1, 1, dim)
        #         conf_vec = conf_vec.expand_as(token_embeddings)
        #         fused = token_embeddings + conf_vec
        #         all_embeddings.append(fused)
        #         all_masks.append(attn_mask)
    
        #     return torch.cat(all_embeddings, dim=1), torch.cat(all_masks, dim=1)
    
    
        def encode_with_confidence(self, input_ids_list, attention_masks, scores):
            all_embeddings, all_masks = [], []
            for input_ids, attn_mask, ai in zip(input_ids_list, attention_masks, scores):
                token_embeddings = self.model.encoder.embed_tokens(input_ids)  # (1, seq_len, dim)
                conf_vec = self.conf_proj(torch.tensor([[ai]], device=token_embeddings.device))  # (1, 1, dim)
                conf_vec = conf_vec.expand_as(token_embeddings)
                fused = token_embeddings + conf_vec
                all_embeddings.append(fused)
                all_masks.append(attn_mask)
    
            embeddings = [x.squeeze(0) for x in all_embeddings]  # [seq_len, hidden_dim]
            masks = [x.squeeze(0) for x in all_masks]            # [seq_len]
    
            # Pad
            padded_embeddings = pad_sequence(embeddings, batch_first=True)  # [num_segments, max_seq_len, hidden_dim]
            padded_masks = pad_sequence(masks, batch_first=True)            # [num_segments, max_seq_len]
    
            # Restore batch dim
            padded_embeddings = padded_embeddings.unsqueeze(0)  # [1, num_segments * max_seq_len, hidden_dim]
            padded_masks = padded_masks.unsqueeze(0)            # [1, num_segments * max_seq_len]
    
            # Finally, flatten across segments
            final_embeddings = padded_embeddings.view(1, -1, padded_embeddings.size(-1))  # [1, total_tokens, hidden_dim]
            final_masks = padded_masks.view(1, -1)                                        # [1, total_tokens]
    
            return final_embeddings, final_masks
    
    
    
        def forward(self, input_ids_list, attention_masks, raw_confidences, decoder_input_ids, labels=None):
            # Convert LLM outputs to decoded strings
            answers = [self.tokenizer.decode(ids[0], skip_special_tokens=True) for ids in input_ids_list]
    
            sem_sim = self.compute_semantic_agreement(answers)
            final_scores = self.compute_final_confidence_scores(raw_confidences, sem_sim)
    
            # Encoder
            fused_embeddings, fused_mask = self.encode_with_confidence(input_ids_list, attention_masks, final_scores)
    
            output = self.model(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
    
            return output
    
    
    
        def generate(self, input_ids_list, attention_masks, raw_confidences, max_length=128):
            answers = [self.tokenizer.decode(ids[0], skip_special_tokens=True) for ids in input_ids_list]
            sem_sim = self.compute_semantic_agreement(answers)
            final_scores = self.compute_final_confidence_scores(raw_confidences, sem_sim)
    
            fused_embeddings, fused_mask = self.encode_with_confidence(input_ids_list, attention_masks, final_scores)
    
            generated_ids = self.model.generate(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        
    model_config = "Con_Model"
    model = ConfidenceAwareFusionModel().to(device)
    model_name_ = base_model_name.split("/")[-1]
    checkpoint = torch.load(os.path.join(base_folder_model,f"{model_name_}-{model_config}-epoch-{epochs}-v2.pth")) #     f"{model_name_}-{model_config}-epoch-{epochs-1}.pth"
    model.load_state_dict(checkpoint['model_state_dict'])   
    
    
    
else: 
    
    class ConfidenceAwareFusionModelALL(nn.Module):
        def __init__(self, encoder_decoder_model_name=base_model_name, sentence_encoder_name="all-mpnet-base-v2"):
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_decoder_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(encoder_decoder_model_name)
            self.sentence_encoder = SentenceTransformer(sentence_encoder_name)
    
            hidden_size = self.model.config.d_model
            self.conf_proj = nn.Linear(1, hidden_size)
            self.bias_mlp = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
    
            # Store original forward methods
            self.original_decoder_forward = self.model.decoder.forward
            self.model.decoder.forward = self.decoder_forward_with_confidence_bias
    
        def decoder_forward_with_confidence_bias(self,
                                              input_ids=None,
                                              attention_mask=None,
                                              encoder_hidden_states=None,
                                              encoder_attention_mask=None,
                                              inputs_embeds=None,
                                              past_key_values=None,
                                              **kwargs):
    
            # First get the standard decoder outputs
            decoder_outputs = self.original_decoder_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                **kwargs
            )
    
            # Get the confidence scores from the encoder
            if not hasattr(self, 'current_confidence_scores') or self.current_confidence_scores is None:
                return decoder_outputs
    
            batch_size = decoder_outputs.last_hidden_state.size(0)
    
            # Modify cross-attention in each layer
            for layer in self.model.decoder.block:
                cross_attn = layer.layer[1].EncDecAttention
    
                # Get query states
                query_states = cross_attn.q(decoder_outputs.last_hidden_state)
    
                # Compute confidence bias
                # Expand confidence scores to match batch and sequence dimensions
                conf_scores = self.current_confidence_scores.to(query_states.device)
                conf_scores = conf_scores.view(1, 1, -1).expand(batch_size, query_states.size(1), -1)  # (batch, tgt_len, src_len)
    
                # Project confidence scores
                conf_projected = self.conf_proj(conf_scores.unsqueeze(-1)).squeeze(-1)  # (batch, tgt_len, src_len, d_model)
    
                # Compute dot product between query and confidence projection
                beta = torch.matmul(query_states.unsqueeze(2), conf_projected.transpose(-1, -2)).squeeze(2)  # (batch, tgt_len, src_len)
    
                # Apply MLP to get bias term
                bias = self.bias_mlp(beta.unsqueeze(-1)).squeeze(-1)  # (batch, tgt_len, src_len)
    
                # Modify attention weights
                if hasattr(cross_attn, 'attention_scores'):
                    cross_attn.attention_scores = cross_attn.attention_scores + bias
    
            return decoder_outputs
    
        # Rest of your methods remain the same...
        def compute_semantic_agreement(self, answers: List[str]) -> torch.Tensor:
            embeddings = self.sentence_encoder.encode(answers, convert_to_tensor=True)
            sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
            return sim_matrix
    
        def compute_final_confidence_scores(self, raw_confidences: List[float], sem_sim_matrix: torch.Tensor, lambda_param=0.7) -> torch.Tensor:
            n = len(raw_confidences)
            norm_conf = torch.tensor(raw_confidences) / max(raw_confidences)
            final_scores = torch.zeros_like(norm_conf)
    
            for i in range(n):
                sem_agree = (torch.sum(sem_sim_matrix[i]) - 1) / (n - 1)
                final_scores[i] = lambda_param * norm_conf[i] + (1 - lambda_param) * sem_agree
    
            return final_scores
    
    
        def encode_with_confidence(self, input_ids_list, attention_masks, scores):
            all_embeddings, all_masks = [], []
            for input_ids, attn_mask, ai in zip(input_ids_list, attention_masks, scores):
                token_embeddings = self.model.encoder.embed_tokens(input_ids)  # (1, seq_len, dim)
                conf_vec = self.conf_proj(torch.tensor([[ai]], device=token_embeddings.device))  # (1, 1, dim)
                conf_vec = conf_vec.expand_as(token_embeddings)
                fused = token_embeddings + conf_vec
                all_embeddings.append(fused)
                all_masks.append(attn_mask)
    
            embeddings = [x.squeeze(0) for x in all_embeddings]  # [seq_len, hidden_dim]
            masks = [x.squeeze(0) for x in all_masks]            # [seq_len]
    
            # Pad
            padded_embeddings = pad_sequence(embeddings, batch_first=True)  # [num_segments, max_seq_len, hidden_dim]
            padded_masks = pad_sequence(masks, batch_first=True)            # [num_segments, max_seq_len]
    
            # Restore batch dim
            padded_embeddings = padded_embeddings.unsqueeze(0)  # [1, num_segments * max_seq_len, hidden_dim]
            padded_masks = padded_masks.unsqueeze(0)            # [1, num_segments * max_seq_len]
    
            # Finally, flatten across segments
            final_embeddings = padded_embeddings.view(1, -1, padded_embeddings.size(-1))  # [1, total_tokens, hidden_dim]
            final_masks = padded_masks.view(1, -1)                                        # [1, total_tokens]
    
            return final_embeddings, final_masks
    
        def forward(self, input_ids_list, attention_masks, raw_confidences, decoder_input_ids, labels=None):
            answers = [self.tokenizer.decode(ids[0], skip_special_tokens=True) for ids in input_ids_list]
            sem_sim = self.compute_semantic_agreement(answers)
            final_scores = self.compute_final_confidence_scores(raw_confidences, sem_sim)
    
            fused_embeddings, fused_mask = self.encode_with_confidence(input_ids_list, attention_masks, final_scores)
    
            output = self.model(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
    
            if hasattr(self, 'current_confidence_scores'):
                del self.current_confidence_scores
    
            return output
    
        def generate(self, input_ids_list, attention_masks, raw_confidences, max_length=128):
            answers = [self.tokenizer.decode(ids[0], skip_special_tokens=True) for ids in input_ids_list]
            sem_sim = self.compute_semantic_agreement(answers)
            final_scores = self.compute_final_confidence_scores(raw_confidences, sem_sim)
    
            fused_embeddings, fused_mask = self.encode_with_confidence(input_ids_list, attention_masks, final_scores)
    
            generated_ids = self.model.generate(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
    
            if hasattr(self, 'current_confidence_scores'):
                del self.current_confidence_scores
    
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    model_config = "Con_Model_All"
    model = ConfidenceAwareFusionModelALL().to(device)
    model_name_ = base_model_name.split("/")[-1]
    checkpoint = torch.load(os.path.join(base_folder_model,f"{model_name_}-{model_config}-epoch-{epochs}-v2.pth")) #     f"{model_name_}-{model_config}-epoch-{epochs-1}.pth"
    model.load_state_dict(checkpoint['model_state_dict'])       

    




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
base_paths = ["/workspace/data/Momojit/ensemble-llm/base-res-test/mistralai-2/", 
             "/workspace/data/Momojit/ensemble-llm/base-res-test/llama-3-hf/" , 
             "/workspace/data/Momojit/ensemble-llm/base-res-test/mistralai-3/", 
             "/workspace/data/Momojit/ensemble-llm/base-res-test/llama-2-hf/", 
             "/workspace/data/Momojit/ensemble-llm/base-res-test/Saul/", 
             "/workspace/data/Momojit/ensemble-llm/base-res-test/quen-2/"]



    
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


all_outputs= []
all_confidences= []


n = len(all_outputs_mis2)
for i in range(n): 
    
    all_outputs.append([all_outputs_mis2[i] , all_outputs_llama3[i] , all_outputs_mis3[i]])
    all_confidences.append([all_confidences_mis2[i],all_confidences_llama3[i],all_confidences_mis3[i]])
   
    

Pred_Ans = []
ans_mis2 = []
ans_llama3 = []
ans_mis3 = []

for i in range(0,n): 
    


    input_ids_list, attn_masks = [], []
    
    for out in all_outputs[i]:
        
        encoded = model.tokenizer(out, return_tensors="pt",padding=True, truncation=True).to(device)
        input_ids_list.append(encoded['input_ids'])
        attn_masks.append(encoded['attention_mask'])
    
#     labels = model.tokenizer(all_targets[i], return_tensors="pt",padding=True, truncation=True).input_ids.to(device)
    decoder_input_ids = model.tokenizer("summarize:", return_tensors="pt").input_ids.to(device)
        

    fused_answer = model.generate(input_ids_list, attn_masks, all_confidences[i])
    
    Pred_Ans.append(fused_answer)
    ans_mis2.append(all_outputs[i][0])
    ans_llama3.append(all_outputs[i][1])
    ans_mis3.append(all_outputs[i][2])
    
    
df = {"mis2" : ans_mis2, 
      "llama3" : ans_llama3, 
      "mis3" : ans_mis3, 
      "Pred" : Pred_Ans, 
     "GT" :  all_targets}

df = pd.DataFrame(df)

df.to_csv(os.path.join(base_folder_infer, f"{model_name_}-pred-{model_config}-{dataset_name[i_]}.csv"), index = False)
