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
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--f1', type=int, default = 1)
parser.add_argument('--f2', type=int, default = 1)
parser.add_argument('--b', type=int, default = 4)
parser.add_argument('--e', type=int, default = 1)
parser.add_argument('--lr', type=float, default = 0.00001*2)
parser.add_argument('--n', type=int, default = 15)



args, unknown_args = parser.parse_known_args()


flag = args.f1
flag2 = args.f2
epochs = args.e
batch_size = args.b 
lr = args.lr
n_ = args.n

base_folder_model = "/workspace/data/ensemble-llm/pre-train-model/models/"
base_folder_graph = "/workspace/data/ensemble-llm/pre-train-model/loss_curves/"
base_folder_infer = "/workspace/data/ensemble-llm/pre-train-model/infer-res/"


if(flag == 1):
    
    base_model_name="google-t5/t5-small"

elif(flag == 2): 
    
    base_model_name="google-t5/t5-base"
    
elif(flag == 3): 
    
    base_model_name="google-t5/t5-large"

if(flag == 4):
    
    base_model_name="facebook/bart-base"


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    


print()
print()
print()
print("*"*100)
print()
print("*"*100)
print()
print("*"*100)
print()
print(f"base_model_name: {base_model_name}")
print(f"flag {flag}")
print(f"flag2 {flag2}")
print(f"epochs {epochs}")
print(f"batch_size  {batch_size }")
print(f"lr {lr}")
print(f"model_config: {model_config}")
print(f"n_ : {n_}")
print()
print("*"*100)
print()
print("*"*100)
print()
print("*"*100)
print()
print()
    



class FusionDataset(Dataset):
    def __init__(self, all_outputs, all_confidences, all_targets, tokenizer):
        self.all_outputs = all_outputs
        self.all_confidences = all_confidences
        self.all_targets = all_targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.all_outputs)

    def __getitem__(self, idx):
        llm_outputs = self.all_outputs[idx]
        confidences = self.all_confidences[idx]
        target = self.all_targets[idx]

        # Tokenize input answers
        input_ids_list, attention_masks = [], []
        for out in llm_outputs:
            encoded = self.tokenizer(out, return_tensors="pt", padding=True, truncation=True)
            input_ids_list.append(encoded['input_ids'].squeeze(0))  # (seq_len,)
            attention_masks.append(encoded['attention_mask'].squeeze(0))

        # Tokenize target
        label_ids = self.tokenizer(target, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0)

        return input_ids_list, attention_masks, confidences, label_ids


def fusion_collate_fn(batch):
    input_ids_list_batch = []
    attn_masks_batch = []
    confidences_batch = []
    labels_batch = []

    for input_ids_list, attn_masks_list, confs, label in batch:
        input_ids_list_batch.append(input_ids_list)
        attn_masks_batch.append(attn_masks_list)
        confidences_batch.append(confs)
        labels_batch.append(label.squeeze(0))  # remove batch dim if exists

    # Pad labels to the same length
    labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=model.tokenizer.pad_token_id)

    return input_ids_list_batch, attn_masks_batch, confidences_batch, labels_batch




def extract_answers_and_scores(responses_text):
    extracted_data = []

    # Regex to find each "Question: ... Answer: ... Score: ..." block
    # This pattern captures the "answer" and "score" from each block.
    # It's more robust to match the whole block first
    # and then parse the "answer" and "score" within that block.
    # Using re.findall will give us all non-overlapping matches.
    # We use a non-greedy match for '.*?' to avoid matching across blocks.

    # This regex looks for the pattern starting with "Question:" and ending with "Score: X.X" or "Score: X"
    # and captures the answer and score within each such block.
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




df = pd.read_csv("/workspace/data/ensemble-llm/pre-train-model/data/pre-data-4774.csv")
df = df.sample(frac=1, random_state=42)

df = df.head(1500)

a_mis2 = list(df["mis-2"])
a_llama3 = list(df["llama-3"])
a_mis3 = list(df["mis-3"])

c_mis2 = list(df["con-mis-2"])
c_llama3 = list(df["con-llama-3"])
c_mis3 = list(df["con-mis-3"])

all_targets = list(df["target"])


all_outputs = []
all_confidences = []


for i in range(0,len(a_mis2)):
    
    all_outputs.append([a_mis2[i],a_llama3[i],a_mis3[i]])
    all_confidences.append([c_mis2[i],c_llama3[i],c_mis3[i]])
    

ei = int(len(all_outputs)*0.80)

all_outputs_train = all_outputs[:ei]
all_confidences_train = all_confidences[:ei]
all_targets_train = all_targets[:ei]


all_outputs_test = all_outputs[ei:]
all_confidences_test = all_confidences[ei:]
all_targets_test = all_targets[ei:]



print()
print("*"*100)
print()
print()
print("*"*100)
print()
print(f"df len(df)")
print(f"all_outputs: {len(all_outputs)}")
print(f"all_confidences: {len(all_confidences)}")
print(f"all_targets: {len(all_targets)}")
print()
print("*"*100)
print()
print()
print("*"*100)
print()


train_dataset = FusionDataset(
    all_outputs=all_outputs_train,
    all_confidences=all_confidences_train,
    all_targets=all_targets_train,
    tokenizer=model.tokenizer
)

val_dataset = FusionDataset(
    all_outputs=all_outputs_test,
    all_confidences=all_confidences_test,
    all_targets=all_targets_test,
    tokenizer=model.tokenizer
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fusion_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=fusion_collate_fn)





def train_model(model,train_dataloader,optimizer):
    
    model = model.train()
    
    epoch_loss = 0

    for batch in tqdm(train_dataloader):
        input_ids_list_batch, attn_masks_batch, confidences_batch, labels_batch = batch

        # Process each sample in the batch (since fusion model is not batchified)
        for input_ids_list, attention_masks, raw_confidences, label in zip(input_ids_list_batch, attn_masks_batch, confidences_batch, labels_batch):
            optimizer.zero_grad()

            # Move tensors to device
            input_ids_list = [x.to(device) for x in input_ids_list]
            attention_masks = [x.to(device) for x in attention_masks]
            raw_confidences = [float(c) for c in raw_confidences]


            # labels = label.unsqueeze(0).to(device)

            # decoder_input_ids = model.tokenizer("summarize:", return_tensors="pt").input_ids.to(device)

            decoder_input = "summarize:"
            decoder_input_ids = model.tokenizer(decoder_input, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            # Tokenize the target (label) properly
            if isinstance(label, str):
                labels = model.tokenizer(label, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            else:
                labels = label.unsqueeze(0).to(device)


            output = model(input_ids_list, attention_masks, raw_confidences, decoder_input_ids=None, labels=labels)

            loss = output.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


    print()
    print(f"train Loss: {epoch_loss:.4f}")
    print()
    
    return epoch_loss



def val_model(model,val_dataloader):
    
    model = model.eval()
    
    epoch_loss = 0

    for batch in tqdm(val_dataloader):
        input_ids_list_batch, attn_masks_batch, confidences_batch, labels_batch = batch

        # Process each sample in the batch (since fusion model is not batchified)
        for input_ids_list, attention_masks, raw_confidences, label in zip(input_ids_list_batch, attn_masks_batch, confidences_batch, labels_batch):
            optimizer.zero_grad()

            # Move tensors to device
            input_ids_list = [x.to(device) for x in input_ids_list]
            attention_masks = [x.to(device) for x in attention_masks]
            raw_confidences = [float(c) for c in raw_confidences]


            # labels = label.unsqueeze(0).to(device)

            # decoder_input_ids = model.tokenizer("summarize:", return_tensors="pt").input_ids.to(device)

            decoder_input = "summarize:"
            decoder_input_ids = model.tokenizer(decoder_input, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            # Tokenize the target (label) properly
            if isinstance(label, str):
                labels = model.tokenizer(label, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            else:
                labels = label.unsqueeze(0).to(device)

            with torch.no_grad():
                
                output = model(input_ids_list, attention_masks, raw_confidences, decoder_input_ids=None, labels=labels)

            loss = output.loss
            epoch_loss += loss.item()


    print()
    print(f"val Loss: {epoch_loss:.4f}")
    print()
    
    return epoch_loss


model_name_ = base_model_name.split("/")[-1]

if(epochs > 1):


  model_ = f"{model_name_}-{model_config}-epoch-{epochs-1}-v2.pth"
  

  checkpoint = torch.load(os.path.join(base_folder_model,model_))
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


E = []
T_Loss = []
E_Loss = []


t_start = time.time()

for e in range(epochs,epochs+n_): 
    
    print()
    print("*"*100)
    print()
    print(f"Epoch Number: {e}")
    print()
    train_loss = train_model(model,train_dataloader,optimizer)
    val_loss = val_model(model,val_dataloader)
    print()
    print("*"*100)
    print()
    E.append(e+1)
    T_Loss.append(train_loss)
    E_Loss.append(val_loss)
    
    
    df = {"Epoch" : E, 
         "Train_Loss" : T_Loss, 
         "Eval_Loss" : E_Loss}
    
    
    df = pd.DataFrame(df)
    
    
    df.to_csv(os.path.join(base_folder_graph, f"{model_name_}-{model_config}-epoch-{e}-v2.csv"), index = False)
    
    
    
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()

}



torch.save(checkpoint, os.path.join(base_folder_model,f"{model_name_}-{model_config}-epoch-{e}-v2.pth"))


t_end = time.time()




print()
print()
print("*"*100)
print("*"*100)
print()
print()
print()
print()
print("*"*100)
print("*"*100)
print()
print()
print(f"total time: {t_end - t_start}")
print()
print()
print("*"*100)
print("*"*100)
print()
print()
print()
print()
print("*"*100)
print("*"*100)
print()
print()