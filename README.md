# ai_code_detector
This is the code i ran on kaggle to train [model](https://huggingface.co/santh-cpu/ai_code_detect).

## Architecture

Semantic Engine: Salesforce/codet5-base
Statistical Extraction: microsoft/codebert-base-mlm (Calculates Entropy and Log-Rank across 256 tokens)
Fusion Network: 1D CNN for temporal feature extraction + Dense Feed-Forward Classifier

## Performance Metrics

Trained on a polyglot dataset (Python, Java, C++) to prevent single-language overfitting.

Training Validation F1: 0.9861

Unseen SemEval-2026 Audit (F1): 0.9921

Overall Accuracy: 99.20%

(on the dataset's validation set)

## Datasets used

[dataset1](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
[dataset2](https://huggingface.co/datasets/HungPhamBKCS/magecode-dataset)

## How to run
```python
from huggingface_hub import hf_hub_download
import torch

weights_path = hf_hub_download(repo_id="santh-cpu/ai_code_detect", filename="pytorch_model.bin")

model = TemporalFusionClassifier(base_model)
model.load_state_dict(torch.load(weights_path))
model.eval()
```

## Example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, T5EncoderModel, AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import hf_hub_download

class TemporalFusionClassifier(nn.Module):
    def __init__(self, base, metric_dim=7):
        super().__init__()
        self.base = base
        h = base.config.hidden_size 
        
        self.metric_cnn = nn.Sequential(
            nn.Conv1d(metric_dim, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(h + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)
        )

    def forward(self, input_ids, attention_mask, metric_vector):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state            
        mask = attention_mask.unsqueeze(-1).float()          
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-4)
        
        cnn_features = self.metric_cnn(metric_vector.transpose(1, 2)).squeeze(-1) 
        return self.classifier(torch.cat([pooled, cnn_features], dim=1)) 

class AICodeDetector:
    def __init__(self, repo_id="santh-cpu/ai_code_detect"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_len = 256
        
        self.cb_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        self.cb_model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(self.device).eval()

        self.t5_tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
        base_t5 = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
        
        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        self.detector = TemporalFusionClassifier(base_t5).to(self.device)
        self.detector.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.detector.eval()

    def analyze(self, code_snippet):
        with torch.no_grad():
            cb_in = self.cb_tokenizer(code_snippet, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len).to(self.device)
            logits = self.cb_model(**cb_in).logits
                
            seq_len = cb_in["attention_mask"][0].sum().item()
            metrics = torch.zeros((1, self.max_len, 7), device=self.device)
            
            if seq_len > 1:
                seq_logits = logits[0:1, :seq_len-1, :]
                seq_labels = cb_in["input_ids"][0:1, 1:seq_len]
                probs = F.softmax(seq_logits, dim=-1)
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                ranks = (torch.argsort(seq_logits, dim=-1, descending=True) == seq_labels.unsqueeze(-1)).nonzero(as_tuple=True)[2].view(1, -1) + 1
                
                token_metrics = torch.stack([
                    torch.log(probs.gather(2, seq_labels.unsqueeze(-1)).squeeze(-1) + 1e-9), 
                    torch.log(ranks.float()), 
                    entropy,
                    (ranks <= 10).float(),
                    ((ranks > 10) & (ranks <= 100)).float(),
                    ((ranks > 100) & (ranks <= 1000)).float(),
                    (ranks > 1000).float()
                ], dim=-1)
                metrics[0, :token_metrics.size(1), :] = token_metrics[0]
            
            clean_metrics = torch.nan_to_num(metrics, nan=0.0, posinf=10.0, neginf=-100.0)
            t5_in = self.t5_tokenizer(code_snippet, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len).to(self.device)
            prob = torch.sigmoid(self.detector(t5_in["input_ids"], t5_in["attention_mask"], clean_metrics)).item()
            
            return {"prediction": "AI Generated" if prob > 0.5 else "Human Written", "ai_probability": round(prob * 100, 2)}

if __name__ == "__main__":
    detector = AICodeDetector()
    sample = "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b"
    print(detector.analyze(sample))
```
