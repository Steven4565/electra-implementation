# ELECTRA: PyTorch Reimplementation

This repository contains a complete PyTorch reimplementation of the ELECTRA model  
as described in the ICLR 2020 paper:

> **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**  
> Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning  
> [Paper Link](https://arxiv.org/abs/2003.10555)

---

## 📐 Project Objective

- Reproduce the ELECTRA-Small model from scratch using **only PyTorch**
- Train on small-scale corpus (e.g. OpenWebText or Wiki subset)
- Fine-tune on selected **GLUE tasks**
- Propose and test our own improvements

---

## 🧠 Model Architecture

### ELECTRA Overview

The ELECTRA architecture consists of:

1. **Generator**  
   - Small BERT-style masked language model  
   - Predicts masked tokens  
   - Only used to corrupt original inputs

2. **Discriminator**  
   - Binary classifier (real or replaced)  
   - Trained on **all tokens** (higher sample efficiency)

### Architecture Diagram

![ELECTRA Architecture](./images/project_steps_en.png)

---

## 🗂️ Project Structure

electra-project/
├── models/
│ ├── generator.py # Generator implementation (MLM-style BERT)
│ ├── discriminator.py # Discriminator for replaced token detection
├── data/
│ └── dataset.py # Token replacement dataset
├── train/
│ ├── pretrain.py # Training loop for ELECTRA pretraining
│ └── finetune.py # Fine-tuning on GLUE tasks
├── images/
│ ├── project_steps_en.png
│ └── project_steps_ko.png
├── utils/
│ └── tokenization.py # BERT tokenizer or WordPiece utilities
├── configs/
│ └── electra_small_config.json
├── README.md
└── requirements.txt

yaml
복사
편집

---

## ⚙️ Training Workflow

```text
1. Raw text corpus
     ↓
2. Apply token masking
     ↓
3. Generator predicts masked tokens
     ↓
4. Replaced input created using generator output
     ↓
5. Discriminator predicts real/fake for each token
     ↓
6. Binary classification loss used to update model
🧪 Evaluation Plan
Fine-tune on:

SST-2 (Sentiment Classification)

RTE or CoLA

Compare with original ELECTRA-Small benchmarks

Propose minor architectural changes (e.g. dropout, reduced heads)

👥 Team Members
Member A: Model architecture & training implementation

Member B: Data preprocessing, token replacement logic

Member C: Evaluation, visualization, fine-tuning analysis

📅 Project Timeline
April 17: Paper selection ✅

May: Implementation & training 🛠️

June 13: Final presentation 🧑‍🏫

June 24: Final report & GitHub submission 📝