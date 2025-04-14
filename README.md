# Knowledge Graph from Unstructured Product Catalogue

This project focuses on converting unstructured product descriptions into structured knowledge graphs based on a custom ontology. The result is a system where you can extract, infer, and query product information in a structured and meaningful way.

![Pipeline Diagram](https://github.com/user-attachments/assets/19d32793-fce6-41c9-a57a-84fd1dac9f90)

---

## 🚀 Project Objective

The goal is to:
- Build a **knowledge graph** representing a complete product catalog.
- Convert **unstructured product descriptions** into a structured format (JSON) following a defined schema.
- Use **fine-tuned LLMs** to extract structured information.
- Allow querying the knowledge graph for product insights and metadata.

---

## 🛠️ How to Run the Project

### 0. Generate Synthetic Data
Use the provided script to generate synthetic data for fine-tuning:
```bash
python generating_synthetic_training_data.py
```
Make sure `sample_small_dataset.json` is available as your base data.

### 1. Convert Data to JSONL
Convert the structured dataset into JSONL format required for fine-tuning:
```bash
# Output: training_data_jsonl_format.jsonl
```

### 2. Fine-Tune the Model
Use the script below to fine-tune the base LLM using LoRA or other techniques:
```bash
python finetune_LLAMA2_usingLORA_V5.py
```

- The schema (`product_schema.ttl`) is passed in the prompt during fine-tuning to ensure schema-aware training.

### 3. Inference / Query the Fine-Tuned Model
Run the inference script to:
- Convert new unstructured product descriptions into JSON.
- Query existing data from the knowledge graph.
```bash
python query_after_training.py
```

---

## 🧪 Example Use Case
> Given a paragraph describing a new product, the model outputs a structured JSON based on the provided schema, and updates the knowledge graph accordingly.

---

## ⚙️ Hyperparameters You Can Tune

| Parameter         | Description                          | Example Value   |
|------------------|--------------------------------------|-----------------|
| `learning_rate`  | Learning rate for fine-tuning        | 5e-5            |
| `batch_size`     | Batch size during training           | 8               |
| `epochs`         | Number of fine-tuning epochs         | 3               |
| `max_seq_length` | Maximum sequence length of the input | 512             |

---

## 🧩 Major Steps Overview

1. Create a custom ontology for the product domain.
2. Generate and format synthetic training data using the ontology.
3. Fine-tune a base LLM (e.g., LLaMA2) using this data.
4. Convert raw product text into structured JSON using the fine-tuned LLM.
5. Build and query the knowledge graph using the structured outputs.

---

## 📁 Directory Structure

```
.
├── README.md
├── finetune_LLAMA2_usingLORA_V5.py              # Fine-tuning script
├── generating_synthetic_training_data.py        # Synthetic data generator
├── product_schema.ttl                           # Ontology/schema file
├── sample_small_dataset.json                    # Base structured dataset
├── training_data_jsonl_format.jsonl             # Final training data for LLM
├── query_after_training.py                      # Inference & query script
```

---

## 📌 Notes
- Schema-awareness is critical. We include the ontology during fine-tuning to avoid hallucinations and preserve concept consistency.
- The knowledge graph can be visualized or queried after extraction, enabling rich product analytics and metadata exploration.

---
