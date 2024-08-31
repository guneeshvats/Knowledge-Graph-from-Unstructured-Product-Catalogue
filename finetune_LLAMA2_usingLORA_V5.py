# C:/Users/Guneesh Vats/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb
from transformers.integrations import WandbCallback

# Set the local path where the model is saved
local_model_path = "/home/Proxzar/KG_V1/Pretrained_Model/Llama-2-7b-chat-hf"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Ensure the output directory exists
output_dir = "/home/Proxzar/KG_V1/Fine_Tuned_Model_1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the model and tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(local_model_path, device = device)
model.config.use_cache = False
model.config.pretraining_tp = 1


# Load the dataset
dataset = load_dataset('json', data_files='/home/Proxzar/KG_V1/training_data.jsonl', split='train')

# Tokenization function with chat template application
def tokenize_function(examples):
    try:
        batch_inputs = []
        batch_labels = []

        for example in examples['messages']:
            messages = example  # No need to index [0] as we assume messages are directly in the list

            chat = {"role": "system", "content": messages[0]['content']}
            user_message = {"role": "user", "content": messages[1]['content']}
            assistant_message = {"role": "assistant", "content": messages[2]['content']}

            chat = [chat, user_message, assistant_message]
            tokenizer.use_default_system_prompt = False

            # Check if `apply_chat_template` method exists
            if hasattr(tokenizer, 'apply_chat_template'):
                tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
            else:
                raise AttributeError("The tokenizer does not have the method 'apply_chat_template'.")

            print("Chat template after applying:", tokenized_chat)

            inputs = [message['content'] for message in chat if message['role'] == 'user']
            outputs = [message['content'] for message in chat if message['role'] == 'assistant']

            # Tokenize inputs
            tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

            # Tokenize outputs
            tokenized_labels = tokenizer(text_target=outputs, padding="max_length", truncation=True, max_length=512)

            batch_inputs.append(tokenized_inputs['input_ids'][0])
            batch_labels.append(tokenized_labels['input_ids'][0])

        return {"input_ids": batch_inputs, "labels": batch_labels}

    except Exception as e:
        print(f"Error during tokenization: {e}")
        print(f"Example causing error: {examples}")
        raise e

print("Starting tokenization of the dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print("Split dataset into training and validation sets")

# Define LoRA Configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
print("Loaded model with LoRA configuration")

# Apply LoRA Configuration
model = get_peft_model(model, lora_config)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    weight_decay=0.001,
    save_steps=1000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    report_to="wandb"
)

# Initialize W&B
wandb.init(project="fine-tuning-llama2", entity="Proxzar")

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    callbacks=[WandbCallback()],  # W&B callback
    packing=False
)

try:
    # Move model to the appropriate device
    model.to(device)

    # Fine-Tune the Model
    print("Starting the training process...")
    trainer.train()
    print("Training completed successfully.")

    # Save the fine-tuned model
    print("Saving the model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuned LLaMA 2 model with LoRA and tokenizer have been saved.")
except Exception as e:
    print(f"An error occurred: {e}")
