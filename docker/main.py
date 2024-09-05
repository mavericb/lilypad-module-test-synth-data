import os
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from solcx import compile_source, install_solc, set_solc_version
import re
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Function to load Solidity contracts from the directory
def load_contract_examples(path_to_contracts: str):
    print("Loading contract examples from:", path_to_contracts)
    contract_texts = []
    for filename in os.listdir(path_to_contracts):
        if filename.endswith(".sol"):  # Only load Solidity files
            with open(os.path.join(path_to_contracts, filename), 'r') as file:
                contract_text = file.read()
                contract_texts.append(contract_text)
    print(f"Loaded {len(contract_texts)} contracts from {path_to_contracts}")
    return contract_texts

# Function to clean and add necessary pragma and imports
def add_pragma_and_imports(synthetic_contract):
    pragma_statement = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n\n"
    if not synthetic_contract.startswith("pragma solidity"):
        synthetic_contract = pragma_statement + synthetic_contract

    # Clean up unnecessary or incomplete comment blocks (e.g., /** ... */)
    synthetic_contract = re.sub(r'/\*\*.*?\*/', '', synthetic_contract, flags=re.DOTALL)

    return synthetic_contract

# Create a custom dataset for fine-tuning
class SolidityDataset(Dataset):
    def __init__(self, contracts, tokenizer, max_length=512):
        self.contracts = contracts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contracts)

    def __getitem__(self, idx):
        contract = self.contracts[idx]
        tokenized = self.tokenizer(
            contract,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Set the labels equal to the input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["labels"].squeeze()  # Labels for loss computation
        }

# Function to generate synthetic smart contract
def generate_synthetic_contract(model, tokenizer, prompt, max_length=512):
    print("Generating synthetic contract...")
    device = model.device  # Get the device the model is using (GPU or CPU)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    synthetic_contract = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return add_pragma_and_imports(synthetic_contract)

# Function to validate smart contract using py-solc-x
def validate_contract(contract_code: str):
    print("Validating contract...")
    try:
        set_solc_version('0.8.20')
        compiled_sol = compile_source(contract_code, output_values=["abi", "bin"])
        print("Contract validation successful.")
        return True, None
    except Exception as e:
        print(f"Contract validation failed: {e}")
        return False, str(e)

# Function to categorize errors for targeted fine-tuning
def categorize_error(error_message):
    if "pragma" in error_message or "import" in error_message:
        return "pragma_or_import_issue"
    elif "primary expression" in error_message or re.search(r'!+', error_message):
        return "syntax_noise"  # Handling invalid characters like '!'
    return "general_error"

# Function to perform basic syntactical cleaning
def basic_syntax_check(contract_code):
    # Allow more flexibility: only reject contracts with extreme issues like excessive repeated characters
    if re.search(r'!{5,}', contract_code):  # Check if there are more than 5 consecutive '!'
        return False
    # Ensure there is no hanging multi-line comment
    if re.search(r'/\*\*[^*]*$', contract_code):  # Incomplete comment block
        return False
    return True


print("main.py started")

# Install Solidity compiler  #OK
print("Installing Solidity compiler...")
install_solc(version="0.8.20")

# Seed, number of contracts, and token standard as environment inputs #OK
seed = int(os.environ.get("SEED", 42))  # Default to 42 if SEED is not provided
num_contracts = int(os.environ.get("NUM_CONTRACTS", 1))  # Default to 1 contract if NUM_CONTRACTS is not provided
token_standard = os.environ.get("TOKEN_STANDARD", "ERC-20")  # Default to ERC-20 if TOKEN_STANDARD is not provided

print(f"Environment Variables - SEED: {seed}, NUM_CONTRACTS: {num_contracts}, TOKEN_STANDARD: {token_standard}")

# Set random seed #OK
random.seed(seed)
torch.manual_seed(seed)

input = os.environ.get("INPUT") or "question mark floating in space"
print(f"Input: {input}")

# Use local paths for model and dataset
model_name = "/app/model"
contracts_path = "/app/dataset/contracts"  # Directory where ERC-20 contracts are stored
new_model = "llama-3-8b-synth-data" #OK

print(f"Using model path: {model_name}")
print(f"Using contract path: {contracts_path}")

# QLoRA parameters #OK
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# bitsandbytes parameters #OK
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# TrainingArguments parameters #OK
output_dir = "/app/results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25

# SFT parameters #OK
max_seq_length = None
packing = False
device_map = {"": 0}


# Load model and tokenizer
print("Loading model and tokenizer...")
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)

# Load dataset from local file
# Load the ERC-20 contract examples from the specified directory
try:
    print("Preparing the dataset for fine-tuning...")
    contract_examples = load_contract_examples(contracts_path)
    dataset = SolidityDataset(contract_examples, tokenizer)
    print(f"Dataset loaded successfully from {contracts_path}")
    print(f"Dataset info: {dataset}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise


# Fine-tune the model using the loaded contracts
def finetune_model(model, tokenizer, contract_examples):
    print("Preparing the dataset for fine-tuning...")
    #dataset = SolidityDataset(contract_examples, tokenizer)

    print("Starting the fine-tuning process...")

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=packing,
        )
        print("SFTTrainer initialized successfully")
    except Exception as e:
        print(f"Error initializing SFTTrainer: {e}")
        raise
    print("Fine-tuning completed.")

    # Train model
    try:
        print("Starting model training...")
        trainer.train()
        print("Model training completed successfully")
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

    return model




# Fine-tune the model using the contract examples
model = finetune_model(model, tokenizer, contract_examples)

# Generate the required number of valid contracts
valid_contracts = []
generated_count = 0
error_feedback = []

while len(valid_contracts) < num_contracts:
    print(f"Generating contract {generated_count + 1}/{num_contracts}...")
    prompt = random.choice(contract_examples)[:512]
    synthetic_contract = generate_synthetic_contract(model, tokenizer, prompt)

    # Perform basic syntax validation before compiling
    if not basic_syntax_check(synthetic_contract):
        print("Contract skipped due to obvious syntax issues.")
        continue

    # Validate the synthetic contract
    is_valid, error_message = validate_contract(synthetic_contract)

    if is_valid:
        valid_contracts.append(synthetic_contract)
        generated_count += 1
        print(f"Contract {generated_count} is valid.")
    else:
        error_type = categorize_error(error_message)
        error_feedback.append((error_type, synthetic_contract))
        print(f"Contract {generated_count + 1} is invalid. Error: {error_message}")

        # Skip fine-tuning if the error is due to syntax noise like repeated '!'
        if error_type == "syntax_noise":
            print("Skipping fine-tuning due to noise (syntax).")
            continue

        # Fine-tune the model based on error feedback
        if len(error_feedback) > 0:
            feedback_contracts = [entry[1] for entry in error_feedback if entry[0] == error_type]
            model = finetune_model(model, tokenizer, feedback_contracts)

print(f"Generated {len(valid_contracts)} valid contracts.")