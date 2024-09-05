import os
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from solcx import compile_source, install_solc, set_solc_version
import re
from peft import LoraConfig
from trl import SFTTrainer
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_contract_examples(path_to_contracts: str) -> List[str]:
    logger.info(f"Loading contract examples from: {path_to_contracts}")
    contract_texts = []
    for filename in os.listdir(path_to_contracts):
        if filename.endswith(".sol"):
            with open(os.path.join(path_to_contracts, filename), 'r') as file:
                contract_texts.append(file.read())
    logger.info(f"Loaded {len(contract_texts)} contracts from {path_to_contracts}")
    return contract_texts

def add_pragma_and_imports(synthetic_contract: str) -> str:
    pragma_statement = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n\n"
    if not synthetic_contract.startswith("pragma solidity"):
        synthetic_contract = pragma_statement + synthetic_contract
    synthetic_contract = re.sub(r'/\*\*.*?\*/', '', synthetic_contract, flags=re.DOTALL)
    return synthetic_contract

class SolidityDataset(Dataset):
    def __init__(self, contracts: List[str], tokenizer, max_length: int = 512):
        self.contracts = contracts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.contracts)

    def __getitem__(self, idx: int) -> dict:
        contract = self.contracts[idx]
        tokenized = self.tokenizer(
            contract,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {k: v.squeeze() for k, v in tokenized.items()}

def generate_synthetic_contract(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    logger.info("Generating synthetic contract...")
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    synthetic_contract = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return add_pragma_and_imports(synthetic_contract)

def validate_contract(contract_code: str) -> Tuple[bool, str]:
    logger.info("Validating contract...")
    try:
        set_solc_version('0.8.20')
        compile_source(contract_code, output_values=["abi", "bin"])
        logger.info("Contract validation successful.")
        return True, None
    except Exception as e:
        logger.error(f"Contract validation failed: {e}")
        return False, str(e)

def categorize_error(error_message: str) -> str:
    if "pragma" in error_message or "import" in error_message:
        return "pragma_or_import_issue"
    elif "primary expression" in error_message or re.search(r'!+', error_message):
        return "syntax_noise"
    return "general_error"

def basic_syntax_check(contract_code: str) -> bool:
    if re.search(r'!{5,}', contract_code):
        return False
    if re.search(r'/\*\*[^*]*$', contract_code):
        return False
    return True

def main():
    logger.info("Smart Contract Generator started")

    # Environment setup
    install_solc(version="0.8.20")
    seed = int(os.environ.get("SEED", 42))
    num_contracts = int(os.environ.get("NUM_CONTRACTS", 1))
    token_standard = os.environ.get("TOKEN_STANDARD", "ERC-20")

    logger.info(f"Environment Variables - SEED: {seed}, NUM_CONTRACTS: {num_contracts}, TOKEN_STANDARD: {token_standard}")

    random.seed(seed)
    torch.manual_seed(seed)

    # Paths and configurations
    model_name = "/app/model"
    contracts_path = "/app/dataset/contracts"
    output_dir = "/app/results"

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
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

    # Load dataset
    contract_examples = load_contract_examples(contracts_path)
    dataset = SolidityDataset(contract_examples, tokenizer)

    # Training configurations
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    # Fine-tuning function
    def finetune_model(model, dataset):
        logger.info("Starting the fine-tuning process...")
        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_config,
                dataset_text_field="text",
                max_seq_length=None,
                tokenizer=tokenizer,
                args=training_arguments,
                packing=False,
            )
            trainer.train()
            logger.info("Fine-tuning completed successfully")
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise
        return model

    # Generate contracts
    valid_contracts = []
    error_feedback = []

    while len(valid_contracts) < num_contracts:
        logger.info(f"Generating contract {len(valid_contracts) + 1}/{num_contracts}...")
        prompt = random.choice(contract_examples)[:512]
        synthetic_contract = generate_synthetic_contract(model, tokenizer, prompt)

        if not basic_syntax_check(synthetic_contract):
            logger.info("Contract skipped due to obvious syntax issues.")
            continue

        is_valid, error_message = validate_contract(synthetic_contract)

        if is_valid:
            valid_contracts.append(synthetic_contract)
            logger.info(f"Contract {len(valid_contracts)} is valid.")
        else:
            error_type = categorize_error(error_message)
            error_feedback.append((error_type, synthetic_contract))
            logger.info(f"Contract is invalid. Error: {error_message}")

            if error_type == "syntax_noise":
                logger.info("Skipping fine-tuning due to noise (syntax).")
                continue

            if len(error_feedback) > 0:
                feedback_contracts = [entry[1] for entry in error_feedback if entry[0] == error_type]
                model = finetune_model(model, SolidityDataset(feedback_contracts, tokenizer))

    logger.info(f"Generated {len(valid_contracts)} valid contracts.")

if __name__ == "__main__":
    main()