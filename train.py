import torch
import os
import argparse
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def main(args):
    """
    Main function to run the fine-tuning process.
    """
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Load Dataset ---
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)
    training_dataset = dataset["train"].shuffle().select(range(args.num_train_samples))

    # --- Load Model and Tokenizer ---
    print("Loading base model and tokenizer...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Preprocess and Format Data ---
    def construct_datapoint(example):
        combined_text = example['input'] + example['output']
        tokenized_output = tokenizer(
            combined_text, max_length=512, truncation=True, padding="max_length", return_attention_mask=True
        )
        return {"input_ids": tokenized_output["input_ids"], "attention_mask": tokenized_output["attention_mask"]}

    print("Processing dataset...")
    processed_training_dataset = training_dataset.map(construct_datapoint, batched=False, remove_columns=['input', 'output'])

    # --- Configure PEFT (LoRA) ---
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # --- Set Up Trainer ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_arguments = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=processed_training_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=train_arguments,
    )

    model.config.use_cache = False

    # --- Start Training ---
    print("Starting model fine-tuning...")
    trainer.train()
    print("Fine-tuning complete!")
    
    # Save the final model adapter
    final_path = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(final_path)
    print(f"Final model adapter saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Llama model for Text-to-SQL.")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf", help="The base model to fine-tune.")
    parser.add_argument("--dataset_name", type=str, default="ChrisHayduk/Llama-2-SQL-Dataset", help="The dataset to use for training.")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to save model checkpoints.")
    parser.add_argument("--num_train_samples", type=int, default=5000, help="The number of training samples to use.")
    
    args = parser.parse_args()
    main(args)
