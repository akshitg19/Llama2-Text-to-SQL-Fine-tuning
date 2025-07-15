import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main(args):
    """
    Main function to evaluate the fine-tuned model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Dataset ---
    print("Loading evaluation dataset...")
    dataset = load_dataset(args.dataset_name)
    evaluation_dataset = dataset['eval'].shuffle()

    # --- Load Model and Tokenizer ---
    print("Loading base model and tokenizer...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load Fine-Tuned Adapters ---
    print(f"Loading fine-tuned adapters from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.eval()
    print("Fine-tuned model loaded successfully.")

    # --- Run Evaluation ---
    total_examples = min(args.num_eval_samples, len(evaluation_dataset))
    correct_matches = 0
    
    print(f"Starting evaluation on {total_examples} examples...")

    for i in range(total_examples):
        sample = evaluation_dataset[i]
        prompt = sample['input']
        correct_sql = sample['output'].strip()

        encoded_input = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_tokens = model.generate(**encoded_input, max_new_tokens=256, do_sample=False)
        
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        response_tag = "### Response:"
        if response_tag in generated_text:
            generated_sql = generated_text.split(response_tag, 1)[1].strip()
        else:
            generated_sql = generated_text.strip()

        if generated_sql == correct_sql:
            correct_matches += 1

    accuracy = (correct_matches / total_examples) * 100
    print("\n--- Evaluation Complete ---")
    print(f"Total examples processed: {total_examples}")
    print(f"Correct exact matches: {correct_matches}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Text-to-SQL model.")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf", help="The base model.")
    parser.add_argument("--dataset_name", type=str, default="ChrisHayduk/Llama-2-SQL-Dataset", help="The dataset for evaluation.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapters.")
    parser.add_argument("--num_eval_samples", type=int, default=500, help="Number of samples to evaluate.")
    
    args = parser.parse_args()
    main(args)
