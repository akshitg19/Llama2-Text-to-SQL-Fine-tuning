import torch
import uvicorn
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from contextlib import asynccontextmanager

# --- Global App and Model State ---
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model on startup and clear it on shutdown.
    """
    print("Application startup: Loading model...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        app_state["args"].model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    app_state["tokenizer"] = AutoTokenizer.from_pretrained(app_state["args"].model_name, trust_remote_code=True)
    app_state["tokenizer"].pad_token = app_state["tokenizer"].eos_token
    
    print(f"Loading LoRA adapters from: {app_state['args'].adapter_path}")
    app_state["model"] = PeftModel.from_pretrained(base_model, app_state["args"].adapter_path)
    app_state["model"].eval()
    
    print("Model loaded successfully.")
    yield
    
    print("Application shutdown: Cleaning up.")
    app_state.clear()

app = FastAPI(lifespan=lifespan)

class SQLQueryRequest(BaseModel):
    question: str

@app.post("/generate-sql")
async def generate_sql(request: SQLQueryRequest):
    """
    API endpoint to generate an SQL query from a question.
    """
    model = app_state["model"]
    tokenizer = app_state["tokenizer"]
    device = model.device

    prompt = f"### Instruction:\nGiven an input question, create a syntactically correct SQL query.\n\n### Input:\n{request.question}\n\n### Response:"
    encoded_input = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_tokens = model.generate(**encoded_input, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
    
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    response_tag = "### Response:"
    sql_query = generated_text.split(response_tag, 1)[1].strip() if response_tag in generated_text else generated_text.strip()
        
    return {"generated_sql": sql_query}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Text-to-SQL FastAPI server.")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf", help="The base model name.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapters.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on.")
    
    args = parser.parse_args()
    app_state["args"] = args
    
    uvicorn.run(app, host=args.host, port=args.port)
