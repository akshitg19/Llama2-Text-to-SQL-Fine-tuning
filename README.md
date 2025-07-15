# LLM Fine-tuning for Text-to-SQL Generation

This project demonstrates a complete pipeline for fine-tuning the Llama-2 (7B) model for text-to-SQL generation. Using a dataset of 5,000 SQL examples, the model was trained to translate natural language questions into precise SQL queries, achieving **30% exact match accuracy** on a held-out evaluation set.

The core of this project is the implementation of memory-efficient training techniques, including Parameter-Efficient Fine-Tuning (PEFT) with LoRA and 4-bit quantization, which enabled the fine-tuning process to be completed on a single GPU with just 16GB of VRAM.

## Key Features

-   **Fine-Tuning**: A script to fine-tune the `NousResearch/Llama-2-7b-hf` model on a specialized SQL dataset.
-   **Memory-Efficient Training**: Implements LoRA, 4-bit quantization, and gradient accumulation to drastically reduce memory footprint.
-   **Evaluation**: A script to measure the exact-match accuracy of the fine-tuned model.
-   **API Server**: A RESTful API built with FastAPI to serve the trained model for real-world applications.

## Project Structure

```
.
├── train.py              # Script for fine-tuning the model
├── evaluate.py           # Script for evaluating the model's performance
├── api.py                # FastAPI script to serve the model
├── requirements.txt      # Required Python packages
└── README.md             # This file
```

## Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <repository-name>
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
Install all required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

Run the `train.py` script to start fine-tuning. You must provide a directory to save the model checkpoints. This process will create a folder containing the trained LoRA adapters.
```bash
python train.py --output_dir ./sql-llama-finetuned
```

### 2. Evaluating the Model

Run the `evaluate.py` script to test the model's accuracy. You must point it to the directory containing your saved adapters.
```bash
python evaluate.py --adapter_path ./sql-llama-finetuned/final_checkpoint
```

### 3. Serving the Model via API

Run the `api.py` script to start the web server. You must provide the path to your trained adapters. The API will be available at `http://127.0.0.1:8000`.
```bash
python api.py --adapter_path ./sql-llama-finetuned/final_checkpoint
```

#### Example API Request

You can test the running API with `curl`:
```bash
curl -X POST "[http://127.0.0.1:8000/generate-sql](http://127.0.0.1:8000/generate-sql)" \
-H "Content-Type: application/json" \
-d '{
  "question": "What are the names of all the departments?"
}'
```

The expected response will be:
```json
{
  "generated_sql": "SELECT name FROM department"
}
