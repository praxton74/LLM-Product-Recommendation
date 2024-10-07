from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

# Path to fine-tuned model directory
MODEL_DIR = "./model/GPT2_model"

# Load pre-trained model tokenizer 
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

# Load fine-tuned model (weights)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

# switch model to evaluation mode
model.eval()  

class TextGenerationRequest(BaseModel):
    text: str
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    num_return_sequences: int = 1

@app.post("/generate-text/")
async def generate_text(request: TextGenerationRequest):
    # Encode the text input from the request
    input_ids = tokenizer.encode(request.text, return_tensors='pt')
    
    # Generate the text
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=request.max_length,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_sample=True,
        num_return_sequences=request.num_return_sequences
    )

    # Decode the generated sequence
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]
    return {"generated_texts": generated_texts}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
