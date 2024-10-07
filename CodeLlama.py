import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments



# Load dataset
dataset = torch.load("path/to/dataset.pt")


# Hugging Face model id
model_id = "codellama/CodeLlama-7b-hf" # 

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' 

# # set chat template to OAI chatML
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config 
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

# Training arguments
args = TrainingArguments(
    output_dir="code-llama-7b-text-to-sql", 
    num_train_epochs=3,                  
    per_device_train_batch_size=3,         
    gradient_accumulation_steps=2,          
    gradient_checkpointing=True,           
    optim="adamw_torch_fused",             
    logging_steps=10,                       
    save_strategy="epoch",                 
    learning_rate=2e-4,                    
    bf16=True,                             
    tf32=True,                             
    max_grad_norm=0.3,                      
    warmup_ratio=0.03,                      
    lr_scheduler_type="constant",           
    push_to_hub=True,                      
    report_to="tensorboard",                
)


max_seq_length = 3072 # max sequence length 

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  
        "append_concat_token": False, 
    }
)

# train model
trainer.train()

# save model
trainer.save_model()
