import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from model import load_model


#loading dataset
def load_dataset(path):
    """adding function to load dataset here"""
    pass

def format_instruction(sample):
	return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{sample['response']}

### Response:
{sample['instruction']}
"""

def finetune_llama2():
    #set up flash attention to speed up and reduce mem
    flash_attention = False
    #load model
    model_id = "NouseResearch/Llama-2-7b-hf"

    # setup QloRA 4-bit quantization with BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_use_double_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=flash_attention,
        device_map="auto"
    )
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #LoRA Configuration for k-bit training
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    #define training arguments
    args = TrainingArguments(
        output_dir="./llama2_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=6 if flash_attention else 4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True
    )

    #create and start the SFTTrainer
    dataset = load_dataset("training_set_path")
    max_seq_length =2048 #max sequence length for model
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        packing=True,
        formatting_func=format_instruction,
    )

    #Train 
    trainer.train()

    #save model
    trainer.save_model("./llama2_finetuned")

if __name__ == "__main__":
    finetune_llama2()