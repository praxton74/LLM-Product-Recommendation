from langchain.llms import BaseLLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class MyGPT2Model(BaseLLM):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    def __call__(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
