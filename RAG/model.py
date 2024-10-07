from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForQuestionAnswering, BertTokenizer


def load_model(model_name):
    if model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        return model, tokenizer
    elif model_name =='bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        return model, tokenizer
    else:
        raise ValueError('model_name should be gpt2 or bert')
    