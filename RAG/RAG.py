from transformers import RagTokenizer, RagRetriever, AutoModelForCausalLM, GPT2ForQuestionAnswering
import os
import argparse


def load_gpt_model(path):
    if path and os.path.exists(path):
        model = AutoModelForCausalLM.from_pretrained(path)    
    
    else:
        #load gpt2 pretrained model from Hugging Face
        model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str, help='path to model')
    args = parser.parse_args()
    #load generative model
    model = load_gpt_model(args.path)
    print("Model Loaded")

    #retriever
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
    #create RAG tokenizer
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

    #example
    querry = "What is langchain and explain it clearly with example."

    #retrieve relevant documents
    retrieved_docs = retriever.retrieve(querry)

    #generate answer 
    for doc in retrieved_docs:
        input_text = f"{querry} Context: {doc['text']}"
        input_ids = rag_tokenizer.encode(input_text, return_tensors="pt")
        response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = rag_tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print("Answer:", response)