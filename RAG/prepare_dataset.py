import json
import pandas as pd
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

def process_batch(batch):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    return [tokenizer.encode_plus(q, a, truncation=True, max_length=512, padding='max_length', return_tensors='pt') for q, a in batch]

def load_batches(file_path, batch_size=1000):
    with open(file_path, 'r') as file:
        batch, all_data = [], []
        for line in file:
            data = json.loads(line)
            batch.append((data['questionText'], data['answers'][0]['answerText']))
            if len(batch) == batch_size:
                all_data.extend(process_batch(batch))
                batch = []
        if batch:  # Process the last batch
            all_data.extend(process_batch(batch))
    return pd.DataFrame({'encodings': all_data})

def save_dataframe(df, file_path):
    df.to_pickle(file_path)

def split_dataset(df, test_size=0.1):
    return train_test_split(df, test_size=test_size)

if __name__ == '__main__':
    file_path = './dataset/test-qar_all.jsonl'
    df = load_batches(file_path, batch_size=300)
    # train_df, test_df = split_dataset(df)
    # save_dataframe(train_df, 'train_dataset.pkl')
    # save_dataframe(test_df, 'test_dataset.pkl')
    # Create a smaller subset (10% of the original data)

    subset_df = df.sample(frac=0.1, random_state=42)
    
    train_df, test_df = split_dataset(subset_df)
    
    # Save the resulting DataFrames to pickle files for later use in train.py
    save_dataframe(train_df, 'train_dataset.pkl')
    save_dataframe(test_df, 'test_dataset.pkl')