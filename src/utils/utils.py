import yaml 
import openai
import numpy as np
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')


def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_socratic_ratio(row):
    return row.count('yes') / len(row)

def fetch_embeddings(df, text_column):
    # Create a list to store embeddings
    embeddings = []

    # Iterate through each text in the specified column
    for text in df[text_column]:
        try:
            # Fetch embedding from OpenAI
            response = openai.Embedding.create(
                engine="text-embedding-3-small",
                input=text,
            )
            
            embedding = np.array(response['data'][0]['embedding'])
            embeddings.append(embedding)
            
        except Exception as e:
            print(f"Error fetching embedding for text: {text}. Error: {e}")
            # If an error occurs, append zeros to indicate failure
            embeddings.append(np.zeros(1536))  # Assuming GPT-3 embeddings of size 768

    # Convert the list of embeddings to a Pandas Series
    embedding_series = pd.Series(embeddings, name=f'{text_column}_embeddings')
    print(f"Embeddings created for {text_column}")
    return embedding_series

def ingest_files(files: list) -> pd.DataFrame:
    print('=======Reading files========')
    filtered_files=[]

    for file in files:
        sheet_name= 'Sheet1'
        if files[file]['sheet_name'] is not None:
            sheet_name = files[file]['sheet_name'] 
        print(os.getcwd())
        df= pd.read_excel(os.path.join('./prompt_files', file, ), sheet_name= sheet_name)
        print(df)
        if 'query' not in df.columns:
            print(f'''file {file} does not contain the column "query" so removing from test set''')
            continue
        print(f"file {file} added to test set list")
        
        df['file_source'] = file
        filtered_files.append(df)
    return pd.concat(filtered_files)