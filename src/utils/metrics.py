from src.utils.utils import fetch_embeddings
from uptrain.operators import CosineSimilarity
from uptrain import EvalLLM, Evals
import pandas as pd
import numpy as np
import polars as pl
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import json
from utils.socratic_helpers import parse_socratic_parsing, socratic_prompt
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')


def calc_cosine_sim(response_df: pd.DataFrame, compare_col1: str, compare_col2: str) -> pd.DataFrame:
    print("'=======Creating Embeddings========'")
    for text_col in [compare_col1, compare_col2]:
        col_name = text_col + "_vector"
        response_df[col_name] = fetch_embeddings(response_df, text_col)
 
    cosine_pl= pl.DataFrame(response_df[[(compare_col1+ "_vector"),(compare_col2 + "_vector")]])
    similarity_op = CosineSimilarity(col_in_vector_1=(compare_col1 + "_vector"), col_in_vector_2=(compare_col2 + "_vector"))

    result = similarity_op.run(cosine_pl)

    result_df = result['output'].to_pandas()
    result_df.rename(columns={'cosine_similarity' : compare_col1 + "_" + compare_col2 + "_cosine_sim"}, inplace=True)
    
    return pd.concat([response_df, result_df], axis=1)


def calc_response_relevance(response_df: pd.DataFrame, question_col: str, response_col: str) -> pd.DataFrame:
    eval_llm = EvalLLM(openai_api_key=openai_api_key)
    prompt_data = [{'question': q, 'response': r} for q, r in zip(response_df[question_col], response_df[response_col])]
    
    res = eval_llm.evaluate(
    data = prompt_data,
    checks = [Evals.RESPONSE_RELEVANCE,
            Evals.RESPONSE_COMPLETENESS]
    )
    res_df = pd.DataFrame(res)
    res_df.drop(columns= ['question','response'], inplace=True)
    return pd.concat([response_df, pd.DataFrame(res)], axis=1)


def compute_socratic_score(criterion: str, generated_reponse: str, ):
    if pd.isna(criterion):
        print("no criterion found so skipping")
        return None
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model= 'gpt-3.5-turbo-0125')
    human_message_prompt = HumanMessagePromptTemplate.from_template(socratic_prompt)
    chat_prompt = ChatPromptTemplate.from_messages(
    [human_message_prompt]
)

    # get a chat completion from the formatted messages
    llm_response= chat(
        chat_prompt.format_prompt(
        generated_answer= generated_reponse, criterion= criterion
        ).to_messages()
    )
    parsed_response= parse_socratic_parsing(llm_response.content)
    output_parsed = parsed_response.split("<output>")[-1].split("</output>")
    response_dict = json.loads(output_parsed[0])
    socratic_results = response_dict['socratic_results']
    return socratic_results