import pandas as pd
import asyncio
import aiohttp
from datetime import datetime
from utils.utils import ingest_files, calculate_socratic_ratio
from utils.utils import read_yaml_config
from utils.metrics import calc_cosine_sim, calc_response_relevance, compute_socratic_score
from tenacity import retry, wait_random_exponential, stop_after_attempt

config = read_yaml_config('src/config/config.yaml')



def set_up_tests() -> pd.DataFrame:
    print('=======Entering test zone========')
    
    files_to_read = config['prompt_files']
    print(f"files inserted for testing are: {files_to_read}")
    question_set = ingest_files(files_to_read)
    return question_set


async def process_query_async(query: str, park: str, id, session):
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
    async def send_request():
        print(f"sending in request for query {query}")
        payload = {config['query_parameter']: query, 'selected_park': park}
        if config['additional_payload_params'] is not None:
            payload.update(config['additional_payload_params'])
        
        timeout = aiohttp.ClientTimeout(total=240)
        start_time = datetime.now()
        # Send the POST request using aiohttp
        async with session.post(config['endpoint'], json=payload, timeout= timeout) as response:
            # Extract the raw JSON response
            try:
                raw_json = await response.json()
                response_data = raw_json['response']
                for i, doc in enumerate(response_data.get('docs', []), start=1):
                    for key, value in doc.items():
                        response_data[f'doc_{i}_{key}'] = value
                
                for i, chunk in enumerate(response_data.get('chunks', []), start=1):
                    response_data[f'chunk_{i}'] = chunk['text']
                    response_data[f'chunk_{i}_score'] = chunk['score']
                
                # Remove the 'docs' key as its data is now flattened into the main dictionary
                response_data.pop('docs', None)
                response_data.pop('chunks')
                time_diff= (datetime.now() - start_time).total_seconds()
                response_data['latency'] = time_diff
                print(f"response data processed for query {query}")
                return response_data
            except:
                print(f"Invalid response to query {query} so skipping")
                return {}
    
    try:
        return await send_request()
    except:
        print(f"Question {query} skipped due to a timeout and retry error")
        return {}
    
        

async def run_tests_async(question_set: pd.DataFrame):
    print('=======Query zone========')
    flattened_responses = []
    queries = question_set[config['query_parameter']]
    if config['park_column'] is not None:
        parks = question_set[config['park_column']]  
    else: 
        parks = ['Orlando'] * len(queries)

    ids = range(len(queries))

    async with aiohttp.ClientSession() as session:
        # Split the list of tasks into chunks of 20
        chunk_size = 20
        for i in range(0, len(queries), chunk_size):
            chunk_queries = queries[i:i+chunk_size]
            chunk_parks = parks[i:i+chunk_size]
            chunk_ids = ids[i:i+chunk_size]
            print(f"=======Running group {(i/20) +1} for group size of {chunk_size}=======")
            # Create tasks for the current chunk
            tasks = [process_query_async(query, park, id, session) for query, park, id in zip(chunk_queries, chunk_parks, chunk_ids)]
            
            # Await the tasks for the current chunk
            results = await asyncio.gather(*tasks)
            flattened_responses.extend(results)

    print('=======Processing results========')
    response_df = pd.DataFrame(flattened_responses)
    
    response_df = response_df.merge(question_set, how='left', left_on='user_query', right_on=config['query_parameter'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config['cosine_compare_cols'] is not None:
        response_df = calc_cosine_sim(response_df=response_df, compare_col1=config['cosine_compare_cols'], compare_col2=config['response_column'])
    if config['calculate_llm_metrics']:
        response_df = calc_response_relevance(response_df=response_df, question_col=config['query_parameter'], response_col=config['response_column'])
    if config['socratic_criteria_column'] is not None:
        response_df['socratic_response']= response_df.apply(lambda row: compute_socratic_score(criterion= row[config['socratic_criteria_column']], generated_reponse= row[config['response_column']]), axis=1)
        response_df['socratic_ratio'] = response_df['socratic_response'].apply(lambda row: calculate_socratic_ratio([x.lower() for x in row]) if row is not None else None)

       
    print('=======Writing file ========')
    response_df.to_excel(f'./output/{timestamp}_watchtower_output.xlsx')



if __name__ == "__main__":
    question_set= set_up_tests()
    asyncio.run(run_tests_async(question_set=question_set))



