# Evaluation & Test Suite
Framework to conduct testing and evaluation of UDX TM Virtual Assistant.



## Quick Start

The following setup items are needed to start this server and send it requests to work as intended.
This assumes you already have the backend stood up and ready to recieve requests. If you are unsure of how to do so please checkout the read me
at https://github.com/Bain/aag-j6qx-base. 

Create .env files from .env.example

From the root directory:

```bash
docker compose up --build 
```

The above command is only needed the first time you run the script. For subsequent runs you can use:
```bash
docker compose up 
```
## Configuration

# Adding questions to be tested

Within the promt_files folder you can add any amount of excels containing test questions. The excels MUST contain
a column named 'query'. This column will be used to send questions to the endpoint. Files that do not contain the header
'query' will be skipped and not included in the test run. To enable a file to be added to the test set simply add the file name in the prompt_files list in the config.yaml

# Adjusting other configuration settings

Below you can find the available configurations and their definitions:
- endpoint: This is the api endpoint you want to query or ping. If you are running this locally and have your backend docker spun up you can use the preamble: "http://host.docker.internal:8080/<\endpoint to hit>"

- prompt_files: These are the files to be run during your test. You can add any number of excel files into here and they will be tested as long as they fit our guidelines. Files are expected to live in the prompt_files folder. Files are added as a dictionary item in the config appended with a ":" and then have one entry for the sheet name. If left blank watch assumes the sheet name is "Sheet1"

- query_parameter: Parameter for sending in a question to the chatbot

- cosine_compare_cols: column to run a cosine similarity with the response column. If left blank no column is run and step is skipped

- calculate_llm_metrics: true false entry, if enabled the metrics for relevance and completeness will run

- response_column: column that represents the generated answers from chat

- park_column: park designation for query, if left blank defaults all queries to orlando

- socratic_criteria_column: column to use for socratic evaluation of your responses. This column should contain your socratic questions. If left blank this step will be skipped

- additional_payload_params: Additional parameters to be added to payload when pinging output

# Output

Files will be output to the output folder and timestamped. No output files will be uploaded to github. 

