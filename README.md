# transactions-classifier

API for classifying transactions by type using OpenAI Text Completion API

## Endpoints

- /classify_transactions/ - takes in a CSV file (use the one provided in /data as an example) and returns a JSON with transactions grouped by their type

## Transaction Types:

Groceries, Shopping, Building Improvement, Work, Utility Bills, Professional Services, Software/IT and Other

## How to run:

- Install all depedencies with `poetry install`

- insert your OpenAI API Key in `local.env`

- (optional) set limit to how many transactions get processed in `local.env`

- run with `python routes.py`

- access Swagger at `http://localhost:8000/docs`