from collections import defaultdict
import openai
import pandas as pd


COMPLETIONS_MODEL = "gpt-4o"

client = openai.OpenAI(api_key="<your_key>")

transactions = pd.read_csv('./data/25000_spend_dataset_current.csv', encoding='unicode_escape')
transactions['Transaction value'] = transactions['Transaction value'].str.replace(',', '').astype(float)


def request_completion(prompt):

    completion_response = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a data expert specialising in analyzing transactions."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0,
                            max_tokens=5,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            model=COMPLETIONS_MODEL)

    return completion_response

def classify_transaction(transaction,prompt):

    prompt = prompt.replace('SUPPLIER_NAME',transaction['Supplier'])
    prompt = prompt.replace('DESCRIPTION_TEXT',transaction['Description'])
    prompt = prompt.replace('TRANSACTION_VALUE',str(transaction['Transaction value']))

    classification = request_completion(prompt).choices[0].message.content.replace('\n','')

    return classification


def get_classification():

    zero_shot_prompt = '''
    You are analysing all provided transactions and classifying them into one of five categories.
    The five categories are Groceries, Shopping, Building Improvement, Work, Utility Bills, Professional Services and Software/IT.
    Not all categories have to be used, only if they match. If you can't tell what it is, say Other

    Transaction:

    Supplier: SUPPLIER_NAME
    Description: DESCRIPTION_TEXT
    Value: TRANSACTION_VALUE

    The classification is:'''

    # Select the first 25 rows from the transactions DataFrame
    test_transactions = transactions.iloc[:10].copy()

    # Apply the classification function and store the results in a new Series
    classification_results = test_transactions.apply(lambda x: classify_transaction(x, zero_shot_prompt), axis=1)

    # Assign the results back to the 'Classification' column using .loc
    test_transactions.loc[:, 'Classification'] = classification_results

    # Group the transactions by the 'Classification' column
    grouped_transactions = test_transactions.groupby('Classification')
    data = defaultdict(list)
    for classification, group in grouped_transactions:
        records = group.to_dict(orient='records')
        for record in records:
            data[str(classification)].append({
                "date": record["Date"],
                "merchant": record["Supplier"],
                "amount": float(record["Transaction value"])
                                
            })

    result_data = dict(data)

    return result_data     
