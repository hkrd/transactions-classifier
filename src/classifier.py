from collections import defaultdict
from typing import Any, Dict
import openai
import pandas as pd

from src.models import Settings


ZERO_SHOT_PROMPT = """
You are analysing all provided transactions and classifying them into one of five categories.
The five categories are Groceries, Shopping, Building Improvement, Work, Utility Bills, Professional Services and Software/IT.
Not all categories have to be used, only if they match. If you can't tell what it is, say Other

Transaction:

Supplier: SUPPLIER_NAME
Description: DESCRIPTION_TEXT
Value: TRANSACTION_VALUE

The classification is:"""


class Classifier:

    def __init__(self) -> None:
        self.settings = Settings()
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)

    def request_completion(self, prompt: str) -> Any:

        completion_response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a data expert specialising in analyzing transactions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=self.settings.completions_model,
        )

        return completion_response

    def classify_transaction(self, transaction: pd.DataFrame, prompt: str) -> Any:

        prompt = prompt.replace("SUPPLIER_NAME", transaction["Supplier"])
        prompt = prompt.replace("DESCRIPTION_TEXT", transaction["Description"])
        prompt = prompt.replace(
            "TRANSACTION_VALUE", str(transaction["Transaction value"])
        )

        classification = (
            self.request_completion(prompt).choices[0].message.content.replace("\n", "")
        )

        return classification

    def get_classification(self, transactions: pd.DataFrame) -> Dict:

        transactions["Transaction value"] = (
            transactions["Transaction value"].str.replace(",", "").astype(float)
        )

        test_transactions = transactions.iloc[: self.settings.transaction_limit].copy()

        classification_results = test_transactions.apply(
            lambda x: self.classify_transaction(x, ZERO_SHOT_PROMPT), axis=1
        )

        test_transactions.loc[:, "Classification"] = classification_results

        grouped_transactions = test_transactions.groupby("Classification")
        data = defaultdict(list)
        for classification, group in grouped_transactions:
            records = group.to_dict(orient="records")
            for record in records:
                data[str(classification)].append(
                    {
                        "date": record["Date"],
                        "merchant": record["Supplier"],
                        "amount": float(record["Transaction value"]),
                    }
                )

        result_data = dict(data)

        return result_data
