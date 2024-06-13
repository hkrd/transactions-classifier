import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, RootModel
from typing import Dict, List, Any
import pandas as pd
from classifier import get_classification
import logging
import logging.config
import sys


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": True,
        },
        "uvicorn.error": {
            "level": "INFO",
        },
        "uvicorn.access": {
            "level": "INFO",
        },
    },
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Define Pydantic models
class Transaction(BaseModel):
    date: str
    merchant: str
    amount: float

class GroupedTransactions(RootModel[Dict[str, List[Transaction]]]):
    pass


app = FastAPI()

# Pydantic model for the request body 
class TransactionsInput(BaseModel):
    transactions: List[Transaction]
    zero_shot_prompt: str

@app.post("/classify_transactions/", response_model=GroupedTransactions)
async def classify_transactions():
    try:
        # # Convert input transactions to a DataFrame
        # transactions_df = pd.DataFrame([t.dict() for t in input.transactions])

        # # Apply the classification function
        # transactions_df['Classification'] = transactions_df.apply(
        #     lambda x: classify_transaction(x, input.zero_shot_prompt), axis=1
        # )

        # # Group transactions by Classification
        # grouped_transactions = transactions_df.groupby('Classification').apply(lambda x: x.to_dict(orient='records')).to_dict()
        return GroupedTransactions.model_validate(get_classification())


    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
