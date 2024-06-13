import io
from fastapi import FastAPI, File, HTTPException, UploadFile
import pandas as pd
from src.classifier import Classifier
import logging
import logging.config

from src.models import GroupedTransactions, Transaction


app = FastAPI()


@app.post("/classify_transactions/", response_model=GroupedTransactions)
async def classify_transactions(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Only CSV files are allowed."
        )

    try:
        contents = await file.read()
        csv_string = contents.decode("utf-8")

        Transaction.validate_csv(csv_string)

        csv_io = io.StringIO(csv_string)
        df = pd.read_csv(csv_io)
        classifier = Classifier()

        return GroupedTransactions.model_validate(classifier.get_classification(df))

    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
