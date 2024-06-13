import csv
import io
from typing import Dict, List
from fastapi import HTTPException
from pydantic import BaseModel, RootModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Transaction(BaseModel):
    date: str
    merchant: str
    amount: float

    @classmethod
    def map_row(cls, row):
        return {
            "date": row["Date"],
            "merchant": row["Supplier"],
            "amount": float(row["Transaction value"].replace(",", "")),
        }

    @classmethod
    def validate_csv(cls, csv_str: str):
        try:
            csv_content = io.StringIO(csv_str)

            reader = csv.DictReader(csv_content)
            for row in reader:
                mapped_row = cls.map_row(row)
                Transaction(**mapped_row)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"CSV validation error: {e}")


class GroupedTransactions(RootModel[Dict[str, List[Transaction]]]):
    pass


class Settings(BaseSettings):
    openai_api_key: str
    completions_model: str
    transaction_limit: int

    model_config = SettingsConfigDict(env_file="local.env", env_file_encoding="utf-8")
