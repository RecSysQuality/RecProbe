from pydantic import BaseModel
from typing import Optional, ClassVar, Dict
import pandas as pd

class ReviewModel(BaseModel):
    # Standard fields
    user_id: str
    item_id: str
    rating: float
    review_text: str
    title: Optional[str] = None
    timestamp: Optional[str] = None

    # Mapping per dataset
    dataset_mappings: ClassVar[Dict[str, Dict[str, Optional[str]]]] = {
        "amazon": {
            "user_id": "user_id",
            "item_id": "parent_asin",
            "rating": "rating",
            "review_text": "text",
            "title": "title",
            "timestamp": "timestamp"
        },
        "yelp": {
            "user_id": "user_id",
            "item_id": "business_id",
            "rating": "stars",
            "review_text": "text",
            "title": None,   # Yelp non ha titolo
            "timestamp": "date"
        }
        # add here custom mapping according to your datasets as defined in the config.yaml
    }

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        mapping = cls.dataset_mappings.get(dataset)
        if mapping is None:
            raise ValueError(f"No mapping found for dataset {dataset}")

        # Seleziona e rinomina le colonne
        df = df[list(filter(None, mapping.values()))]  # ignora None
        rename_map = {v: k for k, v in mapping.items() if v is not None}
        df = df.rename(columns=rename_map)
        return df