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
            "parent_asin": "item_id",
            "rating": "rating",
            "text": "review_text",
            "title": "title",
            "timestamp": "timestamp"
        },
        "yelp": {
            "user_id": "user_id",
            "business_id": "item_id",
            "stars": "rating",
            "text": "review_text",
            "date": "timestamp"
        }
        # add here custom mapping according to your datasets as defined in the config.yaml
    }

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        if 'amazon' in dataset:
            dataset = 'amazon'
        elif 'yelp' in dataset:
            dataset = 'yelp'
        mapping = cls.dataset_mappings.get(dataset)
        if mapping is None:
            raise ValueError(f"No mapping found for dataset {dataset}")

        # Seleziona e rinomina le colonne
        df = df[list(filter(None, mapping.keys()))]  # ignora None
        rename_map = {k: v for k, v in mapping.items()}
        df = df.rename(columns=rename_map)
        if 'title' not in df:
            df['title'] = ''
        return df