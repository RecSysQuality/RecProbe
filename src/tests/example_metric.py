"""Example of a custom metric plugged in through evaluation.yaml."""
def target_share(ranked, test, train, k):
    """Share of top-k slots occupied by the 10 most popular training items."""
    top = set(train["item_id"].value_counts().index[:10])
    r = ranked[ranked["rank"] <= k]
    return float(r["item_id"].isin(top).mean()) if len(r) else 0.0
