def compute_coverage(recommendations, all_items):
    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update(recs)
    return len(recommended_items) / len(all_items)


def compute_novelty(recommendations, interaction_counts, all_items):
    import numpy as np
    total_interactions = sum(interaction_counts[i] for i in all_items)
    user_novelties = []
    for user, recs in recommendations.items():
        if not recs:
            continue
        item_novelties = [-np.log2((interaction_counts.get(item,0)+1)/total_interactions) for item in recs]
        user_novelties.append(np.mean(item_novelties))

    return np.mean(user_novelties)/ np.log2(total_interactions) if user_novelties else 0.0
