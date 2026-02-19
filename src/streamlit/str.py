import streamlit as st
import os
import yaml
from datetime import datetime
# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="RecProbe", page_icon="🛡️", layout="wide")
st.title("🛡️ RecProbe")

# --------------------------------------------------
# SIDEBAR — BASE CONFIG
# --------------------------------------------------
st.sidebar.header("📦 Base Configuration")

# Dataset
dataset_name = st.sidebar.text_input("Dataset name", "amazon_All_Beauty")
dataset_path = st.sidebar.text_input("Dataset path", "data/input/amazon_All_Beauty")

# Input files
st.sidebar.subheader("📥 Input Files")
reviews_file = st.sidebar.text_input("Reviews file name", "All_Beauty")
reviews_format = st.sidebar.selectbox("Reviews format", ["csv", "json"], index=1)
reviews_sep = st.sidebar.text_input("Reviews separator", ",")

items_file = st.sidebar.text_input("Items file name", "meta_All_Beauty")
items_format = st.sidebar.selectbox("Items format", ["csv", "json"], index=0)
items_sep = st.sidebar.text_input("Items separator", ",")

# Output files
st.sidebar.subheader("📤 Output Files")
output_reviews_format = st.sidebar.selectbox("Output reviews format", ["csv", "json"], index=0)
output_reviews_sep = st.sidebar.text_input("Output reviews separator", ",")
output_items_format = st.sidebar.selectbox("Output items format", ["csv", "json"], index=0)
output_items_sep = st.sidebar.text_input("Output items separator", ",")

# Noise and preprocessing
st.sidebar.subheader("⚙️ Preprocessing / Noise")
noise_profile = st.sidebar.selectbox("Noise profile", ["rating", "review", "combined"], index=0)
kcore = st.sidebar.number_input("k-core", min_value=0, value=20)
min_rating = st.sidebar.slider("Min rating", 1, 5, 1)
min_review_length = st.sidebar.number_input("Min review length", value=0)

# Split configuration
st.sidebar.subheader("🔀 Split Configuration")
train_ratio = st.sidebar.slider("Training ratio", 0.0, 1.0, 0.8)
validation_ratio = st.sidebar.slider("Validation ratio", 0.0, 1.0, 0.1)
test_ratio = st.sidebar.slider("Test ratio", 0.0, 1.0, 0.1)
noise_in_test = st.sidebar.checkbox("Inject noise in test set", False)
split_strategy = st.sidebar.selectbox(
    "Split strategy",
    ["random holdout", "temporal leave-one-out", "random leave-one-out"]
)

# Global settings
st.sidebar.subheader("🛠️ Global Settings")
drop_duplicates = st.sidebar.checkbox("Drop duplicates", True)
seed = st.sidebar.number_input("Random seed", value=42)
evaluation = st.sidebar.selectbox(
    "Evaluation",
    ["cornac", "recbole", "custom"])
verbose = st.sidebar.checkbox("Verbose logging", True)


budget = st.sidebar.number_input("Budget", 1, 1_000_000, 1000)
avoid_duplicates = st.sidebar.checkbox("Avoid duplicated interactions",True)

# --------------------------------------------------
# Shared Blocks
# --------------------------------------------------
def temporal_interval_block(default_start=1609459200, default_end=1640995200):
    col1, col2 = st.columns(2)
    with col1:
        start = st.number_input("Start timestamp", value=default_start)
    with col2:
        end = st.number_input("End timestamp", value=default_end)
    return {"start_timestamp": int(start), "end_timestamp": int(end)}

def rating_behavior_block(default="gaussian"):
    col1, col2, col3 = st.columns(3)
    with col1:
        #min_rating_val = st.slider("Min rating", 1, 100, 1)
        min_rating_val = st.number_input("Min rating", value=1)

    with col2:
        #max_rating_val = st.slider("Max rating", 1, 100, 5)
        max_rating_val = st.number_input("Max rating", value=5)

    with col3:
        sampling = st.selectbox("Sampling", ["gaussian", "uniform"],
                                index=0 if default == "gaussian" else 1)
    return {
        "min_rating": min_rating_val,
        "max_rating": max_rating_val,
        "sampling_strategy": sampling,
    }


# --------------------------------------------------
# CONTEXT BUILDERS
# --------------------------------------------------

def build_semantic_drift():
    st.header("🧠 Semantic Drift")
    operation = "corrupt"
    target = st.selectbox("Target", ["item", "user"])
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"])
    max_reviews = st.number_input("Max reviews per node", value=50)
    min_reviews = st.number_input("Min reviews per node", value=1)
    min_length = st.number_input("Min review length", value=10)
    model = st.text_input("Model", "Qwen/Qwen2.5-1.5B-Instruct")

    return {
        "operation": operation,
        "target": target,
        "selection_strategy": selection_strategy,
        "max_reviews_per_node": max_reviews,
        "min_reviews_per_node": min_reviews,
        "min_length_of_review": min_length,
        "model": model,
        "temporal_interval": temporal_interval_block(),
        "rating_behavior": rating_behavior_block(),
    }


def build_rating_review_burst():
    st.header("💥 Rating / Review Burst")
    operation = st.selectbox("Operation", ["add", "remove", "corrupt"], index=2)
    target = st.selectbox("Target", ["item", "user"])
    modify = st.selectbox("Modify", ["rating", "review"])
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"])
    max_reviews = st.number_input("Max reviews per node", value=500)
    min_reviews = st.number_input("Min reviews per node", value=1)
    min_length = st.number_input("Min review length", value=1)
    model = st.text_input("Model", "Qwen/Qwen2.5-1.5B-Instruct")

    return {
        "operation": operation,
        "target": target,
        "modify": modify,
        "selection_strategy": selection_strategy,
        "max_reviews_per_node": max_reviews,
        "min_reviews_per_node": min_reviews,
        "min_length_of_review": min_length,
        "model": model,
        "temporal_interval": temporal_interval_block(),
        "rating_behavior": rating_behavior_block(),
    }


def build_realistic_noise():
    st.header("🎯 Random inconsistencies")
    #operation = st.selectbox("Operation", ["add", "remove"], index=0)
    target = st.selectbox("Target", ["item", "user"])
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"])
    max_ratings = st.number_input("Max ratings per node", value=100)
    min_ratings = st.number_input("Min ratings per node", value=10)
    preserve_degree = st.checkbox("Preserve degree distribution", True)
    if profile == "hybrid":
        operation = "corrupt"
        #st.write("Profile is hybrid → operation forced to 'corrupt'")
        flip = st.selectbox("Flip", ["rating", "Review"], index=0)
        return {
            "operation": operation,
            "target": target,
            "flip": flip,
            "selection_strategy": selection_strategy,
            "max_ratings_per_node": max_ratings,
            "min_ratings_per_node": min_ratings,
            "preserve_degree_distribution": preserve_degree,
            "rating_behavior": rating_behavior_block(),
            "temporal_interval": temporal_interval_block(),
        }
    else:
        operation = st.selectbox("Operation", ["add", "remove"], index=0)
        return {
            "operation": operation,
            "target": target,
            "selection_strategy": selection_strategy,
            "max_ratings_per_node": max_ratings,
            "min_ratings_per_node": min_ratings,
            "preserve_degree_distribution": preserve_degree,
            "rating_behavior": rating_behavior_block(),
            "temporal_interval": temporal_interval_block(),
        }




def build_realistic_noise_hybrid():
    st.header("🎯 Random inconsistencies")
    operation = "corrupt"
    target = st.selectbox("Target", ["item", "user"])
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"])
    max_ratings = st.number_input("Max ratings per node", value=100)
    min_ratings = st.number_input("Min ratings per node", value=10)
    preserve_degree = st.checkbox("Preserve degree distribution", True)

    return {
        "operation": operation,
        "target": target,
        "selection_strategy": selection_strategy,
        "max_ratings_per_node": max_ratings,
        "min_ratings_per_node": min_ratings,
        "preserve_degree_distribution": preserve_degree,
        "rating_behavior": rating_behavior_block(),
        "temporal_interval": temporal_interval_block(),
    }

def build_rating_burst_noise():
    st.header("⭐ Rating Burst")
    operation = "add"
    target = st.selectbox("Target", ["item", "user"], index=0)
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"], index=1)
    max_ratings = st.number_input("Max ratings per node", value=300)
    min_ratings = st.number_input("Min ratings per node", value=100)

    return {
        "operation": operation,
        "target": target,
        "selection_strategy": selection_strategy,
        "max_ratings_per_node": max_ratings,
        "min_ratings_per_node": min_ratings,
        "rating_behavior": rating_behavior_block(default="gaussian"),
        "temporal_interval": temporal_interval_block(default_start=1009459200, default_end=1798761600),
    }


def build_user_burst_noise():
    st.header("👤 User Burst Noise")
    operation = "add"
    target = "user"
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"], index=0)
    max_ratings = st.number_input("Max ratings per user", value=40)
    min_ratings = st.number_input("Min ratings per user", value=20)

    return {
        "operation": operation,
        "target": target,
        "selection_strategy": selection_strategy,
        "max_ratings_per_node": max_ratings,
        "min_ratings_per_node": min_ratings,
        "rating_behavior": rating_behavior_block(),
        "temporal_interval": temporal_interval_block(),
    }


def build_timestamp_corruption():
    st.header("⏱️ Timestamp Corruption")
    target = st.selectbox("Target", ["user", "item"])
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"])
    max_ratings = st.number_input("Max ratings per node", value=40)
    min_ratings = st.number_input("Min ratings per node", value=20)

    st.subheader("Temporal behavior")
    corruption_mode = st.selectbox("Corruption mode", ["uniform", "forward", "backward"])
    intensity = st.selectbox("Intensity", ["low", "medium", "high"])

    st.subheader("Rating behavior")
    rating = rating_behavior_block(default="uniform")

    return {
        "target": target,
        "selection_strategy": selection_strategy,
        "max_ratings_per_node": max_ratings,
        "min_ratings_per_node": min_ratings,
        "temporal_behavior": {
            "corruption_mode": corruption_mode,
            "intensity": intensity,
        },
        "rating_behavior": rating,
    }


# --------------------------------------------------
# REVIEW BURST NOISE BUILDERS
# --------------------------------------------------

def build_remove_reviews():
    st.header("🗑️ Remove Reviews")
    operation = "remove"
    target = st.selectbox("Target", ["item", "user"], index=0)
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"], index=0)
    max_reviews = st.number_input("Max reviews per node", value=50)
    min_reviews = st.number_input("Min reviews per node", value=1)
    min_length = st.number_input("Min length of review (tokens)", value=20)

    return {
        "operation": operation,
        "target": target,
        "selection_strategy": selection_strategy,
        "max_reviews_per_node": max_reviews,
        "min_reviews_per_node": min_reviews,
        "min_length_of_review": min_length,
        "temporal_interval": temporal_interval_block(default_start=1609459200, default_end=1640995200),
        "rating_behavior": rating_behavior_block(default="gaussian"),
    }


def build_review_burst_noise():
    st.header("💥 Review Burst Noise")
    operation = "add"
    target = st.selectbox("Target", ["item", "user"], index=1)
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"], index=1)
    max_reviews = st.number_input("Max reviews per node", value=300)
    min_reviews = st.number_input("Min reviews per node", value=100)
    min_length = st.number_input("Min length of review (tokens)", value=20)

    st.subheader("Near duplicates configuration")
    model = st.text_input("Paraphrasing model", "ramsrigouthamg/t5_paraphraser")
    review_text = st.text_area("Review text for near duplicates",
                               "This product is absolutely one of my favourites! It lasts all day and the quality remains great from morning to night, which is something I really value when I use a product like this. I have used it many times over a long period of time, and every time the experience has been very good, very reliable, and very satisfying. The performance is excellent, the quality is excellent, and overall it feels like a great choice for everyday use, even after many hours, even after repeated use, and even when expectations are high. I would definitely recommend it because the product works well, works consistently, and works exactly as expected.")
    title = st.text_input("Review title", "The product you cannot live without")
    #rating = st.slider("Rating for near duplicate reviews", 1, 100, 1)
    rating = st.slider("Rating for near duplicate reviews", value=5)

    return {
        "operation": operation,
        "target": target,
        "selection_strategy": selection_strategy,
        "max_reviews_per_node": max_reviews,
        "min_reviews_per_node": min_reviews,
        "min_length_of_review": min_length,
        "temporal_interval": temporal_interval_block(default_start=1609459200, default_end=1640995200),
        "near_duplicates_configuration": {
            "model": model,
            "review": review_text,
            "title": title,
            "rating": rating,
        },
        "rating_behavior": rating_behavior_block(default="gaussian"),
    }


def build_sentence_noise():
    st.header("📝 Sentence Noise / Text Shuffle")
    operation = "corrupt"
    target = st.selectbox("Target", ["item", "user"], index=0)
    intensity = st.selectbox("Intensity", ["low", "medium", "high"], index=0)
    selection_strategy = st.selectbox("Selection strategy", ["uniform", "top", "least"], index=0)
    max_reviews = st.number_input("Max reviews per node", value=50)
    min_reviews = st.number_input("Min reviews per node", value=1)
    min_length = st.number_input("Min length of review (tokens)", value=10)
    noise_type = st.selectbox("Noise type", ["shuffle", "token_noise", "sentence_noise"], index=1)
    model = st.text_input("Model for text corruption", "Qwen/Qwen2.5-1.5B-Instruct")

    return {
        "operation": operation,
        "target": target,
        "intensity": intensity,
        "selection_strategy": selection_strategy,
        "max_reviews_per_node": max_reviews,
        "min_reviews_per_node": min_reviews,
        "min_length_of_review": min_length,
        "noise_type": noise_type,
        "model": model,
        "temporal_interval": temporal_interval_block(default_start=1609459200, default_end=1640995200),
        "rating_behavior": rating_behavior_block(default="gaussian"),
    }

# --------------------------------------------------
# Profile Selection
# --------------------------------------------------
profile = st.sidebar.selectbox("Profile", ["rating", "review", "hybrid"], index=0)

profile_context_map = {
    "rating": ["random_inconsistencies", "rating_burst", "timestamp_corruption"],
    "review": ["random_inconsistencies", "review_burst", "sentence_noise"],
    "hybrid": ["random_inconsistencies", "hybrid_burst", "semantic_drift"],
}

available_contexts = profile_context_map[profile]
context = st.sidebar.selectbox("Context", available_contexts)

# --------------------------------------------------
# Dispatcher
# --------------------------------------------------
builder_map = {
    "semantic_drift": build_semantic_drift,
    "hybrid_burst": build_rating_review_burst,
    "random_inconsistencies": build_realistic_noise,
    "rating_burst": build_rating_burst_noise,
    "timestamp_corruption": build_timestamp_corruption,
    "remove_reviews": build_remove_reviews,
    "review_burst": build_review_burst_noise,
    "sentence_noise": build_sentence_noise,
}
context_block = builder_map[context]()

# --------------------------------------------------
# FINAL CONFIG
# --------------------------------------------------
config = {
    "input": {
        "reviews": {"file_name": reviews_file, "format": reviews_format, "separator": reviews_sep},
        "items": {"file_name": items_file, "format": items_format, "separator": items_sep},
    },
    "output": {
        "reviews": {"format": output_reviews_format, "separator": output_reviews_sep},
        "items": {"format": output_items_format, "separator": output_items_sep},
    },
    "noise_profile": noise_profile,
    "kcore": kcore,
    "min_rating": min_rating,
    "min_review_length": min_review_length,
    "split": {
        "training": train_ratio,
        "validation": validation_ratio,
        "test": test_ratio,
        "noise_in_test": noise_in_test,
        "strategy": split_strategy,
    },
    "drop_duplicates": drop_duplicates,
    "dataset": dataset_name,
    "random_seed": seed,
    "evaluation": evaluation,
    "verbose": verbose,
    "context": context,
    "budget": budget,
    "avoid_duplicates":avoid_duplicates,
    context: context_block,
}

st.subheader("📦 Configuration Preview")
st.json(config)

# --------------------------------------------------
# SAVE FUNCTION
# --------------------------------------------------
def save_config(config):
    folder = os.path.join("generated_configs", '')
    os.makedirs(folder, exist_ok=True)

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(folder, f"config.yaml")

    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False)


    return filepath

if st.button("🛡️ Save Configuration", use_container_width=True):
    path = save_config(config)
    st.success("Configuration saved!")
    st.info(path)
