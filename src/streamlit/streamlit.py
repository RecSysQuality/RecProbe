import streamlit as st
from datetime import datetime

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="RecProbe",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# Custom CSS for a cleaner look
# --------------------------------------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("## 🛡️")
with col2:
    st.markdown('<div class="title">RecProbe</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Semantic Drift Simulation & Review Protection Interface</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# Sidebar – Global settings
# --------------------------------------------------
st.sidebar.header("⚙️ Global Settings")

context = st.sidebar.selectbox(
    "Context",
    options=["semantic_drift"],
)

budget = st.sidebar.number_input(
    "Budget (total reviews to modify)",
    min_value=1,
    max_value=100000,
    value=5000,
    step=100,
)

avoid_duplicates = st.sidebar.checkbox(
    "Avoid duplicate (user–item) reviews",
    value=True,
)

# --------------------------------------------------
# Main content
# --------------------------------------------------
st.markdown("### 🧠 Semantic Drift Configuration")

# --- Operation & Target ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Operation & Target")

    col1, col2, col3 = st.columns(3)

    with col1:
        operation = st.selectbox(
            "Operation",
            options=["add", "remove", "corrupt"],
            index=2,
        )

    with col2:
        target = st.selectbox(
            "Target",
            options=["item", "user"],
        )

    with col3:
        selection_strategy = st.selectbox(
            "Selection Strategy",
            options=["uniform", "top", "least"],
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --- Review Constraints ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Review Constraints")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_reviews_per_node = st.number_input(
            "Min reviews per node",
            min_value=1,
            value=1,
        )

    with col2:
        max_reviews_per_node = st.number_input(
            "Max reviews per node",
            min_value=1,
            value=50,
        )

    with col3:
        min_length_of_review = st.number_input(
            "Min review length (tokens)",
            min_value=1,
            value=10,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --- Model ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Language Model")

    model = st.text_input(
        "Model used for semantic corruption",
        value="Qwen/Qwen2.5-1.5B-Instruct",
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --- Temporal Interval ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Temporal Interval")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start date",
            value=datetime.fromtimestamp(1609459200),
        )

    with col2:
        end_date = st.date_input(
            "End date",
            value=datetime.fromtimestamp(1640995200),
        )

    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())

    st.markdown("</div>", unsafe_allow_html=True)

# --- Rating Behavior ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Rating Behavior")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_rating = st.slider(
            "Min rating",
            min_value=1,
            max_value=5,
            value=1,
        )

    with col2:
        max_rating = st.slider(
            "Max rating",
            min_value=1,
            max_value=5,
            value=5,
        )

    with col3:
        rating_sampling_strategy = st.selectbox(
            "Sampling strategy",
            options=["gaussian", "uniform"],
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Final configuration preview
# --------------------------------------------------
st.markdown("### 📦 Configuration Preview")

config = {
    "context": context,
    "budget": budget,
    "avoid_duplicates": avoid_duplicates,
    "semantic_drift": {
        "operation": operation,
        "target": target,
        "selection_strategy": selection_strategy,
        "max_reviews_per_node": max_reviews_per_node,
        "min_reviews_per_node": min_reviews_per_node,
        "min_length_of_review": min_length_of_review,
        "model": model,
        "temporal_interval": {
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
        },
        "rating_behavior": {
            "min_rating": min_rating,
            "max_rating": max_rating,
            "sampling_strategy": rating_sampling_strategy,
        },
    },
}

st.json(config)

# --------------------------------------------------
# Action button
# --------------------------------------------------
st.markdown("---")
if st.button("🛡️ Run RecProbe Simulation", use_container_width=True):
    st.success("RecProbe configuration submitted successfully!")
