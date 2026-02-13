# RecProbe: Stress-Testing Recommender Systems via Systematic Noise Injection 
RecProbe is currently under review as a resource paper at SIGIR 2026.

![RecProbe](logo.png)


**RecProbe**, is a resource designed to evaluate robustness by generating controlled perturbations of ratings, reviews, and their combinations. RecProbe features a flexible perturbation module along with a comprehensive evaluation pipeline that compares baseline models on both clean and perturbed datasets, allowing for systematic analysis of noise effects. 

# :bookmark: Outline


# :airplane: Before starting 
## What is RecProbe?
RecProbe is a framework designed to systematically perturb user–item interaction datasets and evaluate how recommender systems behave under realistic noise conditions.
RecProbe provides three level of noise injections, all configurable with dedicated YAML files.
- :star: **rating level**: the noise applies to rating values
  - *Random inconsistencies*: simulates the behavior of a real user, introducing occasional misreported ratings.
  - *Rating bursts*: simulates abnormal activity where a user or an item receives many ratings in a short time, mimicking bots or coordinated behaviors.
  - *Timestamp corruption*: introduces missing, incorrect, or inconsistent timestamps to test models relying on temporal patterns.
- :abcd: **review level**: the noise applies to the textual review
  - *Random inconsistencies*: missing or incomplete reviews to mimic users who rate without providing text.
  - *Semantic noise*: introduces off-topic, irrelevant, or incoherent content, including potential AI/bot-generated text.
  - *Review bursts*: injects large volumes of reviews from a few users or targeting specific items in a short period, simulating spam or fraudulent activity.
- :heavy_plus_sign: **hybrid level**: the noise applies to the combination of textual review and rating.
  - *Random inconsistencies*: creates contradictions between ratings and review sentiment (e.g., a low rating with a highly positive review).
  - *Semantic drift*: introduces divergence between review content and item context (e.g., a review for hiking boots describing them as “perfect for the beach”).
  - *Hybrid bursts*: systematically pairs positive ratings with negative reviews and vice versa, stressing models that rely on both ratings and text.


## Pipeline
![RecProbe](pipeline.png)

**Setup and configuration.** In this phase, the user supplies a configuration for each stage of the pipeline. The first step involves compiling three YAML files, which are used to perturb the original dataset and calculate baseline results.

**Data processing.** The initial dataset is preprocessed for the subsequent phases of the pipeline: in this phase the user can (optionally) extract the k-core, filter by rating o review length.

**Noise injection.** The perturbation is applied, according to the configuration provided by the user. The injection generates a new noisy dataset. 

**Evaluation.** The user defines a set of baselines that enable comparison between the original (unperturbed) dataset and the perturbed version. RecProbe then generates a single comparison table summarizing the results.

## RecProbe Configuration
### Input and Output


### YAML Configuration
### Base
### Injections
### Evaluation


# Installation
## From Docker
RecProbe is distributed as a docker image in order to provide easy deployment independently of the host operative system and infrastructure. 

First, it is required to build the docker image:
```
docker build -t recprobe .
```

### Streamlit user interface


### Manual YAML Configuration
The, run the container and setup manually:
```
docker run --rm  -ti --gpus '"device=1"' --name recprobe-container -v /src/data/:/code/src/ recprobe:latest python3 main.py --profile=rating --noise_injection=rating_burst --baselines
```

The, run the container:
```
docker run --rm  -ti --gpus '"device=1"' --name recprobe-container -v /src/data/:/code/src/ recprobe:latest python3 main.py --profile=rating --noise_injection=rating_burst --baselines
```

To simply enter the container shell:
```
docker run --rm  -ti --gpus '"device=1"' --name recprobe-container -v /src/data/:/code/src/ recprobe:latest 
```
If you want to run the container with no gpu, just avoid the ```--gpus``` parameter.


## From source (for development)
If you want to provide custom methods and implementations, it is recommended to have the entire project. 

```
git clone https://github.com/RecSysQuality/RecProbe.git
cd RecProbe
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Customization

