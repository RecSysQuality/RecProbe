# RecProbe: Stress-Testing Recommender Systems via Systematic Noise Injection 

![RecProbe](logo.png)


RecProbe, is a resource designed to evaluate robustness by generating controlled perturbations of ratings, reviews, and their combinations. RecProbe features a flexible perturbation module along with a comprehensive evaluation pipeline that compares baseline models on both clean and perturbed datasets, allowing for systematic analysis of noise effects. We release RecProbe and its accompanying test suite to facilitate reproducible comparisons across diverse noise settings.

RecProbe is currently under review as a resource paper at SIGIR 2026.

# Installation
## From Docker
RecProbe is distributed as a docker image in order to provide easy deployment independently of the host operative system and infrastructure. 

First, it is required to build the docker image:
```
docker build -t recprobe .
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
# From source (for development)
If you want to provide custom methods and implementations, it is recommended to have the entire project. 

```
git clone https://github.com/RecSysQuality/RecProbe.git
cd RecProbe
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

