# Olfaction-Vision-Language Embeddings
(In work)


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow?logo=google-colab)](https://colab.research.google.com/drive/)
[![Paper](https://img.shields.io/badge/Research-Paper-red)](https://arxiv.org/abs/2506.00398)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces)

</div>

---

## Description

This repository is a series of multimodal machine learning models trained on olfaction, vision, and language data specifically for tasks in robotics and embodied artificial intelligence.


To the best of our knowledge, there are currently no open-source datasets that provide aligned olfactory, visual, and linguistic annotations. 
A “true” multimodal evaluation would require measuring the chemical composition of scenes (e.g., using gas chromatography mass spectrometry) while simultaneously capturing images and collecting perceptual descriptors from human olfactory judges. 
Such a benchmark would demand substantial new data collection efforts and instrumentation.
Consequently, we evaluate our models indirectly, using surrogate metrics (e.g., cross-modal retrieval performance, odor descriptor classification accuracy, clustering quality). 
While these evaluations do not provide ground-truth verification of odor presence in images, they offer a first step toward demonstrating alignment between modalities.
We draw analogy from past successes in ML datasets such as precursors to CLIP that lacked large paired datasets and were evaluated on retrieval-like tasks.
As a result, we release this model to catalyze further research and encourage the community to contribute to building standardized datasets and evaluation protocols for olfaction-vision-language learning.


## Models
We offer four embedding models with this repository:
 - (1) `ovle-large-base`: The original OVL base model. This model is optimal for online tasks where accuracy is paramount.
 - (2) `ovle-large-graph`: The OVL base model built around a graph-attention network. This model is optimal for online tasks where accuracy is paramount and inference time is not as critical.
 - (3) `ovle-small-base`: The original OVL base model optimized for faster inference and edge-based robotics. This model is optimized for export to common frameworks that run on Android, iOS, Rust, and others.
 - (4) `ovle-small-graph`: The OVL graph model optimized for faster inference and edge robotics applications.

## Directory Structure

```text
Olfaction-Vision-Language-Embeddings-Models/
├── data/                     # Training datasets
├── requirements.txt          # Python dependencies
├── model/                    # Embeddings models
├── model_cards/              # Specifications for each embedding model
├── notebooks/                # Notebooks for loading the models for inference
└── README.md                 # Overview of repository contributions and usage
```

---

## Citation
If you use any of these models, please cite:
```
@misc{scentience2025ovle,
    title = {Scentience-OVLE-Base-v1: Joint Olfaction-Vision-Language Embeddings},
    author = {Kordel Kade France},
    year = {2025},
    howpublished = {Hugging Face},
    url = {https://huggingface.co/your-username/Scentience-OVLE-Large-v1}
}
```