---
language: 
  - en
tags:
- embeddings
- multimodal
- olfaction-vision-language
- olfaction
- olfactory
- scentience
- neural-network
- graph-neural-network
- gnn
- vision-language
- vision
- language
- robotics
- multimodal
- smell
- clip
- siglip
license: mit
datasets:
- kordelfrance/olfaction-vision-language-dataset
- detection-datasets/coco
- seyonec/goodscents_leffingwell
base_model: Scentience-COLIP-Base
---

# COLIP: Contrastive Olfaction-Language-Image Pre-training Model
## Olfaction-Vision-Language Embeddings


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow?logo=google-colab)](https://colab.research.google.com/drive/1H5OSeO43YfhAT9MqcJKaaSknFYhjimvg?usp=sharing)
[![Paper](https://img.shields.io/badge/Research-Paper-red)](https://arxiv.org/abs/2506.00398)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/kordelfrance/Olfaction-Vision-Language-Embeddings)

</div>

---

## Description

This repository is a foundational series of multimodal joint embedding models trained on olfaction, vision, and language data.
These models are built specifically for prototyping and exploratory tasks within AR/VR, robotics, and embodied artificial intelligence.
Analogous to how CLIP and SigLIP embeddings give vision-language relationships, our embeddings models here give olfaction-vision-language (OVL) relationships.
The models were built on top of CLIP and extended to olfaction.

Whether these models are used for better vision-scent navigation with drones, triangulating the source of an odor in an image, extracting aromas from a scene, or augmenting a VR experience with scent, we hope their release will catalyze further research in olfaction, especially olfactory robotics.
We especially hope these models encourage the community to contribute to building standardized datasets and evaluation protocols for olfaction-vision-language learning.

## Models
We offer four olfaction-vision-language (OVL) embedding models with this repository:
 - (1) `colip-large-base`: The original OVL base model. This model is optimal for online tasks where accuracy is critical.
 - (2) `colip-large-gat`: The OVL base model built around a graph-attention network. This model is optimal for online tasks where accuracy is paramount and inference time is not as critical.
 - (3) `colip-small-base`: The original OVL base model optimized for faster inference and edge-based robotics. This model is optimized for export to common frameworks that run on Android, iOS, Rust, and others.
 - (4) `colip-small-gat`: The OVL graph-attention model optimized for faster inference and edge robotics applications.

## Training Data
A sample dataset is included, but the full datasets are linked in the `Datasets` pane of this repo.
Training code for replicating full construction of all models will be released soon.


To the best of our knowledge, there are currently no open-source datasets that provide jointly aligned olfactory, visual, and linguistic annotations.
A “true” multimodal evaluation would require measuring the chemical composition of scenes (e.g., using gas-chromatography mass-spectrometry) while simultaneously capturing images and collecting perceptual descriptors from human olfactory judges. 
Such a benchmark would demand substantial new data collection efforts and instrumentation.
Consequently, we evaluate our models indirectly, using surrogate metrics (e.g., cross-modal retrieval performance, odor descriptor classification accuracy, clustering quality). 
While these evaluations do not provide ground-truth verification of odor presence in images, they offer a first step toward demonstrating alignment between modalities and are perfect for exploratory research.
We draw analogy from past successes in ML datasets, such as precursors to CLIP that lacked large data pairings and were evaluated on retrieval-like tasks.
Just as CLIP used contrastive objectives to construct vision-language relationships, we borrow similar principles to strengthen olfaction-vision-language weights. 
Humans interpret smell with lingual descriptors such as "fruity" and "musky", allowing language to act as a bridge between olfaction and vision data.


## Directory Structure

```text
Olfaction-Vision-Language-Embeddings-Models/
├── data/                     # Sample training dataset
├── requirements.txt          # Python dependencies
├── model/                    # COLIP embedding models
├── model_cards/              # Specifications for each embedding model
├── notebooks/                # Notebooks for replicating training and loading the models for inference
├── src/                      # Source code for inference, model loading, utils
└── README.md                 # Overview of repository contributions and usage
```

---

## Citation
If you use any of these models, please cite:
```
    @misc{france2025colip-ovlembeddings,
        title = {Scentience-COLIP-v1: Joint Olfaction-Vision-Language Embeddings},
        author = {Kordel Kade France},
        year = {2025},
        howpublished = {Hugging Face},
        url = {https://huggingface.co/kordelfrance/Olfaction-Vision-Language-Embeddings}
    }
```

```
    @misc{france2025olfactionstandards,
          title={Position: Olfaction Standardization is Essential for the Advancement of Embodied Artificial Intelligence}, 
          author={Kordel K. France and Rohith Peddi and Nik Dennler and Ovidiu Daescu},
          year={2025},
          eprint={2506.00398},
          archivePrefix={arXiv},
          primaryClass={cs.AI},
          url={https://arxiv.org/abs/2506.00398}, 
    }
```


With any use of COLIP, please also cite the original CLIP and SigLIP papers:
```
    @misc{radford2021clip,
        title        = {Learning Transferable Visual Models From Natural Language Supervision},
        author       = {Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
        year         = 2021,
        url          = {https://arxiv.org/abs/2103.00020},
        eprint       = {2103.00020},
        archiveprefix = {arXiv},
        primaryclass = {cs.CV}
    }
```

```
    @misc{zhai2023siglip,
          title={Sigmoid Loss for Language Image Pre-Training}, 
          author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
          year={2023},
          eprint={2303.15343},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2303.15343}, 
}
```