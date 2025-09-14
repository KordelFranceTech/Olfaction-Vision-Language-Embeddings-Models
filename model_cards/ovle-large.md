# Model Card: Scentience-OVLE-Large-v1


## Model Details
- **Model Name:** `Scentience OVLE Large v1`
- **Developed by:** Kordel K. France
- **Date:** September 2025
- **Architecture:**
  - **Olfaction encoder:** 138-sensor embedding
  - **Vision encoder:** CLIP-based
  - **Language encoder:** CLIP-based
  - **Fusion strategy:** Joint embedding space via multimodal contrastive training
  - **Parameter Count:** 8.7M (without CLIP), 160.0M (with CLIP)
- **License:** MIT
- **Contact:** kordel@scentience.ai, kordel.france@utdallas.edu

---

## Intended Use
- **Primary purpose:** Research in multimodal machine learning involving olfaction, vision, and language.  
- **Example applications:**
  - Cross-modal retrieval (odor → image, odor → text, etc.)
  - Robotics and UAV navigation guided by chemical cues
  - Chemical dataset exploration and visualization
- **Intended users:** Researchers, developers, and educators working in ML, robotics, chemistry, and HCI.
- **Out of scope:** Not intended for safety-critical tasks (e.g., gas leak detection, medical diagnosis, or regulatory use).

---

## Training Data
- **Olfaction data:** Language-aligned olfactory data curated from GoodScents and LeffingWell datasets.
- **Vision data:** COCO dataset.
- **Language data:** Smell descriptors and text annotations curated from literature.

---

## Evaluation
- Retrieval tasks: odor→image (Top-5 recall = 62%)
- Odor descriptor classification accuracy = 71%
- Cross-modal embedding alignment qualitatively verified on 200 sample triplets.

---

## Limitations of Evaluation
To the best of our knowledge, there are currently no open-source datasets that provide aligned olfactory, visual, and linguistic annotations. A “true” multimodal evaluation would require measuring the chemical composition of scenes (e.g., using gas chromatography mass spectrometry) while simultaneously capturing images and collecting perceptual descriptors from human olfactory judges. Such a benchmark would demand substantial new data collection efforts and instrumentation.
Consequently, we evaluate our models indirectly, using surrogate metrics (e.g., cross-modal retrieval performance, odor descriptor classification accuracy, clustering quality). While these evaluations do not provide ground-truth verification of odor presence in images, they offer a first step toward demonstrating alignment between modalities.
We draw analogy from past successes in ML datasets such as precursors to CLIP that lacked large paired datasets and were evaluated on retrieval-like tasks.
As a result, we release this model to catalyze further research and encourage the community to contribute to building standardized datasets and evaluation protocols for olfaction-vision-language learning.

---

## Limitations
- Limited odor diversity (approx. 5000 unique compounds).
- Embeddings depend on sensor calibration; not guaranteed across devices.
- Cultural subjectivity in smell annotations may bias embeddings.

---

## Ethical Considerations
- Not to be used for covert detection of substances or surveillance.
- Unreliable in safety-critical contexts (e.g., gas leak detection).
- Recognizes cultural sensitivity in smell perception.

---

## Environmental Impact
- Trained on 4×A100 GPUs for 48 hours (~200 kg CO2eq).
- Sensor dataset collection required ~500 lab hours.

---

## Citation
If you use this model, please cite:
```
    @misc{scentience2025ovle,
    title = {Scentience-OVLE-Large-v1: Joint Olfaction-Vision-Language Embeddings},
    author = {Kordel Kade France},
    year = {2025},
    howpublished = {Hugging Face},
    url = {https://huggingface.co/your-username/Scentience-OVLE-Large-v1}
    }
```