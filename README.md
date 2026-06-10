# Meridian: Hyperbolic Multimodal Representation Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Framework PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License CC-BY-NC](https://img.shields.io/badge/license-CC--BY--NC-red.svg)](LICENSE)

Meridian is a vision-language representation model built directly on top of standard **CLIP (ViT-B/16)** backbones. Unlike typical contrastive models that project image and text vectors into a flat, Euclidean vector space, Meridian maps multimodal features onto a continuous **hyperbolic manifold** (Lorentz manifold). 

Because the volume of hyperbolic space grows exponentially rather than polynomially, it possesses an intrinsic "tree-like" geometry. This geometry makes Meridian exceptionally robust at capturing semantic hierarchies, complex taxonomies, and fine-grained conceptual ontologies without suffering from the representation collapse typical of Euclidean projections.

---

## Key Features

* **CLIP ViT-B/16 Foundation:** Leverages rich, pre-trained multimodal representations out-of-the-box before mapping to hyperbolic structures.
* **Non-Linear Hyperbolic Projection:** Custom multi-layer projection heads (`LayerAggregators`) map Euclidean embeddings smoothly into stable Lorentz manifold coordinates.
* **Hierarchical Semantic Tracking:** Naturally groups open-vocabulary expressions into clean hierarchical tree splits (via Ward's linkage matrix), cleanly isolating distinct domain clusters.
* **Scalable WebDataset Dataloader:** Built to stream large-scale pretraining datasets like Conceptual Captions 3M (CC3M) even through natural web URL decay.
* **Flexible API:** Production-ready FastAPI endpoint for inference and batch processing [Not ready yet].

---

## Geometric Intuition

In a standard Euclidean embedding space, broad concepts and specific attributes are forced onto the same flat plane, causing unrelated tokens to crowd and overlap near the edges. 

Hyperbolic geometry resolves this by introducing an implicit hierarchical gradient:
* **The Origin (Center):** General, overarching parent concepts (e.g., *Entity*, *Vehicle*, *Animal*) naturally gravitate toward the center of the disk.
* **The Boundary (Edge):** Highly specific leaf-nodes (e.g., *Supersonic jet airplane*, *German shepherd puppy*) branch outwards along continuous paths toward the perimeter.

The geodesic distance between two points on the Lorentz manifold is defined as:

`d_L(x, y) = (1/√c) arcosh(-c ⟨x, y⟩_L)`

where the Lorentzian inner product is:

`⟨x, y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ`

As embeddings approach the boundary (||u||, ||v|| → 1), the denominator shrinks, causing the distance to grow exponentially—giving the model infinite room to isolate dense clusters cleanly.

---

## Installation & Setup

This repository requires Python 3.12+ and utilizes `uv` for ultra-fast, reproducible dependency and environment management.

### Prerequisites

- Python 3.12 or higher
- CUDA 12.8+ (for GPU acceleration; CPU-only installation also supported)
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kaustuk000/Meridian.git
   cd Meridian
   ```

2. **Create a virtual environment using `uv`:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Download pre-trained checkpoints (optional):**
   ```bash
   uv run python -m scripts.download_models
   ```

---

## Quick Start

### Training

To train a Meridian model on the CC3M dataset:

```bash
uv run python -m scripts.train \
    --batch-size 128 \
    --warmup-steps 2000
    --total-iterations 10000 \
    --output-dir meridian/checkpoints/meridian_v1 \
    --workers 4

```

### Inference

Use the built-in API for inference:

```bash
python -m api.app
```

Or programmatically:

```python
import torch

from meridian.model import MeridianModel
from meridian.tokenizer import Tokenizer

# Initialize model
model = MeridianModel()
model.eval()

# Example inputs
pixel_values = torch.randn(1, 3, 224, 224)

tokenizer = Tokenizer()
tokens = tokenizer(["A photo of a golden retriever"])

with torch.no_grad():
    outputs = model(
        pixel_values=pixel_values,
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        eos_indices=tokens["eos_indices"],
    )

# Hyperbolic embeddings
h_image = outputs["h_image"]
h_text = outputs["h_text"]

# Euclidean embeddings
e_image = outputs["e_image"]
e_text = outputs["e_text"]

print("Hyperbolic Image Embedding Shape:", h_image.shape)
print("Hyperbolic Text Embedding Shape:", h_text.shape)
```

### Evaluation

Run evaluation on downstream tasks:

```bash
python scripts/evaluate.py \
    --model-path meridian/checkpoints/meridian_v1 \
    --eval-datasets imagenet zeroshot
```

---

## Project Structure

```
Meridian/
├── api/                      # FastAPI inference server
│   ├── app.py               # Application entry point
│   ├── inference.py         # Inference pipeline
│   └── schemas.py           # Request/response schemas
├── meridian/                # Core library
│   ├── model.py            # Main Meridian model architecture
│   ├── lorentz.py          # Hyperbolic geometry operations
│   ├── losses.py           # Custom loss functions
│   ├── optim.py            # Optimization utilities
│   ├── tokenizer.py        # Text tokenization
│   ├── data/               # Data loading utilities
│   │   ├── cc3m.py        # CC3M dataloader
|   |   ├── cc3m.tsv  # CC3M Image-link tsv([download link](https://huggingface.co/datasets/yxchng/cc15m_yfcc15m/resolve/main/cc3m.tsv))
|   |   ├──cc3m_smoke/
|   |   |   ├── *_stats.json
|   |   |   ├── *_paraquet
|   |   |   └── *.tar
|   |   ├── dataset_download.py # (use to download dataset with --dataset argument or use CLI)
│   │   ├── transforms.py  # Image/text preprocessing
│   │   └── evaluation.py  # Evaluation dataset loaders
│   └── utils/              # Utility functions
│       ├── checkpointing.py
│       ├── distributed.py
│       └── metrics.py
├── scripts/                # Training and evaluation scripts
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation script
│   ├── visualize.py       # Visualization utilities
│   └── download_models.py # Download pre-trained models
├── configs/               # Configuration files
│   ├── train_vit_b.py
│   ├── eval_zsc.py
│   └── eval_retrieval.py
├── notebooks/             # Jupyter notebooks
├── checkpoints/           # Model checkpoints(frozen **CLIP (ViT-B/16)**)
└── README.md
```

---

## Model Architecture

Meridian consists of 5 main components:

1. **Image Encoder:** OpenAI CLIP ViT-B/16 vision transformer
2. **Text Encoder:** OpenAI CLIP text transformer
3. **Hyperbolic Projection Heads:**
   - Layer aggregation for multi-head embeddings
   - Non-linear transformer adapters
   - Exponential map to Lorentz manifold
   - Lorentz distance-based contrastive loss
4. **Euclidean Projection Head:**
   - Same as the above but for normal euclidean manifold
5. **Dynamic Gating between Euclidean Head and Hyperbolic Projection Heads**
   - To let model decide when to focus on which head more

The model is trained end-to-end using a hybrid hyperbolic–Euclidean contrastive loss. The objective combines Lorentzian contrastive learning, Euclidean contrastive learning, gated representation fusion, and hyperbolic entailment modeling while respecting the Riemannian geometry of the Lorentz manifold.

---

## Dependencies

Key dependencies include:

- **PyTorch 2.10+:** Deep learning framework
- **Transformers 5.10+:** Pre-trained models
- **Open-CLIP:** CLIP implementations
- **Scikit-learn:** Classical ML utilities
- **Loguru:** Logging
- **TensorBoard:** Training visualization

See `pyproject.toml` for the complete dependency list.

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

This project uses `black` for formatting and `ruff` for linting:

```bash
black meridian/
ruff check meridian/
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citing Meridian & MERU

### Meridian

If you use Meridian in your research, please use the following BibTeX entry:

```bibtex
@misc{singh2026meridian,
  title        = {Meridian: Hyperbolic Image--Text Representations},
  author       = {Kaustuk Pratap Singh},
  year         = {2026},
  howpublished = {\url{https://github.com/<kaustuk000>/Meridian}},
  note         = {Open-source research project}
}
```

### MERU (Original Hyperbolic Image-Text Representations)

This project builds upon and incorporates concepts from MERU. If you use any parts of the MERU codebase in your research, please cite the original work:

```bibtex
@inproceedings{desai2023meru,
    title     = {{Hyperbolic Image-Text Representations}},
    author    = {Desai, Karan and Nickel, Maximilian and Rajpurohit, Tanmay and Johnson, Justin and Vedantam, Ramakrishna}
    booktitle = {Proceedings of the International Conference on Machine Learning},
    year      = {2023}
}
```

**License Note:** The majority of this project is licensed under CC-BY-NC, however portions adapted from MERU are available under separate license terms: [CLIP](https://github.com/openai/CLIP), [SLIP](https://github.com/facebookresearch/slip), and [ViTEx](https://github.com/kdexd/virtex) are licensed under the MIT license.

---

## License

Meridian is released under the **CC-BY-NC License** (Creative Commons Attribution-NonCommercial 4.0 International). This means:

**You may:**
- Use the code for non-commercial research and educational purposes
- Modify and adapt the code
- Share the code with proper attribution

**You may not:**
- Use the code for commercial purposes without explicit permission
- Remove or modify the license headers

### Third-Party Licenses

This project incorporates code from the following projects, which are licensed under the **MIT License**:

- [OpenAI CLIP](https://github.com/openai/CLIP) - Vision-language model
- [Open-CLIP](https://github.com/mlfoundations/open_clip) - Open-source CLIP implementations

For details, see individual license files in their respective source directories.

---

## Support & Contact

For questions, bug reports, or feature requests:

- **GitHub Issues:** [Create an issue](https://github.com/kaustuk000/Meridian/issues)
- **Discussions:** [Start a discussion](https://github.com/kaustuk000/Meridian/discussions)
- **GitHub:** [@kaustuk000](https://github.com/kaustuk000)

---

## Acknowledgments

- OpenAI for the CLIP model
- Meta AI Research for foundational work on hyperbolic embeddings (MERU)
- Conceptual Captions team for the CC3M dataset

---

## Related Work

- [Learning to Embed Words in Context](https://arxiv.org/abs/1610.02055) - Lorentz model foundations
- [CLIP: Learning Transferable Models](https://arxiv.org/abs/2103.14030) - Vision-language models
- [Hyperbolic Image Embeddings](https://arxiv.org/abs/1904.02239) - Hierarchical image embeddings

---

**Last Updated:** June 2024
