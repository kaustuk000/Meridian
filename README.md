# Meridian: Hyperbolic Multimodal Representation Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Built with UV](https://img.shields.io/badge/built%20with-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Framework PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License CC-BY-NC](https://img.shields.io/badge/license-CC--BY--NC-red.svg)](LICENSE)

Meridian is a vision-language representation model built directly on top of standard **CLIP (ViT-B/16)** backbones. Unlike typical contrastive models that project image and text vectors into a flat, Euclidean vector space, Meridian maps multimodal features onto a continuous **hyperbolic manifold** (Lorentz manifold). 

Because the volume of hyperbolic space grows exponentially rather than polynomially, it possesses an intrinsic "tree-like" geometry. This geometry makes Meridian exceptionally robust at capturing semantic hierarchies, complex taxonomies, and fine-grained conceptual ontologies without suffering from the representation collapse typical of Euclidean projections.

---

## Architecture

![Meridian Architecture](assets/Meridian%20Architecture.png)

## Key Features

* **CLIP ViT-B/16 Foundation:** Leverages rich, pre-trained multimodal representations out-of-the-box before mapping to hyperbolic structures.
* **Non-Linear Hyperbolic Projection:** Custom multi-layer projection heads (`LayerAggregators`) map Euclidean embeddings smoothly into stable Lorentz manifold coordinates.
* **Hierarchical Semantic Tracking:** Naturally groups open-vocabulary expressions into clean hierarchical tree splits (via Ward's linkage matrix), cleanly isolating distinct domain clusters.
* **Scalable WebDataset Dataloader:** Built to stream large-scale pretraining datasets like Conceptual Captions 3M (CC3M) even through natural web URL decay.
* **Flexible API:** Production-ready FastAPI endpoint for inference and batch processing.

---
## Highlights

- **~4× Embedding Compression:** Reduced embedding storage requirements by approximately 4× compared to the CLIP baseline through hierarchical hyperbolic representations.
- **~1.5× Faster Retrieval:** Achieved faster retrieval latency while maintaining strong semantic search capabilities.
- **Hierarchical Multimodal Retrieval:** Organizes image-text data into semantic hierarchies for interpretable exploration and retrieval.
- **Multimodal Search:** Supports text-to-image, image-to-image, and hybrid image+text retrieval.
- **Interactive Visualization:** Generates navigable semantic trees for large-scale image collections.

---

## Performance

Evaluation on MS-COCO retrieval.

| Variant | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |
|----------|----------|----------|-----------|----------|----------|-----------|
| Meridian (64d) | 29.66 | 55.20 | 67.18 | 25.29 | 51.00 | 63.02 |

Where:

- **i2t** = Image → Text Retrieval
- **t2i** = Text → Image Retrieval
- Retained ∼70–78% I2T and ∼87–91% T2I recall (R@5/10) relative to the full-dim CLIP zero-shot baseline, with ∼1.6× retrieval speedup on a 1.7M-item index.

## Hierarchy Comparison

The examples below illustrate how Meridian learns cleaner semantic hierarchies than the frozen CLIP baseline by reducing cross-category mixing and improving semantic specialization when organizing concepts derived from text queries.

### Example 1: Animal Concepts

**Frozen CLIP (ViT-B/16)**

<pre>
Mixed Animal Region
├── Dog + Cat
├── Flower
├── Lion
├── Puppy
├── Tiger
├── Elephant
└── Tree
</pre>

**Fine-Tuned Meridian**

<pre>
Feline Region
├── Domestic Cat
├── Domestic Kitten
├── Tabby Cat
├── Tabby Kitten
├── Tiger
├── Tiger Cub
├── Lion
├── Female Lion
└── Big Cat Cub
</pre>

Meridian forms a coherent feline semantic neighborhood, while the frozen CLIP hierarchy mixes animals, plants, and unrelated concepts within the same region.

---

### Example 2: Transportation Concepts

**Frozen CLIP (ViT-B/16)**

<pre>
Transportation Region
├── Passenger Aircraft
├── Cargo Truck
├── Commercial Airliner
├── Sports Car
└── Semi Truck
</pre>

**Fine-Tuned Meridian**

<pre>
Aircraft
├── Passenger Aircraft
├── Commercial Airliner
├── Fighter Jet
└── Stealth Aircraft

Ground Vehicles
├── Cargo Truck
├── Semi Truck
└── Sports Car
</pre>

---

### Example 3: Computing Devices

**Frozen CLIP (ViT-B/16)**

<pre>
Mixed Technology Region
├── Heavy Commercial Truck
├── Photography Lens
├── Mobile Phone Camera
├── Smartphone
├── Mechanical Keyboard
├── Mirrorless Camera
├── Laptop
├── Monitor
└── Workspace
</pre>

**Fine-Tuned Meridian**

<pre>
Workspace
├── Laptop on Desk
├── Thin Silver Laptop
├── Office Laptop
├── Ultrawide Monitor
├── Keyboard
├── Mouse
├── Desktop Workstation
└── Monitor Setup

Camera System
├── Mobile Phone Camera
├── Photography Lens
├── Mirrorless Camera
└── Visual Memory Device
</pre>

Meridian organizes computing and imaging concepts into coherent semantic regions, whereas the frozen CLIP hierarchy mixes cameras, laptops, peripherals, smartphones, and unrelated objects within the same branch.

Meridian separates aircraft and ground vehicles into distinct semantic branches, whereas the frozen CLIP hierarchy intermixes them within the same cluster.

**Key Observation:** Meridian significantly reduces cross-category mixing and produces cleaner semantic neighborhoods, resulting in more interpretable hierarchical structures.

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

This repository requires Python 3.12+ and utilizes `uv` for dependency management.

### Backend Setup

```bash
git clone https://github.com/kaustuk000/Meridian.git
cd Meridian

uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

uv sync
```

### Frontend Setup

The interactive hierarchy visualization frontend is located in `api/frontend` and uses Vite + React.

```bash
cd api/frontend
npm install
```

### Run Frontend

```bash
cd api/frontend
npm run dev
```

Default URL:

```text
http://localhost:5173
```

### Run Backend API

```bash
uvicorn api.app:app --reload --port 8000  
```

Default API URL:

```text
http://localhost:8000
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

### Reference Training Run

```bash
uv run python -m scripts.train \
    --batch-size 196 \
    --workers 8 \
    --warmup-steps 6000 \
    --total-iterations 150000 \
    --output-dir checkpoints/meridian_v1
```

Key configuration:

- Dataset: CC3M (~1.51M surviving image-text pairs)
- Training Steps: 150,000
- Batch Size: 196
- Warmup Steps: 6,000
- DataLoader Workers: 8
- CLIP ViT-B/16 encoders frozen during training
- Learnable layer aggregation, projection heads, and adaptive gating


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

# Hyperbolic Head embeddings
h_image = outputs["h_image"]
h_text = outputs["h_text"]

# Euclidean Head embeddings
e_image = outputs["e_image"]
e_text = outputs["e_text"]

print("Hyperbolic Image Embedding Shape:", h_image.shape)
print("Hyperbolic Text Embedding Shape:", h_text.shape)
```

---

## Project Structure

```
Meridian/
├── api/
│   ├── app.py            # backend endpoint
│   ├── inference.py      # score calculation for retrival
│   └── frontend/
│       ├── src/
│       ├── public/
│       ├── package.json
│       ├── vite.config.js
│       └── ...
├── meridian/                # Core library
│   ├── model.py            # Main Meridian model architecture
│   ├── lorentz.py          # Hyperbolic geometry operations
│   ├── losses.py           # Custom loss functions
│   ├── optim.py            # Optimization utilities
│   ├── tokenizer.py        # Text tokenization
│   └── data/               # Data loading utilities
│       ├── cc3m.py        # CC3M dataloader
|       ├── cc3m.tsv # CC3M Image-link tsv([download link](https://huggingface.co/datasets/yxchng/cc15m_yfcc15m/resolve/main/cc3m.tsv))
|       ├──cc3m_smoke/
|       |   ├── *_stats.json
|       |   ├── *_paraquet
|       |   └── *.tar
|       ├── dataset_download.py # (use to download dataset with --dataset argument or use CLI)
│       ├── transforms.py  # Image/text preprocessing
│       └── evaluation.py  # Evaluation dataset loaders
│         
├── scripts/                # Training and evaluation scripts
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation script
│   ├── visualize.py      # Visualization utilities
│   ├── build_index.py    # Used for building the index
│   └── download_models.py # Use it only for model modifications requiring CLIP; otherwise, the backend API downloads both Meridian and CLIP from Hugging Face.
├── checkpoints/            
|── assets/
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
  howpublished = {\url{https://github.com/kaustuk000/Meridian}},
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
