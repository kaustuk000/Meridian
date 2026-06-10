"""
visualize_spaces.py
===================
Compares embedding spaces between untuned standard CLIP and fine-tuned Meridian.
Projects embeddings to 2D, spreads labels evenly on an outer ring to eliminate clutter,
and prints dual textual tree diagrams (Untuned vs Fine-Tuned) directly to the console.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from meridian.model import MeridianModel
from meridian.tokenizer import Tokenizer

def project_to_2d(embeddings: torch.Tensor, is_hyperbolic: bool = False) -> np.ndarray:
    """Reduces embeddings to 2D using PCA. Normalizes if hyperbolic."""
    arr = embeddings.detach().cpu().numpy()
    if arr.shape[1] > 2:
        pca = PCA(n_components=2)
        arr_2d = pca.fit_transform(arr)
    else:
        arr_2d = arr

    if is_hyperbolic:
        norms = np.linalg.norm(arr_2d, axis=1, keepdims=True)
        max_norm = np.max(norms)
        if max_norm > 0:
            arr_2d = (arr_2d / max_norm) * 0.90
    return arr_2d

def print_ascii_tree(Z, labels, num_points, title_text):
    """
    Parses the hierarchical linkage matrix and prints a beautifully 
    structured text tree directly to the terminal console.
    """
    # Create a map to hold the string representations of children/branches
    tree_nodes = {i: [f"[{i+1}] {labels[i]}"] for i in range(num_points)}
    
    for i, edge in enumerate(Z):
        child1_idx, child2_idx = int(edge[0]), int(edge[1])
        parent_idx = num_points + i
        
        c1_lines = tree_nodes[child1_idx]
        c2_lines = tree_nodes[child2_idx]
        
        # Build the structured branches textually
        new_lines = []
        new_lines.append("─┬─ " + c1_lines[0])
        for line in c1_lines[1:]:
            new_lines.append(" │  " + line)
        new_lines.append(" └─ " + c2_lines[0])
        for line in c2_lines[1:]:
            new_lines.append("    " + line)
            
        tree_nodes[parent_idx] = new_lines

    # The final root node holds the entire combined tree structure
    root_idx = num_points + len(Z) - 1
    print("\n" + "="*70)
    print(f"      {title_text} (READ LEFT-TO-RIGHT)")
    print("="*70)
    for line in tree_nodes[root_idx]:
        print(line)
    print("="*70 + "\n")

def draw_hierarchical_tree(ax, points, labels, color, is_hyperbolic=False):
    """Plots nodes with cross-reference numbers and callout tracking lines."""
    num_points = len(points)
    
    # 1. Plot the explicit scatter points
    ax.scatter(points[:, 0], points[:, 1], c=color, zorder=5, s=120, edgecolors='black', linewidths=1.0)
    
    # 2. Add identification numbers inside the scatter nodes
    for i in range(num_points):
        ax.text(points[i, 0], points[i, 1], str(i + 1), color='white', fontsize=8, 
                fontweight='bold', ha='center', va='center', zorder=6)

    # 3. Compute uniform outer ring coordinates for labels
    angles = np.arctan2(points[:, 1], points[:, 0])
    sort_idx = np.argsort(angles)
    
    if is_hyperbolic:
        base_radius = 1.20
    else:
        max_dist = np.max(np.linalg.norm(points, axis=1)) if num_points > 0 else 1.0
        base_radius = max(max_dist * 1.4, 1.3)

    target_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    
    for rank, idx in enumerate(sort_idx):
        x, y = points[idx, 0], points[idx, 1]
        label = labels[idx]
        tgt_angle = target_angles[rank]
        
        text_x = base_radius * np.cos(tgt_angle)
        text_y = base_radius * np.sin(tgt_angle)
        
        ha = 'left' if np.cos(tgt_angle) >= 0 else 'right'
        va = 'center'
        
        short_label = label if len(label) < 25 else label[:22] + "..."
        display_str = f"[{idx + 1}] {short_label}"
        
        ax.plot([x, text_x], [y, text_y], color=color, alpha=0.3, linestyle=':', linewidth=1.0, zorder=2)
        ax.text(
            text_x, text_y, display_str, fontsize=8, fontweight='bold', color='black',
            ha=ha, va=va, zorder=6,
            bbox=dict(boxstyle="square,pad=0.2", facecolor='white', alpha=0.95, edgecolor=color, linewidth=0.5)
        )

    # 4. Compute Hierarchical Tree Splits & Plot them
    if num_points > 1:
        Z = linkage(points, method='ward')
        node_positions = {i: (points[i, 0], points[i, 1]) for i in range(num_points)}
        
        for i, edge in enumerate(Z):
            child1_idx, child2_idx = int(edge[0]), int(edge[1])
            parent_idx = num_points + i
            
            p1 = node_positions[child1_idx]
            p2 = node_positions[child2_idx]
            parent_pos = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            node_positions[parent_idx] = parent_pos
            
            ax.plot([p1[0], parent_pos[0]], [p1[1], parent_pos[1]], color=color, alpha=0.4, linestyle='-', linewidth=1.2, zorder=3)
            ax.plot([p2[0], parent_pos[0]], [p2[1], parent_pos[1]], color=color, alpha=0.4, linestyle='-', linewidth=1.2, zorder=3)
            
        # Dynamically trigger console text tree output depending on layout target
        if is_hyperbolic:
            print_ascii_tree(Z, labels, num_points, title_text="FINE-TUNED MERIDIAN HIERARCHY")
        else:
            print_ascii_tree(Z, labels, num_points, title_text="UNTUNED CLIP BASELINE HIERARCHY")
            
    view_pad = base_radius * 1.55
    ax.set_xlim(-view_pad, view_pad)
    ax.set_ylim(-view_pad, view_pad)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running hierarchical visualization on: {device}")

    # =========================================================================
    # 1. Models Initialization
    # =========================================================================
    print("Loading untuned CLIP model...")
    hf_model_id = "checkpoints/clip/hf_vitb16_openai"
    try:
        clip_model = CLIPModel.from_pretrained(hf_model_id).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(hf_model_id)
    except Exception:
        print("Local checkpoint missing. Defaulting to Central HF Hub weights...")
        hf_model_id = "openai/clip-vit-base-patch16"
        clip_model = CLIPModel.from_pretrained(hf_model_id).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(hf_model_id)

    print(f"Loading fine-tuned Meridian model from {args.checkpoint}...")
    meridian_model = MeridianModel(image_hout=16, image_eout=16, text_hout=16, text_eout=16).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    meridian_model.load_state_dict(state_dict)
    meridian_model.eval()
    
    meridian_tokenizer = Tokenizer()

    # =========================================================================
    # 2. Setup Structured Text Data
    # =========================================================================
    texts = [
        "Golden retriever dog", 
        "German shepherd puppy",
        "Fluffy white cat", 
        "Sleek black panther",
        "Red sports car", 
        "Heavy cargo truck",
        "Supersonic jet airplane",
        "Military helicopter",
        "Beautiful red rose", 
        "Bright yellow sunflower",
        "Ancient giant oak tree",
        "Tall green pine tree",
        "Powerful desktop computer",
        "Apple iPhone smartphone",
        "Mechanical gaming keyboard",
        "High resolution monitor",
        "Digital camera lens",
        "Smartphone selfie camera",
        "Stunning sunset over the ocean",
        "Aerial view of the city skyline",
        "Indian elephant",
        "Tiger cub",
        "Cute kitten", "Lion cub",
        "Flamingo", "Eagle",
        "Lioness", "Tiger",
    ]
    dummy_images = [Image.new("RGB", (224, 224), color=(20 * i, 60, 100)) for i in range(len(texts))]

    # =========================================================================
    # 3. Process Embeddings
    # =========================================================================
    hf_inputs = clip_processor(text=texts, images=dummy_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        hf_outputs = clip_model(**hf_inputs)
        hf_txt_embeds = hf_outputs.text_embeds
    hf_txt_2d = project_to_2d(hf_txt_embeds, is_hyperbolic=False)

    tokens = meridian_tokenizer(texts)
    B = len(texts)
    input_ids = torch.zeros(B, 77, dtype=torch.long)
    attention_mask = torch.zeros(B, 77, dtype=torch.long)
    eos_indices = torch.zeros(B, dtype=torch.long)
    
    for i, seq in enumerate(tokens):
        n = min(len(seq), 77)
        input_ids[i, :n] = seq[:n]
        attention_mask[i, :n] = 1
        eos_indices[i] = n - 1
        
    pixel_values = torch.randn(B, 3, 224, 224).to(device) 

    with torch.no_grad():
        meridian_outputs = meridian_model(
            pixel_values=pixel_values,
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            eos_indices=eos_indices.to(device)
        )
    meridian_h_txt_2d = project_to_2d(meridian_outputs["h_text"], is_hyperbolic=True)

    # =========================================================================
    # 4. Generate Plot Layout
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle("Clean Tree Splits & Un-Cluttered Node Visualizations", fontsize=18, fontweight='bold')

    # Left Panel: Euclidean space (Triggers UNTUNED ASCII output to terminal)
    ax1.set_title("Untuned CLIP Baseline Space (Flat Euclidean PCA)", fontsize=14, pad=15)
    draw_hierarchical_tree(ax1, hf_txt_2d, labels=texts, color="royalblue", is_hyperbolic=False)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.set_aspect('equal', 'box')

    # Right Panel: Hyperbolic space (Triggers FINE-TUNED ASCII output to terminal)
    ax2.set_title("Fine-Tuned Meridian Manifold (Hyperbolic Poincaré Tree Splits)", fontsize=14, pad=15)
    disk = Circle((0, 0), 1.0, color='crimson', fill=False, linewidth=2.0, alpha=0.7)
    ax2.add_patch(disk)
    draw_hierarchical_tree(ax2, meridian_h_txt_2d, labels=texts, color="darkviolet", is_hyperbolic=True)
    ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
    ax2.axvline(0, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.set_aspect('equal', 'box')

    plt.tight_layout()
    output_filename = "hierarchical_embedding_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully compiled and written to '{output_filename}'.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Word-Level Hierarchical Trees")
    parser.add_argument("--checkpoint", type=str, default="./output/checkpoint_final.pt", help="Path to Meridian checkpoint file")
    args = parser.parse_args()
    main(args)
