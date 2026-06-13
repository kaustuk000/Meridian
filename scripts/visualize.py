from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import json
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import scipy for rigorous hierarchical clustering
from scipy.cluster.hierarchy import linkage, to_tree, fcluster
import scipy.spatial.distance as ssd

from meridian.lorentz import lorentz_distance
from meridian.model import MeridianModel
from meridian.tokenizer import Tokenizer
from app.inference import MeridianSearchEngine


# ---------------------------------------------------------------------------
# Setup & Helper Processing Methods
# ---------------------------------------------------------------------------

def load_meridian_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    def _find_dim(prefix: str) -> int:
        for k, v in state_dict.items():
            if k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2 and v.shape[1] == 128:
                return int(v.shape[0])
        raise RuntimeError(f"Could not infer dimension from checkpoint prefix '{prefix}'.")

    model = MeridianModel(
        image_hout=_find_dim("hyp_image_head.image_mlp."),
        image_eout=_find_dim("eucl_image_head.image_mlp."),
        text_hout=_find_dim("hyp_text_head.text_mlp."),
        text_eout=_find_dim("eucl_text_head.text_mlp.")
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def pad_token_batch(tokenizer: Tokenizer, texts: list[str], device: torch.device):
    token_list = tokenizer(texts)
    max_len = 77
    B = len(token_list)
    input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    eos_indices = torch.zeros(B, dtype=torch.long, device=device)

    for i, tokens in enumerate(token_list):
        seq = tokens[:max_len].long().to(device)
        n = len(seq)
        input_ids[i, :n] = seq
        attention_mask[i, :n] = 1
        eos_indices[i] = n - 1
    return input_ids, attention_mask, eos_indices

def inverse_clip_normalize(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=batch.device).view(1, 3, 1, 1)
    return (batch * std + mean).clamp(0, 1)

def display_image_grid(images: list[torch.Tensor], titles: list[str], ncols: int = 3, fig_title: str = ""):
    n = len(images)
    if n == 0:
        print("No matches to visualize.")
        return
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    for ax, img, title in zip(axes, images, titles):
        img_np = inverse_clip_normalize(img.unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Hierarchy Builders (Interactive Tree & Physical Folders)
# ---------------------------------------------------------------------------

def build_d3_tree_json(node, labels):
    """Recursively parses scipy linkage tree into a nested dictionary for D3.js."""
    if node.is_leaf():
        return {"name": labels[node.id]}
    return {
        "name": f"Node_{node.id}",
        "children": [
            build_d3_tree_json(node.left, labels),
            build_d3_tree_json(node.right, labels)
        ]
    }

def generate_interactive_html(tree_dict, output_file="interactive_tree.html"):
    """Embeds the hierarchy into a lightweight, interactive HTML file using D3.js."""
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Meridian Interactive Hierarchy</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: sans-serif; background: #f8f9fa; margin: 0; }}
        .node circle {{ fill: #999; stroke: steelblue; stroke-width: 3px; cursor: pointer; }}
        .node text {{ font: 12px sans-serif; fill: #333; }}
        .link {{ fill: none; stroke: #ccc; stroke-width: 2px; }}
        #title {{ text-align: center; padding: 20px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div id="title"><h2>Meridian Hierarchical Clustering Tree</h2><p>Click on nodes to expand/collapse</p></div>
    <div id="d3-container"></div>
    <script>
        const treeData = {json.dumps(tree_dict)};
        
        const width = window.innerWidth - 100;
        const dx = 20; const dy = width / 6;
        const margin = {{top: 40, right: 120, bottom: 40, left: 120}};

        const tree = d3.tree().nodeSize([dx, dy]);
        const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);

        const svg = d3.select("#d3-container").append("svg")
            .attr("width", width)
            .attr("height", 1000)
            .attr("viewBox", [-margin.left, -margin.top, width, dx]);

        const gLink = svg.append("g").attr("fill", "none").attr("stroke", "#555").attr("stroke-opacity", 0.4).attr("stroke-width", 1.5);
        const gNode = svg.append("g").attr("cursor", "pointer").attr("pointer-events", "all");

        const root = d3.hierarchy(treeData);
        root.x0 = dy / 2; root.y0 = 0;
        
        // Collapse by default for large trees
        if (root.children) root.children.forEach(collapse);
        function collapse(d) {{ if(d.children) {{ d._children = d.children; d._children.forEach(collapse); d.children = null; }} }}

        update(root);

        function update(source) {{
            const nodes = root.descendants().reverse();
            const links = root.links();
            
            tree(root);
            
            let left = root; let right = root;
            root.eachBefore(node => {{ if (node.x < left.x) left = node; if (node.x > right.x) right = node; }});
            const height = right.x - left.x + margin.top + margin.bottom;
            svg.transition().duration(250).attr("viewBox", [-margin.left, left.x - margin.top, width, height]);

            const node = gNode.selectAll("g").data(nodes, d => d.id || (d.id = ++i));
            let i = 0;

            const nodeEnter = node.enter().append("g")
                .attr("transform", d => `translate(${{source.y0}},${{source.x0}})`)
                .attr("fill-opacity", 0)
                .attr("stroke-opacity", 0)
                .on("click", (event, d) => {{
                    d.children = d.children ? null : d._children;
                    update(d);
                }});

            nodeEnter.append("circle").attr("r", 4.5).attr("fill", d => d._children ? "#555" : "#999").attr("stroke-width", 10);
            nodeEnter.append("text").attr("dy", "0.31em").attr("x", d => d._children ? -6 : 6).attr("text-anchor", d => d._children ? "end" : "start")
                .text(d => d.data.name).clone(true).lower().attr("stroke-linejoin", "round").attr("stroke-width", 3).attr("stroke", "white");

            const nodeUpdate = node.merge(nodeEnter).transition().duration(250)
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
                .attr("fill-opacity", 1).attr("stroke-opacity", 1);
            
            nodeUpdate.select("circle").attr("fill", d => d._children ? "#555" : "#999");

            const nodeExit = node.exit().transition().duration(250).remove()
                .attr("transform", d => `translate(${{source.y}},${{source.x}})`).attr("fill-opacity", 0).attr("stroke-opacity", 0);

            const link = gLink.selectAll("path").data(links, d => d.target.id);
            const linkEnter = link.enter().append("path").attr("d", d => {{ const o = {{x: source.x0, y: source.y0}}; return diagonal({{source: o, target: o}}); }});
            link.merge(linkEnter).transition().duration(250).attr("d", diagonal);
            link.exit().transition().duration(250).remove().attr("d", d => {{ const o = {{x: source.x, y: source.y}}; return diagonal({{source: o, target: o}}); }});

            root.eachBefore(d => {{ d.x0 = d.x; d.y0 = d.y; }});
        }}
    </script>
</body>
</html>
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    return output_file

# ---------------------------------------------------------------------------
# Core Interactive Search & Hierarchy System
# ---------------------------------------------------------------------------

def run_search_query(model, tokenizer, engine, text_query, image_path, topk, device):
    """Processes search queries with support for single or comma-separated strings."""
    queries = [text_query] if text_query and "," not in text_query else ([t.strip() for t in text_query.split(",")] if text_query else [None])
    
    for q in queries:
        has_text = q is not None and len(q) > 0
        has_img = image_path is not None and os.path.exists(image_path)
        
        pixel_values = torch.zeros(1, 3, 224, 224, device=device) # Dummy tensor placeholder
        input_ids, attention_mask, eos_indices = pad_token_batch(tokenizer, [q if has_text else "a photo"], device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, eos_indices=eos_indices)
            
        query_payload = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
        scores = engine.compute_scores(query_payload, query_has_text=has_text, query_has_image=has_img)
        
        values, indices = torch.topk(scores, k=min(topk, len(scores)))
        
        title_str = f"Query Result for: '{q}'" if q else "Query Result for Image input"
        print(f"\n{title_str}\n" + "-"*len(title_str))
        
        picked_imgs, picked_titles = [], []
        for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
            caption = engine.index["captions"][idx]
            print(f"  {rank}. {caption} (Score={score:.4f})")
            picked_imgs.append(engine.index["pixel_values"][idx])
            picked_titles.append(f"Rank {rank}\nScore: {score:.4f}")
            
        display_image_grid(picked_imgs, picked_titles, ncols=3, fig_title=title_str)

def run_hierarchy_folder(model, tokenizer, folder_path, device):
    """Evaluates images, computes exact hyperbolic manifolds, and outputs interactive trees & folder clusters."""
    if not os.path.exists(folder_path):
        print(f"Error: Target directory does not exist: {folder_path}")
        return
        
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.BMP", "*.webp"]:
        img_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not img_paths:
        print(f"No valid image files discovered inside: {folder_path}")
        return
        
    print(f"\nDiscovered {len(img_paths)} images.")
    use_filenames = input("Do you want to use the filenames as semantic text queries to refine the clustering? (y/n): ").strip().lower() == 'y'
    
    print(f"Processing topological manifold constraints...")
    h_image_feats, h_text_feats, labels = [], [], []
    
    for path in sorted(img_paths):
        name = os.path.basename(path)
        # We parse the filename cleanly just in case they opted in
        text_query = os.path.splitext(name)[0].replace("_", " ").replace("-", " ") if use_filenames else "a photo"
        
        pixel_values = torch.zeros(1, 3, 224, 224, device=device) # Dummy visual tensor 
        input_ids, attention_mask, eos_indices = pad_token_batch(tokenizer, [text_query], device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, eos_indices=eos_indices)
        
        h_image_feats.append(outputs["h_image"])
        h_text_feats.append(outputs["h_text"])
        labels.append(name)

    n = len(labels)
    # Pre-compute exact pairwise distance matrix mathematically
    dist_matrix = np.zeros((n, n))
    curv = torch.tensor(1.0, device=device)
    
    print("Calculating pairwise distances...")
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                # If they opted to use filenames, we fuse the text distance and image distance
                if use_filenames:
                    d_img = lorentz_distance(h_image_feats[i], h_image_feats[j], curv).item()
                    d_txt = lorentz_distance(h_text_feats[i], h_text_feats[j], curv).item()
                    dist = 0.5 * d_img + 0.5 * d_txt
                else:
                    dist = lorentz_distance(h_image_feats[i], h_image_feats[j], curv).item()
                    
                dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Convert to SciPy condensed format and perform hierarchical linking
    condensed_dist = ssd.squareform(dist_matrix)
    Z = linkage(condensed_dist, method='average')
    
    # 1. Output the interactive D3 web tree
    tree_root, _ = to_tree(Z, rd=True)
    tree_dict = build_d3_tree_json(tree_root, labels)
    html_path = generate_interactive_html(tree_dict)
    
    print(f"\nSuccess! Interactive web tree saved to '{html_path}'.")
    webbrowser.open('file://' + os.path.realpath(html_path))
    
    # 2. Provide the option to physically export folders
    print("\nThe model can automatically organize your files based on this structural hierarchy.")
    export_sys = input("Do you want to save this as a physical folder system? (y/n): ").strip().lower() == 'y'
    
    if export_sys:
        num_clusters = input("How many main categorical folders do you want? [Default: 5]: ").strip()
        num_clusters = int(num_clusters) if num_clusters.isdigit() else 5
        
        output_dir = os.path.join(folder_path, "Meridian_Clusters")
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine flat clusters based on max requested folders
        cluster_assignments = fcluster(Z, t=num_clusters, criterion='maxclust')
        
        print(f"\nMoving files into {num_clusters} semantic clusters...")
        for idx, (path, cluster_id) in enumerate(zip(sorted(img_paths), cluster_assignments)):
            cluster_folder = os.path.join(output_dir, f"Cluster_{cluster_id}")
            os.makedirs(cluster_folder, exist_ok=True)
            shutil.copy2(path, os.path.join(cluster_folder, os.path.basename(path)))
            
        print(f"Done! Files organized inside: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Persistent Interactive Visual Hub for Meridian.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint weights.")
    parser.add_argument("--index", type=str, required=True, help="Path to index cache file.")
    parser.add_argument("--topk", type=int, default=9, help="Number of neighbors to return.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing environment components...")
    model = load_meridian_model(args.checkpoint, device)
    tokenizer = Tokenizer()
    engine = MeridianSearchEngine(args.index)

    print("\nInitialization Complete. Entering persistent session loop.")
    while True:
        print("\nSelect a Visual Verification Sub-mode:")
        print("  1. Text Query Search (comma-separated lists supported)")
        print("  2. Image Query Search")
        print("  3. Multimodal Joint Search")
        print("  4. Semantic Folder Organization & Interactive Tree")
        print("  5. Exit")
        
        mode = input("\nEnter choice index [1-5]: ").strip()
        if mode == "5" or mode.lower() == "exit":
            print("Closing session.")
            break
            
        if mode == "1":
            text_in = input("Enter search phrase string: ").strip()
            if text_in: run_search_query(model, tokenizer, engine, text_in, None, args.topk, device)
                
        elif mode == "2":
            img_path = input("Enter path to query image: ").strip()
            if img_path: run_search_query(model, tokenizer, engine, None, img_path, args.topk, device)
                
        elif mode == "3":
            text_in = input("Enter search phrase string: ").strip()
            img_path = input("Enter path to query image: ").strip()
            run_search_query(model, tokenizer, engine, text_in, img_path, args.topk, device)
            
        elif mode == "4":
            dir_path = input("Enter local folder path containing target image files: ").strip()
            if dir_path: run_hierarchy_folder(model, tokenizer, dir_path, device)
        else:
            print("Invalid entry selection.")

if __name__ == "__main__":
    main()