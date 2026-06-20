import io
import os
from pathlib import Path

# REDIRECT HUGGINGFACE CACHE TO LOCAL PROJECT FOLDER
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_HF_CACHE = PROJECT_ROOT / "checkpoints" / "hf_cache"
LOCAL_HF_CACHE.mkdir(parents=True, exist_ok=True)

# Set the environment variable before importing any transformers/hf tools
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)


import io
import os
import json
import logging
import asyncio
import hashlib
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional, List
import uuid
import time
import numpy as np
import torch

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from PIL import Image, UnidentifiedImageError
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, to_tree

from transformers import AutoModel

from meridian.tokenizer import Tokenizer
from meridian.lorentz import lorentz_distance
from meridian.data.transforms import build_eval_transform
from api.inference import MeridianSearchEngine
from api.timing import timer, log_encode, log_request, log_hierarchy



# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("meridian.api")



# Config
CACHE_DIR  = os.getenv("MERIDIAN_CACHE_DIR",  "app_cache")
# Create cache dir at module load time so StaticFiles() below never crashes
# even if the folder was deleted between runs.
os.makedirs(CACHE_DIR, exist_ok=True)



# Pydantic schemas
class QueryConfig(BaseModel):
    has_text: bool
    has_image: bool
    topk: int

class MatchItem(BaseModel):
    id: int
    score: float
    caption: str
    url: str

class SearchResponse(BaseModel):
    query_configuration: QueryConfig
    matches: List[MatchItem]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    device: str
    node_count: Optional[int] = None



# Service: model inference

class InferenceService:
    """Owns the MeridianModel and tokenizer; executes forward passes."""

    def __init__(self, device: torch.device):
            self.device = device
            self.tokenizer = Tokenizer()
            self.model = self._load_model()


    def _load_model(self) -> AutoModel:
        # Hugging Face handles fetching the code, inferring dims from config.json,
        # and loading the safetensors.
        model = AutoModel.from_pretrained("kaustuk000/meridian", trust_remote_code=True)
        model.to(self.device)
        model.eval()
        log.info("Hugging Face model loaded and set to eval mode.")
        return model

        def _infer_dim(prefix: str) -> int:
            for k, v in state_dict.items():
                if (k.startswith(prefix) and k.endswith(".weight")
                        and v.ndim == 2 and v.shape[1] == 128):
                    return int(v.shape[0])
            raise RuntimeError(
                f"Cannot infer output dim for prefix {prefix!r}. "
                "Verify checkpoint structure."
            )

        model = MeridianModel(
            image_hout=_infer_dim("hyp_image_head.image_mlp."),
            image_eout=_infer_dim("eucl_image_head.image_mlp."),
            text_hout =_infer_dim("hyp_text_head.text_mlp."),
            text_eout =_infer_dim("eucl_text_head.text_mlp."),
        ).to(self.device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("Missing keys (%d): %s …", len(missing), missing[:4])
        if unexpected:
            log.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:4])

        model.eval()
        log.info("Model loaded and set to eval mode.")
        return model

    def _tokenize(self, text: str):
        tokens  = self.tokenizer([text])[0]
        max_len = 77
        input_ids      = torch.zeros(1, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(1, max_len, dtype=torch.long, device=self.device)
        seq = tokens[:max_len].long().to(self.device)
        n   = len(seq)
        input_ids[0, :n]      = seq
        attention_mask[0, :n] = 1
        eos_indices = torch.tensor([n - 1], dtype=torch.long, device=self.device)
        return input_ids, attention_mask, eos_indices

    # Public

    def run(self, pixel_values: torch.Tensor, text: str) -> dict:
        input_ids, attention_mask, eos_indices = self._tokenize(text)
        try:
            with torch.no_grad():
                with timer() as t_enc:
                    img_out = self.model.encode_image(pixel_values)
                    txt_out = self.model.encode_text(input_ids, attention_mask, eos_indices)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(503, "GPU memory exhausted — retry after a moment.")

        log_encode(t_enc["ms"])

        out = {}
        out["h_image"]    = img_out["h_image"].to(self.device)
        out["e_image"]    = img_out["e_image"].to(self.device)
        out["a_img"]      = img_out["a_img"].to(self.device)
        out["b_img"]      = img_out["b_img"].to(self.device)
        out["h_text"]     = txt_out["h_text"].to(self.device)
        out["e_text"]     = txt_out["e_text"].to(self.device)
        out["a_txt"]      = txt_out["a_txt"].to(self.device)
        out["b_txt"]      = txt_out["b_txt"].to(self.device)
        out["curv"]       = img_out["curv"]
        out["scale_hyp"]  = img_out["scale_hyp"]
        out["scale_eucl"] = img_out["scale_eucl"]
        return out

    def dummy_pixels(self) -> torch.Tensor:
        return torch.zeros(1, 3, 224, 224, device=self.device)



# Service: image caching

class CacheService:
    """Downloads and persists images; returns /cache/<filename> URLs.

    On download failure returns a placeholder so the search result still
    shows (with caption) rather than being silently dropped.
    """

    _PLACEHOLDER_NAME = "_placeholder.jpg"

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._placeholder_path = os.path.join(cache_dir, self._PLACEHOLDER_NAME)
        self._ensure_placeholder()

    # Internal 

    def _ensure_placeholder(self) -> None:
        """Write a dark grey placeholder JPEG once."""
        if not os.path.exists(self._placeholder_path):
            img = Image.new("RGB", (400, 300), (28, 28, 36))
            img.save(self._placeholder_path, format="JPEG", quality=100)
            log.info("Placeholder created → %s", self._placeholder_path)

    def _content_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:14]

    @staticmethod  
    def _download_sync(url: str, filepath: str) -> bool:  
        """Download one URL, convert to JPEG, and persist to filepath.

        Returns True only if a valid image was saved.
        Uses load() instead of verify() — verify() is stricter than browsers
        and incorrectly rejects many valid JPEGs with minor quirks.
        Retries once on transient failures.
        """
        if os.path.exists(filepath):
            return True

        for attempt in range(1, 3):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 Meridian/2.0"}
                )
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    ct = resp.headers.get("Content-Type", "")
                    
                    if any(t in ct for t in ("text/html", "text/xml", "application/xml")):
                        log.warning(
                            "Non-image Content-Type %r (attempt %d): %s", ct, attempt, url
                        )
                        continue 

                    data = resp.read()

                if not data:
                    log.warning("Empty response (attempt %d): %s", attempt, url)
                    continue

                try:
                    with Image.open(io.BytesIO(data)) as img:
                        img.load()
                        if img.width < 64 or img.height < 64:
                            log.warning(
                                "Image too small (%dx%d), skipping: %s",
                                img.width, img.height, url,
                            )
                            return False
                        img_rgb = img.convert("RGB")
                    img_rgb.save(filepath, format="JPEG", quality=85, optimize=True)
                except Exception as img_exc:
                    log.warning(
                        "PIL decode/save failed (attempt %d) [%s]: %s",
                        attempt, url, img_exc,
                    )
                    # Clean up any partial write before retrying
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass
                    continue

                log.info( 
                    "Cached %s ← %s (raw %d B)",
                    os.path.basename(filepath), url[:72], len(data),
                )
                return True

            except Exception as exc:
                log.warning("Download attempt %d/2 failed [%s]: %s", attempt, url, exc)

        log.warning("All download attempts failed, using placeholder: %s", url)
        return False

    # Public 

    async def cache_url(self, url: str, idx: int) -> str:
        """Return a local /cache/ path — real image or placeholder, never None."""
        if not url:
            return f"/cache/{self._PLACEHOLDER_NAME}"

        h        = self._content_hash(url)
        filepath = os.path.join(self.cache_dir, f"img_{idx}_{h}.jpg")
        success  = await asyncio.to_thread(self._download_sync, url, filepath)

        if success:
            return f"/cache/{os.path.basename(filepath)}"
        return f"/cache/{self._PLACEHOLDER_NAME}"

    async def cache_upload(self, raw: bytes, stem: str, idx: int) -> Optional[str]:
        """Persist an uploaded image; returns None only if raw bytes are unreadable."""
        if not raw:
            return None
        try:
            with Image.open(io.BytesIO(raw)) as img:
                img.load()
                img_rgb = img.convert("RGB")
        except Exception as exc:
            log.warning("Upload decode failed [%s_%d]: %s", stem, idx, exc)
            return None

        safe_stem = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in (stem or "img")
        )[:40] or "img"
        filename = f"{safe_stem}_{idx}_{uuid.uuid4().hex[:10]}.jpg"
        filepath = os.path.join(self.cache_dir, filename)
        try:
            img_rgb.save(filepath, format="JPEG", quality=95, optimize=True)
            return f"/cache/{filename}"
        except Exception as exc:
            log.warning("Failed to save upload [%s]: %s", filename, exc)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            return None


# Service: hierarchy tree building
class HierarchyService:
    """Computes Lorentz distance matrices, runs linkage, renders 3-D HTML."""

    def __init__(self, cache_dir: str, device: torch.device):
        self.cache_dir = cache_dir
        self.device    = device

    def _dist_matrix(
        self, features: list[torch.Tensor], curv: torch.Tensor
    ) -> np.ndarray:
        """Pairwise Lorentz distances — raw positive distances for linkage()."""
        n   = len(features)
        mat = np.zeros((n, n), dtype=np.float64)
        h   = torch.stack(features)
        for i in range(n):
            for j in range(i + 1, n):
                d = lorentz_distance(
                    h[i].unsqueeze(0), h[j].unsqueeze(0), curv=curv
                ).item()
                mat[i, j] = mat[j, i] = max(0.0, d)
        return mat

    @staticmethod
    def _build_tree_json(node, leaf_items: list) -> dict:
        if node.is_leaf():
            item = leaf_items[node.id]
            if isinstance(item, dict):
                payload = {"name": str(item.get("name", ""))}
                if image_url := item.get("image_url"):
                    payload["image_url"] = image_url
                return payload
            return {"name": str(item)}
        return {
            "name": f"Node_{node.id}",
            "children": [
                HierarchyService._build_tree_json(node.left,  leaf_items),
                HierarchyService._build_tree_json(node.right, leaf_items),
            ],
        }

    def _write_html(self, tree_dict: dict, out_path: str) -> None:
        html = _HTML_TEMPLATE.replace(
            "__TREE_DATA__", json.dumps(tree_dict, ensure_ascii=False)
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        log.info("3-D tree written → %s (%d bytes)", out_path, len(html))

    def build_response(
        self,
        features: list[torch.Tensor],
        labels: list,
        curv: torch.Tensor,
        request: Request,
        filename: str,
    ) -> dict:
        with timer() as t_dist:
            mat = self._dist_matrix(features, curv)

        with timer() as t_link:
            condensed = ssd.squareform(mat)
            Z         = linkage(condensed, method="average")
            root, _   = to_tree(Z, rd=True)
            tree_dict = self._build_tree_json(root, labels)

        with timer() as t_render:
            out_path = os.path.join(self.cache_dir, filename)
            self._write_html(tree_dict, out_path)

        log_hierarchy(
            n_items=len(features),
            dist_ms=t_dist["ms"],
            linkage_ms=t_link["ms"],
            render_ms=t_render["ms"],
        )

        base = str(request.base_url).rstrip("/")
        return {
            "status":   "success",
            "message":  "Open the URL below to explore your 3-D semantic tree.",
            "tree_url": f"{base}/cache/{filename}",
        }



# Global service instances (populated in lifespan)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_svc:  Optional[InferenceService]     = None
search_engine:  Optional[MeridianSearchEngine] = None
cache_svc:      Optional[CacheService]         = None
hierarchy_svc:  Optional[HierarchyService]     = None


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_svc, search_engine, cache_svc, hierarchy_svc

    log.info("Meridian API starting on device: %s", device)

    try:
        # 1. Initialize InferenceService (Downloads HF Model automatically)
        inference_svc = InferenceService(device)
        
        # 2. Download/Load HF Index directly through the loaded model
        log.info("Downloading/Loading Hugging Face index...")
        hf_index = inference_svc.model.load_index("kaustuk000/meridian")
        
        # 3. Pass the fetched HF index dictionary to the Search Engine
        search_engine = MeridianSearchEngine(hf_index, device)
        
        # 4. Initialize cache and hierarchy services
        cache_svc     = CacheService(CACHE_DIR)
        hierarchy_svc = HierarchyService(CACHE_DIR, device)

        log.info(
            "Engine ready. Index size: %d items.",
            len(search_engine.index.get("captions", [])),
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            log.info(
                "CUDA memory: %.1f MB allocated.",
                torch.cuda.memory_allocated() / 1e6,
            )
    except Exception:
        log.exception("Fatal error during startup.")
        raise

    yield

    log.info("Shutting down Meridian API.")
    if device.type == "cuda":
        torch.cuda.empty_cache()


# App
app = FastAPI(
    title="Meridian Multi-Space Search API",
    description=(
        "Joint Hyperbolic-Euclidean image-text retrieval with interactive "
        "3-D semantic hierarchy trees."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/cache", StaticFiles(directory=CACHE_DIR), name="cache")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Global exception handler
@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception):
    log.error(
        "Unhandled exception [%s %s]: %s", request.method, request.url, exc, exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error — check server logs."},
    )



# Shared helpers (stateless, no class needed)

def _fetch_pil_from_url(url: str) -> Image.Image:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 Meridian/2.0"}
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    except UnidentifiedImageError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to fetch image from URL: {exc}") from exc


def _safe_index_get(collection: list, idx: int, fallback=""):
    if not collection or idx < 0 or idx >= len(collection):
        return fallback
    return collection[idx]


# Routes

@app.get("/")
def root():
    return {
        "service": "Meridian Multi-Space Search API v2",
        "docs":    "/docs",
        "health":  "/healthz",
    }


@app.get("/healthz", response_model=HealthResponse)
def health():
    return HealthResponse(
        status      = "ok" if inference_svc is not None and search_engine is not None else "degraded",
        model_loaded= inference_svc is not None,
        index_loaded= search_engine is not None,
        device      = str(device),
        node_count  = len(search_engine.index.get("captions", [])) if search_engine else None,
    )


@app.post("/search", response_model=SearchResponse)
async def search(
    text_query: Optional[str]        = Form(None),
    image_file: Optional[UploadFile] = File(None),
    image_url:  Optional[HttpUrl]    = Form(None),
    topk:       int                  = Form(9, ge=1, le=50),
):
    has_text  = bool(text_query and text_query.strip())
    has_image = bool(image_file or image_url)

    if not has_text and not has_image:
        raise HTTPException(400, "Provide at least one modality (text or image).")
    if image_file and image_url:
        raise HTTPException(400, "Provide image_file or image_url — not both.")

    # Build pixel tensor 
    if has_image:
        try:
            if image_file:
                raw     = await image_file.read()
                img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
            else:
                img_pil = await asyncio.to_thread(_fetch_pil_from_url, str(image_url))
            pixel_values = build_eval_transform()(img_pil).unsqueeze(0).to(device)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except UnidentifiedImageError:
            raise HTTPException(415, "File could not be decoded as an image.")
    else:
        pixel_values = inference_svc.dummy_pixels()

    # Forward pass
    with timer() as t_enc:
        query_text = text_query if has_text else "a photo"
        outputs    = inference_svc.run(pixel_values, query_text)

    # Search 
    with timer() as t_search:
        try:
            candidate_pool = min(topk * 2, 200)
            results = search_engine.search(
                outputs,
                query_has_text=has_text,
                query_has_image=has_image,
                topk=candidate_pool,
            )
        except Exception as exc:
            log.exception("Search engine error")
            raise HTTPException(500, f"Search engine error: {exc}")

    # Cache result images concurrently 
    with timer() as t_cache:
        captions    = search_engine.index.get("captions", [])
        raw_urls    = search_engine.index.get("urls",     [])
        dl_tasks    = [cache_svc.cache_url(_safe_index_get(raw_urls, m["id"]), m["id"]) for m in results]
        local_paths = await asyncio.gather(*dl_tasks)

    log_request(
        route="/search",
        encode_ms=t_enc["ms"],
        search_ms=t_search["ms"],
        cache_ms=t_cache["ms"],
        total_ms=t_enc["ms"] + t_search["ms"] + t_cache["ms"],
    )

    matches: list[MatchItem] = []
    for match, lp in zip(results, local_paths):
        matches.append(MatchItem(
            id      = match["id"],
            score   = float(match["score"]),
            caption = _safe_index_get(captions, match["id"], fallback=""),
            url     = lp,
        ))
        if len(matches) == topk:
            break

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return SearchResponse(
        query_configuration=QueryConfig(has_text=has_text, has_image=has_image, topk=topk),
        matches=matches,
    )


@app.post("/hierarchy/text")
async def hierarchy_text(
    request: Request,
    terms:   str = Form(..., description="Comma-separated list of concept terms (≥ 2)."),
):
    term_list = [t.strip() for t in terms.split(",") if t.strip()]
    if len(term_list) < 2:
        raise HTTPException(400, "Provide at least 2 comma-separated terms.")
    if len(term_list) > 2000:
        raise HTTPException(400, "Maximum 2000 terms per request.")

    dummy_px     = inference_svc.dummy_pixels()
    features:    list[torch.Tensor] = []
    valid_terms: list[str]          = []
    failed:      list[str]          = []
    curv:        Optional[torch.Tensor] = None

    for term in term_list:
        try:
            out = inference_svc.run(dummy_px, term)
            features.append(out["h_text"].squeeze(0))
            valid_terms.append(term)
            if curv is None:
                curv = out["curv"]
        except HTTPException:
            raise
        except Exception as exc:
            log.warning("Failed to encode term %r: %s", term, exc)
            failed.append(term)

    if len(features) < 2:
        raise HTTPException(422, f"Could not encode enough terms. Failed: {failed}")

    resp = hierarchy_svc.build_response(
        features, valid_terms, curv,
        request, f"tree_text_{uuid.uuid4().hex}.html",
    )
    if failed:
        resp["skipped_terms"] = failed
    return resp


@app.post("/hierarchy/images")
async def hierarchy_images(
    request:      Request,
    files:        List[UploadFile] = File(...),
    use_captions: bool             = Form(False),
):
    if len(files) < 2:
        raise HTTPException(400, "Upload at least 2 images.")
    if len(files) > 1000:
        raise HTTPException(400, "Maximum 1000 images per request.")

    features: list[torch.Tensor] = []
    payloads: list[dict]         = []
    curv:     Optional[torch.Tensor] = None

    for idx, file in enumerate(files):
        try:
            raw     = await file.read()
            img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
            pv      = build_eval_transform()(img_pil).unsqueeze(0).to(device)
            out     = inference_svc.run(pv, "a photo")
            features.append(out["h_image"].squeeze(0))
            if curv is None:
                curv = out["curv"]

            stem      = os.path.splitext(file.filename or f"img_{idx}")[0]
            image_url = await cache_svc.cache_upload(raw, stem, idx)
            payloads.append({"name": stem, "image_url": image_url})
        except HTTPException:
            raise
        except Exception as exc:
            log.warning("Could not process %r: %s", file.filename, exc)

    if len(features) < 2:
        raise HTTPException(422, "Could not process enough valid images (need ≥ 2).")

    return hierarchy_svc.build_response(
        features, payloads, curv,
        request, f"tree_images_{uuid.uuid4().hex}.html",
    )



# 3-D Hierarchy HTML template

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meridian — 3D Semantic Tree</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#030712; overflow:hidden; font-family:system-ui,-apple-system,'Segoe UI',sans-serif; color:#e2e8f0; user-select:none; -webkit-user-select:none; }
canvas { display:block; }
#ui { position:fixed; inset:0; pointer-events:none; z-index:10; }
#search-wrap { position:absolute; top:16px; left:16px; pointer-events:all; }
#search {
  background:rgba(2,6,23,0.88); border:1px solid rgba(99,102,241,0.32);
  color:#e2e8f0; padding:10px 14px 10px 38px; border-radius:12px;
  font-size:14px; width:260px; backdrop-filter:blur(12px); outline:none;
  transition:border-color 0.2s, box-shadow 0.2s;
}
#search:focus { border-color:rgba(129,140,248,0.7); box-shadow:0 0 0 3px rgba(99,102,241,0.12); }
#search::placeholder { color:#4b5563; }
#search-icon { position:absolute; left:12px; top:50%; transform:translateY(-50%); font-size:14px; color:#4b5563; pointer-events:none; }
#btn-group { position:absolute; top:16px; right:16px; display:flex; flex-direction:column; gap:8px; pointer-events:all; }
.btn {
  background:rgba(2,6,23,0.88); border:1px solid rgba(99,102,241,0.28);
  color:#94a3b8; padding:9px 14px 9px 11px; border-radius:10px;
  font-size:12.5px; cursor:pointer; backdrop-filter:blur(12px);
  transition:all 0.18s; letter-spacing:0.025em; white-space:nowrap; text-align:left;
}
.btn:hover { background:rgba(99,102,241,0.22); border-color:rgba(129,140,248,0.55); color:#e2e8f0; }
.btn:active { transform:scale(0.97); }
.btn span.icon { display:inline-block; width:16px; text-align:center; margin-right:4px; opacity:0.75; }
#tooltip {
  position:absolute; display:none; background:rgba(2,6,23,0.94);
  border:1px solid rgba(99,102,241,0.38); border-radius:14px;
  padding:14px 17px; font-size:12.5px; pointer-events:none;
  backdrop-filter:blur(16px); max-width:240px;
  box-shadow:0 16px 48px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.04);
}
#tt-name { font-weight:700; color:#f1f5f9; margin-bottom:6px; font-size:14px; }
#tt-info { color:#94a3b8; line-height:1.7; font-size:12.5px; }
.tt-badge { display:inline-block; padding:2px 7px; border-radius:5px; font-size:10px; font-weight:600; margin-top:4px; letter-spacing:0.04em; }
#stats { position:absolute; bottom:52px; left:16px; font-size:12px; color:#374151; line-height:2.0; }
#stats span { color:#6366f1; font-weight:600; }
#hint { position:absolute; bottom:16px; right:16px; font-size:11.5px; color:#1f2937; text-align:right; line-height:1.95; }
#hint b { color:#374151; }
#legend { position:absolute; bottom:16px; left:16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.leg { display:flex; align-items:center; gap:5px; font-size:11px; color:#374151; }
.leg-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
#loader {
  position:fixed; inset:0; background:#030712;
  display:flex; align-items:center; justify-content:center;
  z-index:100; flex-direction:column; gap:12px;
}
#loader p { font-size:12.5px; color:#374151; letter-spacing:0.08em; }
.spinner { width:32px; height:32px; border:2px solid rgba(99,102,241,0.15); border-top-color:#6366f1; border-radius:50%; animation:spin 0.7s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<div id="loader"><div class="spinner"></div><p>BUILDING SCENE</p></div>
<canvas id="c"></canvas>
<div id="ui">
  <div id="search-wrap">
    <span id="search-icon">&#8981;</span>
    <input id="search" type="text" placeholder="Search nodes&#8230;" autocomplete="off">
  </div>
  <div id="btn-group">
    <button class="btn" id="btn-fit"><span class="icon">&#8993;</span>Fit to View</button>
    <button class="btn" id="btn-expand"><span class="icon">&#8862;</span>Expand All</button>
    <button class="btn" id="btn-collapse"><span class="icon">&#8863;</span>Collapse to Root</button>
    <button class="btn" id="btn-reset"><span class="icon">&#8635;</span>Reset Camera</button>
  </div>
  <div id="tooltip"><div id="tt-name"></div><div id="tt-info"></div></div>
  <div id="stats">Visible: <span id="sv">&#8212;</span>&thinsp;/&thinsp;<span id="st">&#8212;</span> nodes<br>Depth: <span id="sd">&#8212;</span></div>
  <div id="hint"><b>Click</b> expand / collapse &nbsp;&middot;&nbsp; <b>Dbl-click</b> zoom<br><b>Drag</b> rotate &nbsp;&middot;&nbsp; <b>Scroll</b> zoom &nbsp;&middot;&nbsp; <b>Right-drag</b> pan</div>
  <div id="legend"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
'use strict';

// ── Inject server data & normalise to {id, name, children, collapsed} ──────
const _RAW = __TREE_DATA__;
(function prep(n) {
  n.id = Math.random().toString(36).slice(2, 10);
  n.collapsed = false;
  if (n.children) n.children.forEach(prep);
  else n.children = [];
})(_RAW);
const ROOT = _RAW;

// ── Renderer ─────────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const W = () => window.innerWidth, H = () => window.innerHeight;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(W(), H());
renderer.setClearColor(0x030712, 1);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x030712, 0.00028);
const camera = new THREE.PerspectiveCamera(55, W() / H(), 0.5, 8000);

// ── Lighting ──────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0x111827, 3));
const keyLight = new THREE.DirectionalLight(0x7799dd, 1.3);
keyLight.position.set(300, 500, 300); scene.add(keyLight);
const fillA = new THREE.PointLight(0x3344ff, 3, 1400); fillA.position.set(-350, 100, 350); scene.add(fillA);
const fillB = new THREE.PointLight(0xff2255, 2, 1100); fillB.position.set(350, -350, -250); scene.add(fillB);
const fillC = new THREE.PointLight(0x22ffaa, 1.5, 900); fillC.position.set(0, -350, 400); scene.add(fillC);

// ── Palette ───────────────────────────────────────────────────────────────────
const PALETTE = [
  { hex: 0xff6b6b, css: '#ff6b6b', name: 'Root'  },
  { hex: 0xffd93d, css: '#ffd93d', name: 'Lvl 1' },
  { hex: 0x6bcb77, css: '#6bcb77', name: 'Lvl 2' },
  { hex: 0x4d96ff, css: '#4d96ff', name: 'Lvl 3' },
  { hex: 0xc77dff, css: '#c77dff', name: 'Lvl 4' },
  { hex: 0xff9f43, css: '#ff9f43', name: 'Lvl 5' },
  { hex: 0x48dbfb, css: '#48dbfb', name: 'Lvl 6' },
];
const col = d => PALETTE[Math.min(d, PALETTE.length - 1)];
document.getElementById('legend').innerHTML = PALETTE.map(p =>
  `<span class="leg"><span class="leg-dot" style="background:${p.css}"></span>${p.name}</span>`
).join('');

// ── Label texture ─────────────────────────────────────────────────────────────
function makeLabel(text, hexColor, childCount, collapsed) {
  const display = (collapsed && childCount > 0) ? ('\\u25b6  ' + text + '  (' + childCount + ')') : text;
  const cv = document.createElement('canvas'); cv.width = 380; cv.height = 86;
  const ctx = cv.getContext('2d');
  ctx.font = 'bold 18px system-ui,sans-serif';
  const tw = Math.min(ctx.measureText(display).width, 330);
  const bw = tw + 44, bh = 48, bx = (380 - bw) / 2, by = 18, r = 11;
  const R8 = (hexColor >> 16) & 0xff, G8 = (hexColor >> 8) & 0xff, B8 = hexColor & 0xff;
  ctx.fillStyle = 'rgba(3,7,18,0.9)';
  roundRect(ctx, bx, by, bw, bh, r); ctx.fill();
  ctx.strokeStyle = `rgba(${R8},${G8},${B8},0.7)`; ctx.lineWidth = 1.5;
  roundRect(ctx, bx, by, bw, bh, r); ctx.stroke();
  ctx.fillStyle = '#f1f5f9'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(display, 190, by + bh / 2, 330);
  return new THREE.CanvasTexture(cv);
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r); ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h); ctx.arcTo(x,y+h,x,y+h-r,r);
  ctx.lineTo(x,y+r); ctx.arcTo(x,y,x+r,y,r); ctx.closePath();
}

// ── Tree stats ────────────────────────────────────────────────────────────────
function leafCount(n)    { return (!n.children.length || n.collapsed) ? 1 : n.children.reduce((s,c)=>s+leafCount(c),0); }
function totalCount(n)   { return 1 + n.children.reduce((s,c)=>s+totalCount(c),0); }
function visibleCount(n) { return (!n.children.length || n.collapsed) ? 1 : 1+n.children.reduce((s,c)=>s+visibleCount(c),0); }
function treeDepth(n,d)  { d=d||0; if(!n.children.length) return d; return Math.max.apply(null,n.children.map(c=>treeDepth(c,d+1))); }

// ── Layout ────────────────────────────────────────────────────────────────────
const Y_STEP = 140;
const R_BASE = [220, 180, 140, 110, 90, 80];
const rFor   = d => R_BASE[Math.min(d, R_BASE.length-1)];

function layoutTree(node, pos, depth, angleCenter, angleBudget) {
  node._pos = pos.clone();
  if (node.collapsed || !node.children.length) return;
  const totalLeafs = node.children.reduce((s,c)=>s+leafCount(c),0);
  const r = rFor(depth), budget = Math.min(angleBudget * 0.93, Math.PI * 2);
  let a = angleCenter - budget / 2;
  node.children.forEach(child => {
    const frac = leafCount(child) / totalLeafs;
    const cb = budget * frac, ca = a + cb / 2;
    layoutTree(child, new THREE.Vector3(pos.x + r*Math.cos(ca), pos.y - Y_STEP, pos.z + r*Math.sin(ca)), depth+1, ca, cb);
    a += cb;
  });
}

// ── Scene graph ───────────────────────────────────────────────────────────────
const nodeMap = new Map();
const edgeGroup = new THREE.Group(); scene.add(edgeGroup);
const GEO = [
  new THREE.SphereGeometry(14,28,28), new THREE.SphereGeometry(10,24,24),
  new THREE.SphereGeometry(7,20,20),  new THREE.SphereGeometry(5,16,16),
];
const geoFor = d => GEO[Math.min(d, GEO.length-1)];

function bezierPoints(a, b, steps) {
  steps = steps || 14;
  const mid = new THREE.Vector3().lerpVectors(a,b,0.5); mid.y += (a.y-b.y)*0.1;
  const pts = [];
  for(let i=0;i<=steps;i++){const t=i/steps,mt=1-t;pts.push(new THREE.Vector3(mt*mt*a.x+2*mt*t*mid.x+t*t*b.x,mt*mt*a.y+2*mt*t*mid.y+t*t*b.y,mt*mt*a.z+2*mt*t*mid.z+t*t*b.z));}
  return pts;
}

function buildScene() {
  nodeMap.forEach(({mesh}) => scene.remove(mesh)); nodeMap.clear(); edgeGroup.clear();
  layoutTree(ROOT, new THREE.Vector3(0,0,0), 0, 0, Math.PI*2);
  function buildNode(node, parentPos, depth) {
    const c = col(depth), geo = geoFor(depth), rr = geo.parameters.radius;
    const mat = new THREE.MeshPhongMaterial({color:c.hex,emissive:c.hex,emissiveIntensity:0.22,shininess:95,transparent:true,opacity:1.0});
    const mesh = new THREE.Mesh(geo, mat); mesh.position.copy(node._pos); scene.add(mesh);
    const ring = new THREE.Mesh(new THREE.RingGeometry(rr*1.65,rr*2.05,40),
      new THREE.MeshBasicMaterial({color:c.hex,transparent:true,opacity:0.1,side:THREE.DoubleSide,depthWrite:false}));
    ring.rotation.x = -Math.PI/2; mesh.add(ring);
    const spriteMat = new THREE.SpriteMaterial({map:makeLabel(node.name,c.hex,node.children.length,node.collapsed),transparent:true,depthWrite:false});
    const sprite = new THREE.Sprite(spriteMat);
    const sw = depth===0?170:118; sprite.scale.set(sw,sw*86/380,1);
    sprite.position.set(0, rr+(depth===0?32:20), 0); mesh.add(sprite);
    mesh.userData.nodeId = node.id;
    nodeMap.set(node.id, {mesh,ring,sprite,node,depth,c});
    if (parentPos) {
      const pts = bezierPoints(parentPos, node._pos);
      edgeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({color:c.hex,transparent:true,opacity:0.32})));
    }
    if (!node.collapsed) node.children.forEach(ch => buildNode(ch, node._pos, depth+1));
  }
  buildNode(ROOT, null, 0);
  document.getElementById('sv').textContent = visibleCount(ROOT);
  document.getElementById('st').textContent  = totalCount(ROOT);
  document.getElementById('sd').textContent  = treeDepth(ROOT);
}

// ── Starfield ─────────────────────────────────────────────────────────────────
{
  const geo = new THREE.BufferGeometry();
  const buf = new Float32Array(6000*3); for(let i=0;i<buf.length;i++) buf[i]=(Math.random()-0.5)*6000;
  geo.setAttribute('position',new THREE.BufferAttribute(buf,3));
  scene.add(new THREE.Points(geo,new THREE.PointsMaterial({color:0x1e2d45,size:1.3,transparent:true,opacity:0.7})));
}

// ── Auto-collapse nodes beyond depth 2 for large trees ───────────────────────
(function ac(n,d){if(d>=3&&n.children.length)n.collapsed=true;else n.children.forEach(c=>ac(c,d+1));})(ROOT,0);

// ── Camera (custom spherical, no OrbitControls) ───────────────────────────────
const sph={theta:0.4,phi:1.05,r:900}, tsph={...sph};
const tgt=new THREE.Vector3(0,-60,0), ttgt=new THREE.Vector3(0,-60,0);
let dragMode=0,lastMX=0,lastMY=0,dragDist=0;
function applyCamera(){
  sph.theta+=(tsph.theta-sph.theta)*0.08; sph.phi+=(tsph.phi-sph.phi)*0.08; sph.r+=(tsph.r-sph.r)*0.08; tgt.lerp(ttgt,0.08);
  const sp=Math.sin(sph.phi);
  camera.position.set(tgt.x+sph.r*sp*Math.sin(sph.theta),tgt.y+sph.r*Math.cos(sph.phi),tgt.z+sph.r*sp*Math.cos(sph.theta));
  camera.lookAt(tgt);
}
canvas.addEventListener('mousedown',e=>{dragMode=e.button===2?2:1;lastMX=e.clientX;lastMY=e.clientY;dragDist=0;e.preventDefault();});
window.addEventListener('mouseup',()=>dragMode=0);
canvas.addEventListener('contextmenu',e=>e.preventDefault());
window.addEventListener('mousemove',e=>{
  const dx=e.clientX-lastMX,dy=e.clientY-lastMY;
  dragDist+=Math.abs(dx)+Math.abs(dy);
  if(dragMode===1){tsph.theta-=dx*0.005;tsph.phi=Math.max(0.12,Math.min(Math.PI-0.12,tsph.phi+dy*0.005));}
  if(dragMode===2){const right=new THREE.Vector3(Math.cos(sph.theta),0,-Math.sin(sph.theta)),sc=sph.r*0.0012;ttgt.addScaledVector(right,-dx*sc);ttgt.y+=dy*sc;}
  lastMX=e.clientX;lastMY=e.clientY;handleHover(e.clientX,e.clientY);
});
canvas.addEventListener('wheel',e=>{
  const rect = canvas.getBoundingClientRect();
  const nx = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
  const ny = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
  const nextR = Math.max(80, Math.min(4000, tsph.r * (1 + e.deltaY * 0.001)));
  const pan = (tsph.r - nextR) * 0.0025;
  const right = new THREE.Vector3(Math.cos(sph.theta), 0, -Math.sin(sph.theta));
  ttgt.addScaledVector(right, nx * pan);
  ttgt.y += -ny * pan * 0.7;
  tsph.r = nextR;
  e.preventDefault();
},{passive:false});
let touchPD=0;
canvas.addEventListener('touchstart',e=>{if(e.touches.length===1){dragMode=1;lastMX=e.touches[0].clientX;lastMY=e.touches[0].clientY;dragDist=0;}else if(e.touches.length===2){dragMode=0;touchPD=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);}e.preventDefault();},{passive:false});
canvas.addEventListener('touchmove',e=>{if(e.touches.length===1&&dragMode===1){const dx=e.touches[0].clientX-lastMX,dy=e.touches[0].clientY-lastMY;dragDist+=Math.abs(dx)+Math.abs(dy);tsph.theta-=dx*0.005;tsph.phi=Math.max(0.12,Math.min(Math.PI-0.12,tsph.phi+dy*0.005));lastMX=e.touches[0].clientX;lastMY=e.touches[0].clientY;}if(e.touches.length===2){const nd=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);if(touchPD>0)tsph.r=Math.max(80,Math.min(4000,tsph.r*(touchPD/nd)));touchPD=nd;}e.preventDefault();},{passive:false});
canvas.addEventListener('touchend',e=>{if(e.touches.length===0&&dragDist<10){const t=e.changedTouches[0];handleClick(t.clientX,t.clientY);}dragMode=0;});

// ── Raycasting & interaction ──────────────────────────────────────────────────
const raycaster=new THREE.Raycaster(),ndcM=new THREE.Vector2();
let hoveredId=null;
const tooltip=document.getElementById('tooltip');
function hitTest(cx,cy){ndcM.set((cx/W())*2-1,-(cy/H())*2+1);raycaster.setFromCamera(ndcM,camera);const hits=raycaster.intersectObjects([...nodeMap.values()].map(d=>d.mesh),false);return hits.length?nodeMap.get(hits[0].object.userData.nodeId)||null:null;}
function handleHover(cx,cy){
  if(dragMode)return;
  const hit=hitTest(cx,cy);
  if(hit){
    hoveredId=hit.node.id; canvas.style.cursor='pointer';
    const n=hit.node,sub=totalCount(n)-1;
    document.getElementById('tt-name').textContent=n.name;
    document.getElementById('tt-info').innerHTML=
      'Depth: '+hit.depth+' &nbsp;&middot;&nbsp; Children: '+n.children.length+'<br>Subtree: '+sub+' node'+(sub!==1?'s':'')+'<br>'+
      (n.children.length>0?(n.collapsed?'<span class="tt-badge" style="background:rgba(255,211,61,.14);color:#ffd93d">&#9654; collapsed &#8212; click to expand</span>':'<span class="tt-badge" style="background:rgba(107,203,119,.14);color:#6bcb77">&#9660; click to collapse</span>'):'<span class="tt-badge" style="background:rgba(99,102,241,.14);color:#818cf8">leaf node &#8212; click to search</span>');
    tooltip.style.display='block';
    tooltip.style.left=Math.min(cx+18,W()-260)+'px';
    tooltip.style.top=Math.max(cy-20,8)+'px';
  }else{hoveredId=null;canvas.style.cursor='default';tooltip.style.display='none';}
}
function handleClick(cx,cy){const hit=hitTest(cx,cy);if(!hit)return; if(hit.node.children.length>0){hit.node.collapsed=!hit.node.collapsed;buildScene();tooltip.style.display='none';} else {window.parent?.postMessage({ type:'meridian-tree-leaf', query: hit.node.name, node: hit.node.name, image_url: hit.node.image_url || null, depth: hit.depth, children: hit.node.children.length }, '*');}}
canvas.addEventListener('click',e=>{if(dragDist>6)return;handleClick(e.clientX,e.clientY);});
canvas.addEventListener('dblclick',e=>{const hit=hitTest(e.clientX,e.clientY);if(hit)zoomTo(hit.node._pos);});

// ── Camera helpers ────────────────────────────────────────────────────────────
function zoomTo(pos){const dir=new THREE.Vector3().subVectors(camera.position,pos).normalize(),cp=pos.clone().add(dir.multiplyScalar(260)),dx=cp.x-ttgt.x,dy=cp.y-ttgt.y,dz=cp.z-ttgt.z;tsph.r=Math.hypot(dx,dy,dz);tsph.theta=Math.atan2(dx,dz);tsph.phi=Math.atan2(Math.hypot(dx,dz),dy);ttgt.copy(pos);}
function fitView(){if(!nodeMap.size)return;const box=new THREE.Box3();nodeMap.forEach(({mesh})=>box.expandByPoint(mesh.position));const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),span=Math.max(size.x,size.y,size.z,300),fov=camera.fov*Math.PI/180,dist=Math.max((span/(2*Math.tan(fov/2)))*1.55,300);ttgt.copy(center);tsph.r=dist;tsph.phi=1.05;tsph.theta=0.4;}
function resetCamera(){ttgt.set(0,-60,0);tsph.r=900;tsph.theta=0.4;tsph.phi=1.05;}
document.getElementById('btn-fit').addEventListener('click',fitView);
document.getElementById('btn-reset').addEventListener('click',resetCamera);
document.getElementById('btn-expand').addEventListener('click',()=>{(function ex(n){n.collapsed=false;n.children.forEach(ex);})(ROOT);buildScene();setTimeout(fitView,60);});
document.getElementById('btn-collapse').addEventListener('click',()=>{ROOT.children.forEach(c=>{(function co(n){if(n.children.length)n.collapsed=true;n.children.forEach(co);})(c);});buildScene();resetCamera();});

// ── Search ────────────────────────────────────────────────────────────────────
document.getElementById('search').addEventListener('input',e=>{
  const q=e.target.value.trim().toLowerCase();
  nodeMap.forEach(({mesh,sprite,node})=>{const m=!q||node.name.toLowerCase().includes(q);mesh.material.opacity=m?1:0.07;mesh.material.emissiveIntensity=m?0.25:0.02;sprite.material.opacity=m?1:0.05;});
  edgeGroup.children.forEach(e=>{e.material.opacity=q?0.08:0.32;});
});

// ── Animation loop ────────────────────────────────────────────────────────────
const _hS=new THREE.Vector3(1.22,1.22,1.22),_nS=new THREE.Vector3(1,1,1);
let tick=0;
function animate(){
  requestAnimationFrame(animate); tick+=0.016; applyCamera();
  nodeMap.forEach(({mesh,ring,node},id)=>{
    const hov=id===hoveredId;
    mesh.material.emissiveIntensity+=(hov?0.75:0.22-mesh.material.emissiveIntensity)*0.1;
    mesh.scale.lerp(hov?_hS:_nS,0.12);
    const pulse=1+Math.sin(tick*2.2+mesh.position.x*0.004)*0.07;
    ring.scale.set(pulse,pulse,pulse);
    ring.material.opacity=0.07+Math.sin(tick*1.7+mesh.position.z*0.005)*0.04;
    if(node.collapsed&&node.children.length>0)mesh.position.y=node._pos.y+Math.sin(tick*2.8+node._pos.x*0.015)*3.5;
    else mesh.position.y=node._pos.y;
  });
  fillA.position.x=-350+Math.sin(tick*0.28)*60;
  fillB.position.z=-250+Math.cos(tick*0.22)*55;
  renderer.render(scene,camera);
}
window.addEventListener('resize',()=>{camera.aspect=W()/H();camera.updateProjectionMatrix();renderer.setSize(W(),H());});

// ── Init ──────────────────────────────────────────────────────────────────────
buildScene(); animate();
setTimeout(()=>{fitView();document.getElementById('loader').style.display='none';},200);
</script>
</body>
</html>
"""