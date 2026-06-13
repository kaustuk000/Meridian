import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlignLeft,
  AlertCircle,
  Download,
  ExternalLink,
  FolderOpen,
  ImageIcon,
  Play,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { generateTextHierarchy, generateImageHierarchy, BASE } from "../api";

const TOPK = [6, 9, 12, 18];

function downloadBlob(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function downloadUrl(filename, url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.click();
}

function csvEscape(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function resolveUrl(pathOrUrl) {
  try {
    return new URL(pathOrUrl, BASE).href;
  } catch {
    return pathOrUrl;
  }
}

async function searchMeridianDirect({ textQuery = "", imageFile = null, imageUrl = null, topk = 9 }) {
  const form = new FormData();
  const text = String(textQuery ?? "").trim();
  if (text) form.append("text_query", text);
  if (imageFile) form.append("image_file", imageFile);
  if (imageUrl) form.append("image_url", resolveUrl(imageUrl));
  form.append("topk", String(topk));

  const res = await fetch(`${BASE.replace(/\/$/, "")}/search`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      detail = data?.detail || detail;
    } catch {
      const txt = await res.text().catch(() => "");
      if (txt) detail = txt;
    }
    throw new Error(detail);
  }

  return res.json();
}

export default function HierarchyView({ onLeafSearch }) {
  const [tab, setTab] = useState("text");
  const [terms, setTerms] = useState("");
  const [files, setFiles] = useState([]);
  const [treeUrl, setTreeUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sidebarWidth, setSidebarWidth] = useState(430);
  const [resizing, setResizing] = useState(false);

  const [selectedLeaf, setSelectedLeaf] = useState(null);
  const [leafQuery, setLeafQuery] = useState("");
  const [leafTopk, setLeafTopk] = useState(9);
  const [leafResults, setLeafResults] = useState([]);
  const [leafLoading, setLeafLoading] = useState(false);
  const [leafError, setLeafError] = useState(null);
  const [leafSearched, setLeafSearched] = useState(false);
  const [leafMode, setLeafMode] = useState("caption");
  const [leafImageFile, setLeafImageFile] = useState(null);
  const [leafImagePreview, setLeafImagePreview] = useState(null);

  const folderRef = useRef();
  const leafFileRef = useRef();

  const termList = useMemo(() => terms.split(",").map((t) => t.trim()).filter(Boolean), [terms]);
  const canBuild = tab === "text" ? termList.length >= 2 : files.length >= 2;

  useEffect(() => {
    function handleMessage(event) {
      const data = event.data;
      if (!data || data.type !== "meridian-tree-leaf") return;

      const raw = String(data.query ?? data.node ?? "").trim();
      if (!raw) return;

      const imageUrl = data.image_url ? String(data.image_url) : null;
      setSelectedLeaf({
        name: raw,
        depth: data.depth ?? null,
        children: data.children ?? 0,
        hierarchyType: tab,
        imageUrl,
      });
      setLeafQuery(raw);
      setLeafError(null);
      setLeafSearched(false);
      setLeafResults([]);
      setLeafMode(imageUrl ? "image" : "caption");
      if (imageUrl) {
        if (leafImagePreview) URL.revokeObjectURL(leafImagePreview);
        setLeafImageFile(null);
        setLeafImagePreview(null);
      } else {
        setLeafImageFile(null);
        if (leafImagePreview) URL.revokeObjectURL(leafImagePreview);
        setLeafImagePreview(null);
      }
    }

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, leafImagePreview]);

  useEffect(() => {
    if (!resizing) return;

    function onMove(e) {
      const next = Math.min(560, Math.max(330, e.clientX));
      setSidebarWidth(next);
    }

    function onUp() {
      setResizing(false);
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizing]);

  useEffect(() => {
    return () => {
      if (leafImagePreview) URL.revokeObjectURL(leafImagePreview);
    };
  }, [leafImagePreview]);

  async function build() {
    setLoading(true);
    setError(null);
    try {
      const data = tab === "text"
        ? await generateTextHierarchy(terms.trim())
        : await generateImageHierarchy(files);
      setTreeUrl(null);
      requestAnimationFrame(() => setTreeUrl(data.tree_url));
      setSelectedLeaf(null);
      setLeafResults([]);
      setLeafSearched(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runLeafSearch() {
    const q = leafQuery.trim();
    const nodeImageUrl = selectedLeaf?.imageUrl || null;
    const manualImage = leafImageFile || null;
    const hasImage = Boolean(nodeImageUrl || manualImage);

    if (leafMode === "caption" && !q) return;
    if (leafMode === "image" && !hasImage) return;
    if (leafMode === "both" && (!q || !hasImage)) return;

    setLeafLoading(true);
    setLeafError(null);
    try {
      const data = await searchMeridianDirect({
        textQuery: leafMode === "image" ? "" : q,
        imageFile: nodeImageUrl ? null : manualImage,
        imageUrl: nodeImageUrl,
        topk: leafTopk,
      });
      setLeafResults(data.matches ?? []);
      setLeafSearched(true);
    } catch (e) {
      setLeafError(e.message);
    } finally {
      setLeafLoading(false);
    }
  }

  function promoteToFullSearch() {
    if (typeof onLeafSearch === "function") onLeafSearch(leafQuery.trim());
  }

  function chooseLeafImage(file) {
    if (!file?.type?.startsWith("image/")) return;
    if (leafImagePreview) URL.revokeObjectURL(leafImagePreview);
    setLeafImageFile(file);
    setLeafImagePreview(URL.createObjectURL(file));
    setLeafError(null);
  }

  function clearLeafImage() {
    if (leafImagePreview) URL.revokeObjectURL(leafImagePreview);
    setLeafImageFile(null);
    setLeafImagePreview(null);
  }

  function downloadTree() {
    if (!treeUrl) return;
    fetch(treeUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((html) => downloadBlob("meridian-tree.html", html, "text/html;charset=utf-8"))
      .catch(() => setError("Could not download the current tree HTML."));
  }

  function downloadLeafResultsJSON() {
    const payload = {
      leaf: selectedLeaf?.name ?? "",
      mode: leafMode,
      query: leafQuery.trim(),
      topk: leafTopk,
      results: leafResults,
      generated_at: new Date().toISOString(),
    };
    downloadBlob("meridian-leaf-search-results.json", JSON.stringify(payload, null, 2), "application/json");
  }

  function downloadLeafResultsCSV() {
    const rows = [
      ["id", "score", "caption", "url"],
      ...leafResults.map((r) => [r.id, r.score, r.caption, `${BASE}${r.url}`]),
    ];
    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    downloadBlob("meridian-leaf-search-results.csv", csv, "text/csv;charset=utf-8");
  }

  return (
    <div style={{ height: "100%", display: "flex", minWidth: 0, background: "#0B0B0E" }}>
      <aside
        style={{
          width: sidebarWidth,
          flexShrink: 0,
          borderRight: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          flexDirection: "column",
          padding: 26,
          gap: 18,
          background: "linear-gradient(180deg, rgba(13,13,16,0.98) 0%, rgba(11,11,14,0.98) 100%)",
          overflowY: "auto",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
            <Sparkles size={15} color="#8B5CF6" />
            <span style={{ fontSize: 12.5, color: "#8A8A93", letterSpacing: "0.08em", textTransform: "uppercase" }}>
              Meridian hierarchy
            </span>
          </div>
          <h2 style={{ fontSize: 24, fontWeight: 800, color: "#fff", margin: 0, letterSpacing: "-0.04em" }}>
            Build a semantic tree
          </h2>
          <p style={{ color: "#8A8A93", fontSize: 14.5, marginTop: 8, marginBottom: 0, lineHeight: 1.65 }}>
            Start with text concepts or a folder of images, then explore the generated hierarchy.
          </p>
        </div>

        <div style={{ borderRadius: 16, border: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.03)", padding: 14 }}>
          <div style={{ fontSize: 18, fontWeight: 800, color: "#fff", letterSpacing: "-0.03em" }}>Meridian</div>
          <div style={{ marginTop: 6, color: "#8A8A93", fontSize: 13.5, lineHeight: 1.6 }}>
            Multimodal search and semantic hierarchy exploration in one workspace.
          </div>
        </div>

        <div style={{ display: "flex", gap: 2, background: "#17171C", borderRadius: 14, padding: 4 }}>
          {[
            { key: "text", icon: <AlignLeft size={14} />, label: "Text" },
            { key: "images", icon: <ImageIcon size={14} />, label: "Images" },
          ].map((t) => (
            <button
              key={t.key}
              onClick={() => {
                setTab(t.key);
                setError(null);
              }}
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 7,
                padding: "11px 0",
                borderRadius: 11,
                border: "none",
                cursor: "pointer",
                fontSize: 13.8,
                fontWeight: 700,
                fontFamily: "inherit",
                transition: "all 0.15s",
                background: tab === t.key ? "rgba(255,255,255,0.09)" : "transparent",
                color: tab === t.key ? "#fff" : "#73737C",
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {tab === "text" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <label style={{ fontSize: 13.5, color: "#8A8A93", fontWeight: 700 }}>Concept terms</label>
            <textarea
              value={terms}
              onChange={(e) => {
                setTerms(e.target.value);
                setError(null);
              }}
              placeholder={"dog, cat, mammal,\nherbivore, animal…"}
              rows={7}
              style={{
                background: "#1A1A1F",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 16,
                padding: "14px 15px",
                color: "#EDEDED",
                fontSize: 15,
                resize: "vertical",
                outline: "none",
                fontFamily: "inherit",
                lineHeight: 1.7,
                transition: "border-color 0.15s",
              }}
              onFocus={(e) => (e.target.style.borderColor = "rgba(139,92,246,0.45)")}
              onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.08)")}
            />
            {termList.length > 0 && (
              <p style={{ fontSize: 12.8, color: "#8A8A93", margin: 0 }}>
                {termList.length} term{termList.length !== 1 ? "s" : ""}
                <span style={{ color: termList.length >= 2 ? "#22C55E" : "#F59E0B", marginLeft: 6 }}>
                  {termList.length >= 2 ? "✓ ready" : "need 2+"}
                </span>
              </p>
            )}
            <p style={{ fontSize: 12.5, color: "#676773", margin: 0 }}>Large hierarchies may take some time to build.</p>
          </div>
        )}

        {tab === "images" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <label style={{ fontSize: 13.5, color: "#8A8A93", fontWeight: 700 }}>Image folder</label>
            <button
              onClick={() => folderRef.current?.click()}
              style={{
                padding: "34px 0",
                borderRadius: 18,
                border: "1px dashed rgba(255,255,255,0.12)",
                background: "#1A1A1F",
                color: "#8A8A93",
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
                transition: "all 0.15s",
                fontFamily: "inherit",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "rgba(139,92,246,0.42)";
                e.currentTarget.style.color = "#B4B4BE";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)";
                e.currentTarget.style.color = "#8A8A93";
              }}
            >
              <FolderOpen size={21} />
              <span style={{ fontSize: 13.8 }}>Select folder</span>
            </button>
            <input
              ref={folderRef}
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              style={{ display: "none" }}
              onChange={(e) => {
                setFiles(Array.from(e.target.files).filter((f) => f.type.startsWith("image/")));
                setError(null);
              }}
            />

            {files.length > 0 && (
              <div style={{ borderRadius: 16, background: "#1A1A1F", border: "1px solid rgba(255,255,255,0.06)", padding: 14 }}>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {files.slice(0, 8).map((f, i) => {
                    const url = URL.createObjectURL(f);
                    return (
                      <img
                        key={i}
                        src={url}
                        alt=""
                        onLoad={() => URL.revokeObjectURL(url)}
                        style={{ width: 50, height: 50, objectFit: "cover", borderRadius: 8 }}
                      />
                    );
                  })}
                  {files.length > 8 && (
                    <div style={{ width: 50, height: 50, borderRadius: 8, background: "#252530", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11.5, color: "#8A8A93" }}>
                      +{files.length - 8}
                    </div>
                  )}
                </div>
                <p style={{ margin: "10px 0 0", fontSize: 12.8, color: "#8A8A93" }}>
                  {files.length} image{files.length !== 1 ? "s" : ""} <span style={{ color: "#22C55E", marginLeft: 4 }}>✓ ready</span>
                </p>
              </div>
            )}
            <p style={{ fontSize: 12.5, color: "#676773", margin: 0 }}>Large hierarchies may take some time to build.</p>
          </div>
        )}

        <div style={{ display: "flex", gap: 10 }}>
          {treeUrl && (
            <button onClick={downloadTree} style={secondaryBtn}>
              <Download size={14} /> Download tree
            </button>
          )}
          {treeUrl && (
            <a
              href={treeUrl}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                ...secondaryBtn,
                textDecoration: "none",
                justifyContent: "center",
              }}
            >
              <ExternalLink size={14} /> Open tree
            </a>
          )}
        </div>

        {treeUrl && (
          <div style={{ borderRadius: 16, background: "#1A1A1F", border: "1px solid rgba(255,255,255,0.06)", padding: "12px 14px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13.5 }}>
            <span style={{ color: "#8A8A93" }}>Status</span>
            <span style={{ color: "#22C55E", fontWeight: 700 }}>Tree ready</span>
          </div>
        )}

        {error && (
          <div style={{ display: "flex", gap: 8, padding: "12px 13px", borderRadius: 16, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#F87171", fontSize: 13.5 }}>
            <AlertCircle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
            <span style={{ lineHeight: 1.5 }}>{error}</span>
          </div>
        )}

        <button
          onClick={build}
          disabled={!canBuild || loading}
          style={{
            marginTop: "auto",
            height: 52,
            borderRadius: 16,
            border: "none",
            background: canBuild && !loading ? "linear-gradient(135deg, #8B5CF6, #3B82F6)" : "#1A1A1F",
            color: canBuild && !loading ? "#fff" : "#555",
            fontSize: 14.8,
            fontWeight: 700,
            cursor: canBuild && !loading ? "pointer" : "default",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            transition: "all 0.15s",
            fontFamily: "inherit",
            boxShadow: canBuild && !loading ? "0 16px 36px rgba(59,130,246,0.16)" : "none",
          }}
        >
          {loading ? <Spinner /> : <><Play size={14} /> Build tree</>}
        </button>
      </aside>

      <div
        onMouseDown={() => setResizing(true)}
        style={{
          width: 10,
          cursor: "col-resize",
          background: resizing ? "rgba(139,92,246,0.18)" : "transparent",
          flexShrink: 0,
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 2,
            height: 64,
            borderRadius: 999,
            background: "rgba(255,255,255,0.10)",
          }}
        />
      </div>

      <div style={{ flex: 1, position: "relative", display: "flex", minWidth: 0, background: "#090909" }}>
        <div style={{ flex: 1, position: "relative", minWidth: 0 }}>
          {treeUrl ? (
            <iframe
              key={treeUrl}
              src={treeUrl}
              title="3D Semantic Tree"
              style={{ width: "100%", height: "100%", border: "none", display: "block" }}
            />
          ) : (
            <EmptyTree canBuild={canBuild} onBuild={build} />
          )}

          {loading && (
            <div style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.60)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 10 }}>
              <div style={{ background: "#1A1A1F", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 18, padding: "32px 48px", textAlign: "center" }}>
                <Spinner large />
                <p style={{ color: "#fff", fontWeight: 700, marginTop: 16, marginBottom: 4, fontSize: 16 }}>Building hierarchy</p>
                <p style={{ color: "#8A8A93", fontSize: 13.5, margin: 0 }}>Embedding in hyperbolic space…</p>
              </div>
            </div>
          )}
        </div>

        {selectedLeaf && (
          <LeafSearchDrawer
            leaf={selectedLeaf}
            query={leafQuery}
            setQuery={setLeafQuery}
            topk={leafTopk}
            setTopk={setLeafTopk}
            results={leafResults}
            loading={leafLoading}
            error={leafError}
            searched={leafSearched}
            onSearch={runLeafSearch}
            onPromote={promoteToFullSearch}
            onDownloadJSON={downloadLeafResultsJSON}
            onDownloadCSV={downloadLeafResultsCSV}
            onLeafImagePick={chooseLeafImage}
            leafImagePreview={leafImagePreview}
            leafImageFile={leafImageFile}
            onClearLeafImage={clearLeafImage}
            mode={leafMode}
            setMode={setLeafMode}
            onClose={() => {
              setSelectedLeaf(null);
              clearLeafImage();
            }}
          />
        )}
      </div>
    </div>
  );
}

function LeafSearchDrawer({
  leaf,
  query,
  setQuery,
  topk,
  setTopk,
  results,
  loading,
  error,
  searched,
  onSearch,
  onPromote,
  onDownloadJSON,
  onDownloadCSV,
  onLeafImagePick,
  leafImagePreview,
  leafImageFile,
  onClearLeafImage,
  mode,
  setMode,
  onClose,
}) {
  const leafFileRef = useRef();
  const isImageTree = leaf?.hierarchyType === "images";
  const hasTreeImage = Boolean(leaf?.imageUrl);
  const hasManualImage = Boolean(leafImageFile);

  const imageReady = hasTreeImage || hasManualImage;
  const searchDisabled =
    loading ||
    (mode === "caption" && !query.trim()) ||
    (mode === "image" && !imageReady) ||
    (mode === "both" && (!query.trim() || !imageReady));

  return (
    <aside
      style={{
        width: 470,
        flexShrink: 0,
        borderLeft: "1px solid rgba(255,255,255,0.06)",
        background: "#0D0D0F",
        display: "flex",
        flexDirection: "column",
        minWidth: 340,
      }}
    >
      <div style={{ padding: 22, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 14 }}>
          <div>
            <div style={{ fontSize: 12.5, color: "#8A8A93", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>
              {isImageTree ? "Image node" : "Text node"}
            </div>
            <h3 style={{ margin: 0, color: "#fff", fontSize: 22, letterSpacing: "-0.03em" }}>
              {leaf.name}
            </h3>
            <p style={{ margin: "8px 0 0", color: "#8A8A93", fontSize: 13.8, lineHeight: 1.65 }}>
              Search this node directly with the same Meridian retrieval flow.
            </p>
          </div>
          <button onClick={onClose} style={closeBtn} aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
          <button onClick={onDownloadJSON} disabled={!results.length} style={drawerBtn}>
            <Download size={14} /> JSON
          </button>
          <button onClick={onDownloadCSV} disabled={!results.length} style={drawerBtn}>
            <Download size={14} /> CSV
          </button>
          <button onClick={onPromote} style={{ ...drawerBtn, marginLeft: "auto" }}>
            Open in main search
          </button>
        </div>
      </div>

      <div style={{ padding: 20, borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
          {[
            { key: "caption", label: "Caption" },
            { key: "image", label: "Image" },
            { key: "both", label: "Both" },
          ].map((item) => (
            <button
              key={item.key}
              onClick={() => setMode(item.key)}
              style={{
                ...modeChip,
                borderColor: mode === item.key ? "rgba(139,92,246,0.55)" : "rgba(255,255,255,0.08)",
                background: mode === item.key ? "rgba(139,92,246,0.14)" : "rgba(255,255,255,0.03)",
                color: mode === item.key ? "#C4B5FD" : "#9A9AA4",
              }}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "stretch" }}>
          <div style={{ flex: 1, position: "relative" }}>
            <Search size={16} style={leafSearchIcon} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSearch()}
              placeholder={mode === "image" ? "Caption optional for image search…" : "Search this leaf…"}
              style={leafInput}
            />
          </div>

          {!isImageTree && (
            <>
              <button onClick={() => leafFileRef.current?.click()} title="Add image" style={leafIconButton}>
                <ImageIcon size={18} />
              </button>
              <input
                ref={leafFileRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => onLeafImagePick(e.target.files[0])}
              />
            </>
          )}

          <button onClick={onSearch} disabled={searchDisabled} style={leafSearchBtn}>
            {loading ? <Spinner /> : "Go"}
          </button>
        </div>

        {hasTreeImage && (
          <div style={leafImageCard}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <img
                src={resolveUrl(leaf.imageUrl)}
                alt={leaf.name}
                style={{ width: 54, height: 54, borderRadius: 10, objectFit: "cover" }}
              />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ color: "#EDEDED", fontSize: 13.5, fontWeight: 600 }}>Tree node image</div>
                <div style={{ color: "#8A8A93", fontSize: 12.5, marginTop: 4 }}>
                  This image is used automatically for image and mixed search.
                </div>
              </div>
            </div>
          </div>
        )}

        {!isImageTree && leafImagePreview && (
          <div style={leafImageCard}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <img src={leafImagePreview} alt="" style={{ width: 54, height: 54, borderRadius: 10, objectFit: "cover" }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ color: "#EDEDED", fontSize: 13.5, fontWeight: 600 }}>Attached image</div>
                <div style={{ color: "#8A8A93", fontSize: 12.5, marginTop: 4 }}>Used for image or mixed search.</div>
              </div>
              <button onClick={onClearLeafImage} style={miniCloseBtn} aria-label="Remove image">
                <X size={14} />
              </button>
            </div>
          </div>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ color: "#8A8A93", fontSize: 13 }}>Show</span>
          {TOPK.map((k) => (
            <button
              key={k}
              onClick={() => setTopk(k)}
              style={{
                ...tinyChip,
                borderColor: topk === k ? "rgba(139,92,246,0.55)" : "rgba(255,255,255,0.08)",
                background: topk === k ? "rgba(139,92,246,0.14)" : "rgba(255,255,255,0.03)",
                color: topk === k ? "#C4B5FD" : "#9A9AA4",
              }}
            >
              {k}
            </button>
          ))}
          <span style={{ marginLeft: "auto", color: "#8A8A93", fontSize: 12.5 }}>
            {mode === "caption" ? "caption search" : mode === "image" ? "image search" : "mixed search"}
          </span>
        </div>

        {error && <p style={{ margin: 0, color: "#F87171", fontSize: 13.5 }}>{error}</p>}
      </div>

      <div style={{ padding: 20, overflowY: "auto", flex: 1 }}>
        {searched && results.length === 0 ? (
          <div style={emptyLeafStyle}>No matches for this leaf yet.</div>
        ) : results.length === 0 ? (
          <div style={emptyLeafStyle}>Run a search to populate results here.</div>
        ) : (
          <div style={{ display: "grid", gap: 12 }}>
            {results.map((r) => (
              <div
                key={r.id}
                style={{
                  ...leafCardStyle,
                  position: "relative",
                }}
              >
                <img
                  src={`${BASE}${r.url}`}
                  alt={r.caption}
                  style={{ width: "100%", height: 160, objectFit: "cover", borderRadius: 12, border: "1px solid rgba(255,255,255,0.05)" }}
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
                <button
                  onClick={() => downloadUrl(`meridian-result-${r.id}.jpg`, `${BASE}${r.url}`)}
                  style={resultDownloadBtn}
                  title="Download image"
                >
                  <Download size={14} />
                </button>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, marginTop: 10 }}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ color: "#EDEDED", fontSize: 13.5, fontWeight: 600, lineHeight: 1.45, overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical" }}>
                      {r.caption}
                    </div>
                    <div style={{ color: "#8A8A93", fontSize: 12, marginTop: 4 }}>id {r.id}</div>
                  </div>
                  <div style={leafScoreStyle}>{(r.score * 100).toFixed(1)}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

function Spinner({ large = false }) {
  return <span style={large ? spinnerLargeStyle : spinnerStyle} />;
}

function EmptyTree({ canBuild, onBuild }) {
  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, userSelect: "none", color: "#333" }}>
      <svg width="120" height="120" viewBox="0 0 100 100" fill="none" style={{ opacity: 0.38 }}>
        <circle cx="50" cy="50" r="46" stroke="#8B5CF6" strokeWidth="1" />
        <circle cx="50" cy="50" r="32" stroke="#7C3AED" strokeWidth="0.8" strokeDasharray="3 3" />
        <circle cx="50" cy="50" r="18" stroke="#3B82F6" strokeWidth="0.8" strokeDasharray="2 4" />
        <circle cx="50" cy="50" r="7" stroke="#3B82F6" strokeWidth="0.6" />
        <path d="M10 50 Q50 12 90 50" stroke="#8B5CF6" strokeWidth="0.7" strokeDasharray="2 3" />
        <path d="M10 50 Q50 88 90 50" stroke="#8B5CF6" strokeWidth="0.7" strokeDasharray="2 3" />
      </svg>

      <div style={{ textAlign: "center", maxWidth: 460 }}>
        <h3 style={{ color: "#EDEDED", fontSize: 19, margin: "0 0 8px" }}>Nothing built yet</h3>
        <p style={{ color: "#666", fontSize: 14.5, lineHeight: 1.75, margin: 0 }}>
          Add at least two terms or two images, then build a tree and click a leaf node to search it on the right.
        </p>
      </div>

      <button
        onClick={onBuild}
        disabled={!canBuild}
        style={{
          ...drawerBtn,
          height: 44,
          paddingInline: 18,
          marginTop: 6,
          opacity: canBuild ? 1 : 0.45,
        }}
      >
        <Play size={14} /> Build tree
      </button>
    </div>
  );
}

const secondaryBtn = {
  flex: 1,
  height: 42,
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#D7D7DE",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 8,
  cursor: "pointer",
  fontFamily: "inherit",
  fontSize: 13.5,
  fontWeight: 600,
};

const drawerBtn = {
  height: 38,
  padding: "0 14px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#D7D7DE",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 8,
  cursor: "pointer",
  fontFamily: "inherit",
  fontSize: 13,
  fontWeight: 600,
};

const closeBtn = {
  width: 36,
  height: 36,
  borderRadius: 10,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#D7D7DE",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  flexShrink: 0,
};

const miniCloseBtn = {
  width: 30,
  height: 30,
  borderRadius: 9,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#D7D7DE",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  flexShrink: 0,
};

const tinyChip = {
  height: 32,
  padding: "0 10px",
  borderRadius: 9,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#9A9AA4",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  fontFamily: "inherit",
  fontSize: 12.5,
  fontWeight: 600,
};

const modeChip = {
  height: 36,
  padding: "0 10px",
  borderRadius: 10,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#9A9AA4",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  fontFamily: "inherit",
  fontSize: 12.8,
  fontWeight: 700,
};

const leafInput = {
  width: "100%",
  height: 46,
  paddingLeft: 40,
  paddingRight: 14,
  background: "#1A1A1F",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 13,
  color: "#EDEDED",
  fontSize: 14.5,
  outline: "none",
  transition: "border-color 0.15s",
  boxSizing: "border-box",
};

const leafSearchIcon = {
  position: "absolute",
  left: 13,
  top: "50%",
  transform: "translateY(-50%)",
  color: "#4B4B57",
  pointerEvents: "none",
};

const leafSearchBtn = {
  height: 46,
  minWidth: 70,
  padding: "0 18px",
  borderRadius: 13,
  border: "none",
  background: "linear-gradient(135deg, #8B5CF6, #3B82F6)",
  color: "#fff",
  fontSize: 13.5,
  fontWeight: 700,
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  flexShrink: 0,
};

const leafIconButton = {
  height: 46,
  width: 46,
  borderRadius: 13,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#D7D7DE",
  cursor: "pointer",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  flexShrink: 0,
};

const leafCardStyle = {
  borderRadius: 16,
  background: "#17171C",
  border: "1px solid rgba(255,255,255,0.05)",
  padding: 12,
};

const leafImageCard = {
  borderRadius: 14,
  border: "1px solid rgba(255,255,255,0.06)",
  background: "#151518",
  padding: 12,
};

const leafScoreStyle = {
  minWidth: 54,
  height: 32,
  borderRadius: 10,
  background: "rgba(0,0,0,0.55)",
  color: "#34D399",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 12,
  fontWeight: 700,
  fontFamily: "ui-monospace, monospace",
};

const resultDownloadBtn = {
  position: "absolute",
  top: 22,
  right: 22,
  width: 34,
  height: 34,
  borderRadius: 10,
  background: "rgba(0,0,0,.75)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "#fff",
  textDecoration: "none",
  border: "1px solid rgba(255,255,255,.10)",
  cursor: "pointer",
};

const spinnerStyle = {
  width: 16,
  height: 16,
  borderRadius: "50%",
  border: "2px solid rgba(255,255,255,0.22)",
  borderTopColor: "#fff",
  display: "inline-block",
  animation: "spin 0.7s linear infinite",
};

const spinnerLargeStyle = {
  width: 36,
  height: 36,
  borderRadius: "50%",
  border: "2px solid rgba(139,92,246,0.20)",
  borderTopColor: "#8B5CF6",
  display: "inline-block",
  animation: "spin 0.7s linear infinite",
};

const emptyLeafStyle = {
  borderRadius: 16,
  border: "1px dashed rgba(255,255,255,0.08)",
  padding: 18,
  color: "#8A8A93",
  fontSize: 13.5,
  lineHeight: 1.7,
  background: "#151518",
};
