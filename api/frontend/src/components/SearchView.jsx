import { useEffect, useMemo, useRef, useState } from "react";
import { Search, ImagePlus, X, Download, Sparkles } from "lucide-react";
import { searchMeridian, BASE } from "../api";

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

export default function SearchView({ presetQuery = "", seedToken = 0 }) {
  const [query, setQuery] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [topk, setTopk] = useState(9);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searched, setSearched] = useState(false);
  const [lastRunQuery, setLastRunQuery] = useState("");
  const fileRef = useRef();

  useEffect(() => {
    const cleaned = (presetQuery ?? "").trim();
    if (!cleaned) return;
    setQuery(cleaned);
    if (seedToken > 0) run(cleaned);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetQuery, seedToken]);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function pickImage(file) {
    if (!file?.type?.startsWith("image/")) return;
    if (preview) URL.revokeObjectURL(preview);
    setImage(file);
    setPreview(URL.createObjectURL(file));
  }

  function clearImage() {
    if (preview) URL.revokeObjectURL(preview);
    setImage(null);
    setPreview(null);
  }

  async function run(overrideQuery = query, overrideImage = image, overrideTopk = topk) {
    const text = overrideQuery.trim();
    if (!text && !overrideImage) return;

    setLoading(true);
    setError(null);
    try {
      const data = await searchMeridian({ text, image: overrideImage, topk: overrideTopk });
      setResults(data.matches ?? []);
      setSearched(true);
      setLastRunQuery(text);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function downloadResultsJSON() {
    const payload = {
      query: lastRunQuery,
      topk,
      results,
      generated_at: new Date().toISOString(),
    };
    downloadBlob("meridian-search-results.json", JSON.stringify(payload, null, 2), "application/json");
  }

  function downloadResultsCSV() {
    const rows = [
      ["id", "score", "caption", "url"],
      ...results.map((r) => [r.id, r.score, r.caption, `${BASE}${r.url}`]),
    ];
    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    downloadBlob("meridian-search-results.csv", csv, "text/csv;charset=utf-8");
  }

  const hasInput = query.trim() || image;
  const resultCount = useMemo(() => results.length, [results]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ padding: "40px 44px 24px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ maxWidth: 980, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 24, marginBottom: 22, flexWrap: "wrap" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                <Sparkles size={16} color="#8B5CF6" />
                <span style={{ fontSize: 12.5, color: "#8F8F98", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                  Meridian search
                </span>
              </div>
              <h1 style={{ fontSize: 36, fontWeight: 800, color: "#fff", letterSpacing: "-0.05em", margin: 0, lineHeight: 1.05 }}>
                Search with text, image, or both
              </h1>
              <p style={{ color: "#8A8A93", marginTop: 10, fontSize: 15.5, lineHeight: 1.65, maxWidth: 740 }}>
                Use a natural-language query, add an image, and refine how many matches you see.
                Leaf-node clicks from the tree open the same retrieval flow with the node name already filled in.
              </p>
            </div>

            {searched && (
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button onClick={downloadResultsJSON} style={downloadBtn}>
                  <Download size={14} /> JSON
                </button>
                <button onClick={downloadResultsCSV} style={downloadBtn}>
                  <Download size={14} /> CSV
                </button>
              </div>
            )}
          </div>

          <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 300, position: "relative" }}>
              <Search size={18} style={searchIconStyle} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && run()}
                placeholder="Describe what you're looking for…"
                style={inputStyle}
                onFocus={(e) => (e.target.style.borderColor = "rgba(139,92,246,0.55)")}
                onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.08)")}
              />
              {preview && (
                <div style={imageChipStyle}>
                  <img src={preview} alt="" style={{ width: 28, height: 28, borderRadius: 6, objectFit: "cover" }} />
                  <button onClick={clearImage} style={chipCloseStyle} aria-label="Remove image">
                    <X size={13} />
                  </button>
                </div>
              )}
            </div>

            <button onClick={() => fileRef.current?.click()} title="Add image" style={iconButtonStyle}>
              <ImagePlus size={18} />
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => pickImage(e.target.files[0])}
            />

            <button
              onClick={() => run()}
              disabled={!hasInput || loading}
              style={{
                height: 52,
                minWidth: 136,
                paddingInline: 22,
                borderRadius: 16,
                border: "none",
                background: hasInput && !loading ? "linear-gradient(135deg, #8B5CF6, #3B82F6)" : "#1A1A1F",
                color: hasInput && !loading ? "#fff" : "#555",
                fontSize: 14.5,
                fontWeight: 700,
                cursor: hasInput && !loading ? "pointer" : "default",
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                transition: "all 0.15s",
                fontFamily: "inherit",
                boxShadow: hasInput && !loading ? "0 16px 36px rgba(59,130,246,0.16)" : "none",
              }}
            >
              {loading ? <Spinner /> : "Search"}
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 16, marginTop: 16, flexWrap: "wrap" }}>
            <div style={{ color: "#8A8A93", fontSize: 13.5 }}>
              {searched ? `${resultCount} results` : "Ready to search"}
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: "auto", flexWrap: "wrap" }}>
              <span style={{ color: "#8A8A93", fontSize: 13.5 }}>Show</span>
              {TOPK.map((k) => (
                <button
                  key={k}
                  onClick={() => setTopk(k)}
                  style={{
                    padding: "6px 12px",
                    borderRadius: 9,
                    border: "1px solid",
                    borderColor: topk === k ? "rgba(139,92,246,0.55)" : "rgba(255,255,255,0.08)",
                    background: topk === k ? "rgba(139,92,246,0.14)" : "rgba(255,255,255,0.02)",
                    color: topk === k ? "#C4B5FD" : "#9A9AA4",
                    fontSize: 12.5,
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>

          {error && <p style={{ maxWidth: 920, margin: "14px 0 0", color: "#F87171", fontSize: 13.5 }}>{error}</p>}
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "28px 44px" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto" }}>
          {results.length === 0 ? (
            <EmptyState searched={searched} />
          ) : (
            <div style={resultsGridStyle}>
              {results.map((r) => (
                <ResultCard key={r.id} r={r} />
              ))}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        input::placeholder { color: #4A4A55; }
      `}</style>
    </div>
  );
}

function Spinner() {
  return <span style={spinnerStyle} />;
}

function ResultCard({ r }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "relative",
        borderRadius: 18,
        overflow: "hidden",
        marginBottom: 14,
        cursor: "pointer",
        breakInside: "avoid",
        background: "#17171C",
        border: "1px solid rgba(255,255,255,0.05)",
        boxShadow: hovered ? "0 20px 44px rgba(0,0,0,0.28)" : "none",
      }}
    >
      <img
        src={`${BASE}${r.url}`}
        alt={r.caption}
        style={{
          width: "100%",
          display: "block",
          transition: "transform 0.4s",
          transform: hovered ? "scale(1.04)" : "scale(1)",
        }}
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />

      <div style={scoreBadgeStyle}>
        {(r.score * 100).toFixed(1)}
      </div>

      <a
        href={`${BASE}${r.url}`}
        download
        target="_blank"
        rel="noopener noreferrer"
        style={{
          position: "absolute",
          top: 14,
          right: 14,
          width: 34,
          height: 34,
          borderRadius: 10,
          background: "rgba(0,0,0,0.72)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
          textDecoration: "none",
          border: "1px solid rgba(255,255,255,0.10)",
          opacity: hovered ? 1 : 0.92,
        }}
        title="Download image"
      >
        <Download size={14} />
      </a>

      <div style={{ ...captionOverlayStyle, opacity: hovered ? 1 : 0 }}>
        <p
          style={{
            color: "rgba(255,255,255,0.94)",
            fontSize: 13.5,
            lineHeight: 1.6,
            margin: 0,
            display: "-webkit-box",
            WebkitLineClamp: 3,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
          }}
        >
          {r.caption}
        </p>
        <p style={{ color: "#8A8A93", fontSize: 11.5, marginTop: 6, marginBottom: 0 }}>
          item #{r.id}
        </p>
      </div>
    </div>
  );
}

function EmptyState({ searched }) {
  return (
    <div style={{ height: "100%", minHeight: 380, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, color: "#555" }}>
      <div style={emptyIconWrap}>
        <Search size={30} color="#8B5CF6" />
      </div>
      <div style={{ textAlign: "center", maxWidth: 520 }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: "#ECECF0", marginBottom: 8 }}>
          {searched ? "No matches returned" : "Start with a query"}
        </div>
        <p style={{ margin: 0, fontSize: 14.5, lineHeight: 1.7, color: "#8A8A93" }}>
          {searched
            ? "Try a broader phrase, switch the top-k count, or add an image for a stronger multimodal query."
            : "Enter a phrase, attach an image, and press Search. The result grid will fill here."}
        </p>
      </div>
    </div>
  );
}

const downloadBtn = {
  height: 40,
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
  fontSize: 13.5,
  fontWeight: 600,
};

const searchIconStyle = {
  position: "absolute",
  left: 14,
  top: "50%",
  transform: "translateY(-50%)",
  color: "#4B4B57",
  pointerEvents: "none",
};

const inputStyle = {
  width: "100%",
  height: 52,
  paddingLeft: 42,
  paddingRight: 14,
  background: "#1A1A1F",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 16,
  color: "#EDEDED",
  fontSize: 15,
  outline: "none",
  transition: "border-color 0.15s",
  boxSizing: "border-box",
};

const imageChipStyle = {
  position: "absolute",
  right: 12,
  top: "50%",
  transform: "translateY(-50%)",
  height: 34,
  padding: "0 8px",
  borderRadius: 10,
  background: "#24242A",
  border: "1px solid rgba(255,255,255,0.08)",
  display: "flex",
  alignItems: "center",
  gap: 8,
};

const chipCloseStyle = {
  width: 22,
  height: 22,
  borderRadius: 7,
  border: "none",
  background: "rgba(255,255,255,0.08)",
  color: "#EDEDED",
  cursor: "pointer",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
};

const iconButtonStyle = {
  height: 52,
  width: 52,
  borderRadius: 16,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.03)",
  color: "#D7D7DE",
  cursor: "pointer",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  flexShrink: 0,
};

const resultsGridStyle = {
  columnCount: 3,
  columnGap: 16,
};

const scoreBadgeStyle = {
  position: "absolute",
  left: 14,
  top: 14,
  minWidth: 54,
  height: 32,
  padding: "0 10px",
  borderRadius: 10,
  background: "rgba(0,0,0,0.60)",
  color: "#34D399",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 12,
  fontWeight: 700,
  fontFamily: "ui-monospace, monospace",
};

const captionOverlayStyle = {
  position: "absolute",
  left: 0,
  right: 0,
  bottom: 0,
  padding: "12px 14px 14px",
  background: "linear-gradient(180deg, rgba(8,8,10,0.00) 0%, rgba(8,8,10,0.92) 100%)",
  transition: "opacity 0.18s",
};

const emptyIconWrap = {
  width: 68,
  height: 68,
  borderRadius: 22,
  background: "rgba(139,92,246,0.10)",
  border: "1px solid rgba(139,92,246,0.18)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
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
