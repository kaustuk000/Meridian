import { useEffect, useState } from "react";
import { Search, TreePine } from "lucide-react";
import SearchView from "./components/SearchView";
import HierarchyView from "./components/HierarchyView";
import { getHealth } from "./api";

export default function App() {
  const [mode, setMode] = useState("search");
  const [health, setHealth] = useState(null);
  const [seedQuery, setSeedQuery] = useState("");
  const [seedToken, setSeedToken] = useState(0);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "error" }));
  }, []);

  const isOnline = health?.status === "ok";

  function openSearchFromTree(query) {
    const cleaned = (query ?? "").trim();
    if (!cleaned) return;
    setSeedQuery(cleaned);
    setSeedToken((t) => t + 1);
    setMode("search");
  }

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "radial-gradient(circle at top, #121218 0%, #0B0B0E 55%, #09090B 100%)",
        color: "#EDEDED",
        fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        overflow: "hidden",
      }}
    >
      <header
        style={{
          minHeight: 92,
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          display: "flex",
          alignItems: "center",
          paddingInline: 24,
          gap: 20,
          flexShrink: 0,
          background: "linear-gradient(180deg, rgba(18,18,24,0.98) 0%, rgba(11,11,14,0.98) 100%)",
          backdropFilter: "blur(18px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 260 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 12,
              background: "linear-gradient(135deg, #8B5CF6, #3B82F6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 14,
              fontWeight: 800,
              color: "#fff",
              boxShadow: "0 16px 30px rgba(59,130,246,0.20)",
              flexShrink: 0,
            }}
          >
            M
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <span style={{ fontWeight: 800, fontSize: 17, color: "#fff", letterSpacing: "-0.03em" }}>
              Meridian
            </span>
            <span style={{ fontSize: 12.5, color: "#8A8A93", lineHeight: 1.45, maxWidth: 340 }}>
              Multimodal search and semantic hierarchy exploration in one workspace.
            </span>
          </div>
        </div>

        <nav
          style={{
            display: "flex",
            gap: 2,
            background: "rgba(255,255,255,0.05)",
            borderRadius: 14,
            padding: 4,
          }}
        >
          {[
            { key: "search", icon: <Search size={14} />, label: "Search" },
            { key: "hierarchy", icon: <TreePine size={14} />, label: "Hierarchy" },
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setMode(tab.key)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 7,
                padding: "10px 16px",
                borderRadius: 11,
                border: "none",
                fontSize: 13.5,
                fontWeight: 700,
                cursor: "pointer",
                transition: "all 0.15s",
                background: mode === tab.key ? "rgba(255,255,255,0.11)" : "transparent",
                color: mode === tab.key ? "#fff" : "#777781",
                fontFamily: "inherit",
              }}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </nav>

        <div
          style={{
            marginLeft: "auto",
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontSize: 12.5,
            color: "#6E6E78",
          }}
        >
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: health === null ? "#F59E0B" : isOnline ? "#22C55E" : "#EF4444",
              display: "inline-block",
              boxShadow: health === null ? "0 0 0 4px rgba(245,158,11,0.10)" : "none",
            }}
          />
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", lineHeight: 1.25 }}>
            <span style={{ color: "#9A9AA4" }}>
              {health === null ? "connecting" : isOnline ? "online" : "offline"}
            </span>
            <span style={{ color: "#666672" }}>
              {health?.node_count ? `${health.node_count.toLocaleString()} nodes` : "ready"}
            </span>
          </div>
        </div>
      </header>

      <main style={{ flex: 1, overflow: "hidden" }}>
        {mode === "search" && <SearchView presetQuery={seedQuery} seedToken={seedToken} />}
        {mode === "hierarchy" && <HierarchyView onLeafSearch={openSearchFromTree} />}
      </main>
    </div>
  );
}
