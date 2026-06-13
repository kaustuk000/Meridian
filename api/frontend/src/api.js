export const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function handleResponse(res) {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.json()).detail ?? msg; } catch { /* keep default */ }
    throw new Error(msg);
  }
  return res.json();
}

/**
 * Multimodal search — text, image, or both.
 * @param {{ text?: string, image?: File, topk?: number }} opts
 */
export async function searchMeridian({ text, image, topk = 9 }) {
  const form = new FormData();
  if (text?.trim())  form.append("text_query", text.trim());
  if (image)         form.append("image_file", image);
  form.append("topk", String(topk));
  return handleResponse(await fetch(`${BASE}/search`, { method: "POST", body: form }));
}

/**
 * Build a 3-D semantic tree from image files.
 * @param {File[]} files  — at least 2 image files
 */
export async function generateImageHierarchy(
  files,
  useCaptions = false
) {
  const form = new FormData();

  files.forEach(f => form.append("files", f));

  form.append(
    "use_captions",
    useCaptions ? "true" : "false"
  );

  return handleResponse(
    await fetch(
      `${BASE}/hierarchy/images`,
      {
        method: "POST",
        body: form
      }
    )
  );
}

/**
 * Build a 3-D semantic tree from concept terms.
 * @param {string} terms  — comma-separated, ≥ 2 terms
 */
export async function generateTextHierarchy(terms) {
  const form = new FormData();
  form.append("terms", terms);
  return handleResponse(await fetch(`${BASE}/hierarchy/text`, { method: "POST", body: form }));
}

/** Health probe — returns { status, model_loaded, index_loaded, device, node_count } */
export async function getHealth() {
  return handleResponse(await fetch(`${BASE}/healthz`));
}