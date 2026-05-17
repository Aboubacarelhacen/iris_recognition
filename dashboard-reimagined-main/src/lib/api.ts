/**
 * Typed client for the Iris Recognition FastAPI backend at /api.
 * The Vite dev server proxies these calls to http://127.0.0.1:8000.
 */

export type Quality = "high" | "medium" | "low" | "unknown";

export interface ModelStats {
  available: boolean;
  eer: number | null;
  threshold: number | null;
  far_at_eer: number | null;
  frr_at_eer: number | null;
  auc: number | null;
  num_images?: number | null;
  num_skipped?: number | null;
  n_gen?: number | null;
  n_imp?: number | null;
}

export interface TopSubject {
  subject_id: string;
  intra_hd: number;
  n_codes: number;
  quality: Quality;
}

export interface Summary {
  n_subjects: number;
  n_codes: number;
  daugman: ModelStats;
  cnn: ModelStats;
  quality_buckets: Record<Quality, number>;
  top_consistent_subjects: TopSubject[];
  has_llm_analysis: boolean;
}

export interface SubjectRow {
  subject_id: string;
  n_codes: number;
  intra_hd: number | null;
  quality: Quality;
}

export interface SubjectCodePreview {
  image_path: string;
  code_png: string;
}

export interface SubjectDetail {
  subject_id: string;
  n_codes: number;
  intra_hd: number | null;
  quality: Quality;
  codes: SubjectCodePreview[];
}

export interface VerifyResult {
  daugman: { hd: number; threshold: number; match: boolean };
  cnn: {
    cosine_distance: number;
    threshold: number;
    match: boolean;
  } | null;
  artifacts: {
    strip_a: string;
    strip_b: string;
    code_a: string;
    code_b: string;
  };
}

export interface TrainingEpoch {
  epoch: number;
  loss: number;
  eer: number;
  ma_eer?: number;
  thr: number;
  margin: number;
}

export interface Health {
  status: string;
  n_subjects: number;
  n_codes: number;
  has_daugman: boolean;
  has_cnn: boolean;
}

// ----------------------------------------------------------------- helpers
async function jget<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function jpost<T>(path: string, body: object = {}): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function verifyUpload(a: File, b: File): Promise<VerifyResult> {
  const fd = new FormData();
  fd.append("image_a", a);
  fd.append("image_b", b);
  const r = await fetch("/api/verify", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`verify: ${r.status} ${await r.text()}`);
  return r.json();
}

// ----------------------------------------------------------------- routes
export const api = {
  health:    () => jget<Health>("/api/health"),
  summary:   () => jget<Summary>("/api/summary"),
  subjects:  (params: { limit?: number; offset?: number; sort?: string; q?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.limit  !== undefined) q.set("limit",  String(params.limit));
    if (params.offset !== undefined) q.set("offset", String(params.offset));
    if (params.sort)  q.set("sort", params.sort);
    if (params.q)     q.set("q", params.q);
    return jget<{ count: number; items: SubjectRow[] }>(`/api/subjects?${q.toString()}`);
  },
  subject: (id: string) => jget<SubjectDetail>(`/api/subjects/${id}`),
  verify:  verifyUpload,
  llmAnalysis:   () => jget<{ available: boolean; text: string | null }>("/api/llm-analysis"),
  regenerateLlm: () => jpost<{ available: boolean; text: string }>("/api/llm-analysis/regenerate"),
  trainingHistory: () => jget<{ history: TrainingEpoch[] }>("/api/training-history"),
};

// ----------------------------------------------------------------- helpers
export function qualityTone(q: Quality): { bg: string; text: string; ring: string } {
  if (q === "high")
    return { bg: "bg-[color:var(--risk-low)]/15",    text: "text-[color:var(--risk-low)]",    ring: "ring-[color:var(--risk-low)]/30" };
  if (q === "medium")
    return { bg: "bg-[color:var(--risk-medium)]/15", text: "text-[color:var(--risk-medium)]", ring: "ring-[color:var(--risk-medium)]/30" };
  if (q === "low")
    return { bg: "bg-[color:var(--risk-high)]/15",   text: "text-[color:var(--risk-high)]",   ring: "ring-[color:var(--risk-high)]/30" };
  return { bg: "bg-muted", text: "text-muted-foreground", ring: "ring-border" };
}

export function fmtPct(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtNum(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}
