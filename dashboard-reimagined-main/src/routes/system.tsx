import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Layers, Bot, RefreshCcw, BookOpen, Cpu, ScanEye, Sparkles, AlertTriangle,
} from "lucide-react";
import { api, fmtNum, fmtPct } from "@/lib/api";
import { Sidebar } from "./index";

export const Route = createFileRoute("/system")({
  component: SystemPage,
  head: () => ({ meta: [{ title: "System · Iris Recognition Console" }] }),
});

function TrainingCurve({ data }: { data: { epoch: number; eer: number; ma_eer?: number }[] }) {
  if (!data?.length) return null;
  const maxEpoch = Math.max(...data.map((d) => d.epoch));
  const maxY = Math.max(...data.map((d) => d.eer)) * 1.05 || 1;
  const w = 720;
  const h = 240;
  const pad = { l: 36, r: 12, t: 12, b: 24 };
  const ix = (e: number) => pad.l + ((e - 1) / Math.max(1, maxEpoch - 1)) * (w - pad.l - pad.r);
  const iy = (y: number) => pad.t + (1 - y / maxY) * (h - pad.t - pad.b);
  const path = (key: "eer" | "ma_eer") =>
    data
      .filter((d) => d[key] !== undefined && d[key] !== null)
      .map((d, i) => `${i === 0 ? "M" : "L"} ${ix(d.epoch)} ${iy(d[key] as number)}`)
      .join(" ");
  const ticks = 4;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-64 w-full">
      {Array.from({ length: ticks + 1 }).map((_, i) => {
        const y = pad.t + (i / ticks) * (h - pad.t - pad.b);
        const v = ((ticks - i) / ticks) * maxY;
        return (
          <g key={i}>
            <line x1={pad.l} x2={w - pad.r} y1={y} y2={y} stroke="currentColor" opacity="0.06" />
            <text x={pad.l - 6} y={y + 3} textAnchor="end" fontSize="10" fill="currentColor" opacity="0.5">
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
      <path d={path("eer")} fill="none" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.5" />
      <path d={path("ma_eer")} fill="none" stroke="var(--lime)" strokeWidth="2.5" />
      {data.map((d) => (
        <circle key={d.epoch} cx={ix(d.epoch)} cy={iy(d.eer)} r="2.5" fill="currentColor" opacity="0.5" />
      ))}
    </svg>
  );
}

function SystemPage() {
  const summaryQ = useQuery({ queryKey: ["summary"], queryFn: api.summary });
  const llmQ = useQuery({ queryKey: ["llm"], queryFn: api.llmAnalysis });
  const histQ = useQuery({ queryKey: ["history"], queryFn: api.trainingHistory });

  const qc = useQueryClient();
  const regen = useMutation({
    mutationFn: api.regenerateLlm,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["llm"] }),
  });

  const s = summaryQ.data;
  const text = llmQ.data?.text;
  const history = histQ.data?.history ?? [];

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="mx-auto flex max-w-[1500px] gap-6">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <div className="flex flex-wrap items-end justify-between gap-6">
            <div>
              <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-foreground/50">
                <Layers size={14} /> System
              </div>
              <h1 className="mt-2 font-display text-3xl font-medium leading-[1.05] tracking-tighter sm:text-4xl md:text-5xl xl:text-[56px]">
                Pipeline{" "}
                <span className="inline-grid h-10 w-14 translate-y-0.5 place-items-center rounded-full bg-lime align-middle md:h-12 md:w-16">
                  <Sparkles size={20} strokeWidth={1.75} />
                </span>{" "}
                & analysis
              </h1>
            </div>
          </div>

          <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-[1fr_360px]">
            {/* Left column */}
            <div className="flex flex-col gap-6">
              <div className="rounded-[2rem] bg-surface p-6">
                <div className="flex items-center gap-3">
                  <div className="grid h-11 w-11 place-items-center rounded-2xl bg-background"><BookOpen size={18} /></div>
                  <h3 className="font-display text-2xl font-medium tracking-tight">Daugman's algorithm</h3>
                </div>
                <ol className="mt-4 flex flex-col gap-3 text-sm leading-relaxed text-foreground/80">
                  <li><span className="font-semibold text-foreground">1. Segmentation —</span> locate iris and pupil with Hough circles after CLAHE contrast enhancement.</li>
                  <li><span className="font-semibold text-foreground">2. Normalization —</span> unwrap the annular iris region to a 64 × 512 strip via the rubber-sheet mapping.</li>
                  <li><span className="font-semibold text-foreground">3. Encoding —</span> apply a 4-orientation Gabor filter bank (quadrature pair) and binarize → 2048-bit IrisCode.</li>
                  <li><span className="font-semibold text-foreground">4. Matching —</span> normalized Hamming distance, minimized over ±2 angular shifts for rotation tolerance.</li>
                </ol>
                <pre className="mt-5 overflow-x-auto rounded-xl bg-ink p-4 text-xs leading-relaxed text-white/80">
{`eye  →  CLAHE  →  Hough  →  rubber-sheet  →  Gabor  →  bits  →  HD`}
                </pre>
              </div>

              <div className="rounded-[2rem] bg-surface p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="grid h-11 w-11 place-items-center rounded-2xl bg-background"><Cpu size={18} /></div>
                    <h3 className="font-display text-2xl font-medium tracking-tight">Model architecture</h3>
                  </div>
                  <div className="hidden text-xs text-foreground/60 md:block">
                    end-to-end iris verification pipeline
                  </div>
                </div>
                <div className="mt-4 overflow-hidden rounded-2xl bg-background/40">
                  <img
                    src="/iris-pipeline.png"
                    alt="Iris recognition pipeline: two iris inputs flow through segmentation, a ResNet-18 encoder, and a comparison step that produces a match decision."
                    className="block h-auto w-full"
                  />
                </div>
                <p className="mt-3 text-center text-xs text-foreground/55">
                  Two iris images → segmentation → ResNet-18 encoder → cosine comparison → match decision.
                </p>
              </div>

              <div className="rounded-[2rem] bg-surface p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="grid h-11 w-11 place-items-center rounded-2xl bg-background"><Bot size={18} /></div>
                    <h3 className="font-display text-2xl font-medium tracking-tight">LLM analysis — Türkçe</h3>
                  </div>
                  <button
                    disabled={regen.isPending}
                    onClick={() => regen.mutate()}
                    className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-40"
                  >
                    <RefreshCcw size={14} className={regen.isPending ? "animate-spin" : ""} />
                    {regen.isPending ? "Generating…" : "Regenerate (Ollama)"}
                  </button>
                </div>
                {regen.isError && (
                  <div className="mt-3 flex items-start gap-2 rounded-xl bg-[color:var(--risk-medium)]/15 p-3 text-xs text-[color:var(--risk-medium)]">
                    <AlertTriangle size={14} /> {(regen.error as Error).message}
                  </div>
                )}
                <div className="mt-4 max-h-[480px] overflow-auto rounded-2xl bg-background/60 p-5 text-sm leading-relaxed">
                  {text ? (
                    <article className="whitespace-pre-wrap font-sans text-foreground/85">{text}</article>
                  ) : (
                    <div className="text-foreground/50">
                      No analysis yet. Start Ollama (<code>ollama serve</code>) and click <em>Regenerate</em>.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right column — metrics summary */}
            <aside className="flex flex-col gap-4">
              <div className="rounded-[2rem] bg-ink p-6 text-white">
                <div className="text-xs uppercase tracking-widest text-white/60">Best CNN checkpoint</div>
                <div className="mt-2 font-display text-3xl tracking-tight text-lime">
                  AUC {fmtNum(s?.cnn?.auc)}
                </div>
                <div className="mt-1 text-sm text-white/70">EER {fmtPct(s?.cnn?.eer)} on held-out subjects</div>
                <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  <div className="rounded-xl bg-white/10 p-3">
                    <div className="text-white/60 text-xs">Threshold</div>
                    <div>{fmtNum(s?.cnn?.threshold)}</div>
                  </div>
                  <div className="rounded-xl bg-white/10 p-3">
                    <div className="text-white/60 text-xs">FAR</div>
                    <div>{fmtPct(s?.cnn?.far_at_eer)}</div>
                  </div>
                  <div className="rounded-xl bg-white/10 p-3">
                    <div className="text-white/60 text-xs">FRR</div>
                    <div>{fmtPct(s?.cnn?.frr_at_eer)}</div>
                  </div>
                  <div className="rounded-xl bg-white/10 p-3">
                    <div className="text-white/60 text-xs">Pairs</div>
                    <div>{s?.cnn?.n_gen ?? "—"}/{s?.cnn?.n_imp ?? "—"}</div>
                  </div>
                </div>
              </div>

              <div className="rounded-[2rem] bg-surface p-6">
                <div className="text-xs uppercase tracking-widest text-foreground/50">Daugman baseline</div>
                <div className="mt-2 font-display text-3xl tracking-tight">{fmtPct(s?.daugman?.eer)}</div>
                <div className="mt-1 text-sm text-foreground/60">EER (1993 reference algorithm)</div>
                <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  <div className="rounded-xl bg-background/60 p-3">
                    <div className="text-foreground/50 text-xs">AUC</div>
                    <div>{fmtNum(s?.daugman?.auc)}</div>
                  </div>
                  <div className="rounded-xl bg-background/60 p-3">
                    <div className="text-foreground/50 text-xs">Images</div>
                    <div>{s?.daugman?.num_images ?? "—"}</div>
                  </div>
                  <div className="rounded-xl bg-background/60 p-3">
                    <div className="text-foreground/50 text-xs">Skipped</div>
                    <div>{s?.daugman?.num_skipped ?? "—"}</div>
                  </div>
                  <div className="rounded-xl bg-background/60 p-3">
                    <div className="text-foreground/50 text-xs">Subjects</div>
                    <div>{s?.n_subjects ?? "—"}</div>
                  </div>
                </div>
              </div>

              <a
                href="https://github.com"
                className="group flex items-center justify-between gap-4 rounded-[1.75rem] bg-lime p-5 text-ink transition hover:opacity-90"
              >
                <div>
                  <div className="font-medium tracking-tight">Open source</div>
                  <div className="text-xs text-ink/70">utils, backend, dashboard — all MIT</div>
                </div>
                <ScanEye size={20} />
              </a>
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}
