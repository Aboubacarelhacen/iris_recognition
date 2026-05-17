import { createFileRoute } from "@tanstack/react-router";
import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import {
  ScanEye, Upload, CheckCircle2, XCircle, RotateCcw, ImageIcon,
  Sparkles, Cpu, AlertTriangle,
} from "lucide-react";
import { api, fmtNum, fmtPct, type VerifyResult } from "@/lib/api";
import { Sidebar } from "./index";

export const Route = createFileRoute("/verify")({
  component: VerifyPage,
  head: () => ({
    meta: [{ title: "Verify · Iris Recognition Console" }],
  }),
});

type Matcher = "both" | "daugman" | "cnn";

function ImageDropzone({
  label, file, onFile,
}: { label: string; file: File | null; onFile: (f: File | null) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const preview = file ? URL.createObjectURL(file) : null;
  return (
    <div className="rounded-[2rem] bg-surface p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-background"><ImageIcon size={18} /></div>
          <span className="font-medium tracking-tight">{label}</span>
        </div>
        {file && (
          <button
            onClick={() => onFile(null)}
            className="rounded-full bg-background p-1.5 text-foreground/60 hover:text-foreground"
            title="Remove"
          >
            <RotateCcw size={14} />
          </button>
        )}
      </div>

      <button
        onClick={() => inputRef.current?.click()}
        className="mt-4 grid aspect-square w-full place-items-center overflow-hidden rounded-2xl border border-dashed border-foreground/20 bg-background/40 transition hover:bg-background/60"
      >
        {preview ? (
          <img src={preview} alt="iris" className="h-full w-full object-cover" />
        ) : (
          <div className="flex flex-col items-center gap-2 text-foreground/50">
            <Upload size={28} strokeWidth={1.5} />
            <span className="text-sm">Click to upload</span>
            <span className="text-xs">JPG · PNG · BMP</span>
          </div>
        )}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      {file && (
        <div className="mt-3 line-clamp-1 text-xs text-foreground/50">{file.name}</div>
      )}
    </div>
  );
}

function VerdictBox({
  match, label, score, threshold,
}: { match: boolean; label: string; score: number; threshold: number }) {
  return (
    <div
      className={`rounded-[2rem] p-6 ring-1 ${
        match
          ? "bg-[color:var(--risk-low)]/10 ring-[color:var(--risk-low)]/30"
          : "bg-[color:var(--risk-high)]/10 ring-[color:var(--risk-high)]/30"
      }`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-foreground/60">{label}</div>
          <div
            className={`mt-1 font-display text-3xl font-medium tracking-tight ${
              match ? "text-[color:var(--risk-low)]" : "text-[color:var(--risk-high)]"
            }`}
          >
            {match ? "SAME PERSON" : "DIFFERENT PERSON"}
          </div>
        </div>
        {match ? (
          <CheckCircle2 size={28} className="text-[color:var(--risk-low)]" />
        ) : (
          <XCircle size={28} className="text-[color:var(--risk-high)]" />
        )}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl bg-background/60 p-3">
          <div className="text-xs text-foreground/50">Distance</div>
          <div className="font-display text-2xl tracking-tight">{fmtNum(score, 4)}</div>
        </div>
        <div className="rounded-xl bg-background/60 p-3">
          <div className="text-xs text-foreground/50">Threshold</div>
          <div className="font-display text-2xl tracking-tight">{fmtNum(threshold, 3)}</div>
        </div>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-background/80">
        <div
          className={`h-full rounded-full ${match ? "bg-[color:var(--risk-low)]" : "bg-[color:var(--risk-high)]"}`}
          style={{ width: `${Math.min(100, (1 - score / Math.max(threshold * 2, 0.001)) * 100)}%` }}
        />
      </div>
    </div>
  );
}

function ArtifactRow({ title, src }: { title: string; src: string }) {
  return (
    <div className="rounded-2xl bg-background/60 p-4">
      <div className="mb-2 text-xs uppercase tracking-widest text-foreground/50">{title}</div>
      {src ? (
        <img src={src} alt={title} className="h-20 w-full rounded-lg object-fill" style={{ imageRendering: "pixelated" }} />
      ) : (
        <div className="grid h-20 w-full place-items-center rounded-lg bg-background text-xs text-foreground/40">unavailable</div>
      )}
    </div>
  );
}

function VerifyPage() {
  const [a, setA] = useState<File | null>(null);
  const [b, setB] = useState<File | null>(null);
  const [matcher, setMatcher] = useState<Matcher>("both");

  const mut = useMutation({
    mutationFn: ({ a, b }: { a: File; b: File }): Promise<VerifyResult> => api.verify(a, b),
  });

  const result = mut.data;
  const canRun = a && b && !mut.isPending;

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="mx-auto flex max-w-[1500px] gap-6">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <div className="flex flex-wrap items-end justify-between gap-6">
            <div>
              <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-foreground/50">
                <ScanEye size={14} /> Verification
              </div>
              <h1 className="mt-2 font-display text-3xl font-medium leading-[1.05] tracking-tighter sm:text-4xl md:text-5xl xl:text-[56px]">
                Compare two{" "}
                <span className="inline-grid h-10 w-14 translate-y-0.5 place-items-center rounded-full bg-lime align-middle md:h-12 md:w-16">
                  <Sparkles size={20} strokeWidth={1.75} />
                </span>{" "}
                iris images
              </h1>
            </div>
            <div className="flex items-center gap-2 rounded-full bg-surface p-1">
              {(["both", "daugman", "cnn"] as Matcher[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMatcher(m)}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                    matcher === m ? "bg-ink text-white" : "text-foreground/70 hover:bg-secondary"
                  }`}
                >
                  {m === "both" ? "Both" : m === "daugman" ? "Daugman" : "CNN"}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-[1fr_1fr_auto]">
            <ImageDropzone label="Image A" file={a} onFile={setA} />
            <ImageDropzone label="Image B" file={b} onFile={setB} />
            <div className="flex flex-col gap-3 rounded-[2rem] bg-ink p-5 text-white">
              <div className="text-xs uppercase tracking-widest text-white/60">Matcher</div>
              <div className="font-display text-2xl tracking-tight">
                {matcher === "both" ? "Daugman + CNN" : matcher === "daugman" ? "Daugman" : "CNN"}
              </div>
              <button
                disabled={!canRun}
                onClick={() => a && b && mut.mutate({ a, b })}
                className="mt-auto inline-flex items-center justify-center gap-2 rounded-full bg-lime px-5 py-3 text-sm font-semibold text-ink transition hover:opacity-90 disabled:opacity-40"
              >
                {mut.isPending ? "Verifying…" : (<><ScanEye size={16} /> Verify identity</>)}
              </button>
              {mut.isError && (
                <div className="rounded-xl bg-white/10 p-3 text-xs">
                  <div className="flex items-center gap-1 font-semibold text-amber-300">
                    <AlertTriangle size={14} /> Failed
                  </div>
                  <div className="mt-1 text-white/70 line-clamp-3">{(mut.error as Error).message}</div>
                </div>
              )}
            </div>
          </div>

          {result && (
            <div className="mt-8 flex flex-col gap-6">
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                {(matcher === "both" || matcher === "daugman") && (
                  <VerdictBox
                    label="Daugman · Hamming Distance"
                    match={result.daugman.match}
                    score={result.daugman.hd}
                    threshold={result.daugman.threshold}
                  />
                )}
                {(matcher === "both" || matcher === "cnn") && (
                  result.cnn ? (
                    <VerdictBox
                      label="CNN · Cosine Distance"
                      match={result.cnn.match}
                      score={result.cnn.cosine_distance}
                      threshold={result.cnn.threshold}
                    />
                  ) : (
                    <div className="rounded-[2rem] bg-surface p-6 text-sm text-foreground/60">
                      <div className="flex items-center gap-2 font-semibold text-foreground">
                        <Cpu size={16} /> CNN unavailable
                      </div>
                      <div className="mt-2">
                        Place <code>model/cnn_iris.pt</code> and restart the backend to enable.
                      </div>
                    </div>
                  )
                )}
              </div>

              {matcher === "both" && result.cnn && (
                <div
                  className={`rounded-2xl p-4 text-center text-sm font-medium ${
                    result.daugman.match === result.cnn.match
                      ? "bg-[color:var(--risk-low)]/10 text-[color:var(--risk-low)]"
                      : "bg-[color:var(--risk-medium)]/10 text-[color:var(--risk-medium)]"
                  }`}
                >
                  {result.daugman.match === result.cnn.match
                    ? "Both methods agree on the verdict."
                    : "Methods disagree — CNN is the stronger signal on this dataset."}
                </div>
              )}

              <div className="rounded-[2rem] bg-surface p-6">
                <div className="text-xs uppercase tracking-widest text-foreground/50">Pipeline artifacts</div>
                <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-4">
                  <ArtifactRow title="Strip A (64 × 512)" src={result.artifacts.strip_a} />
                  <ArtifactRow title="Strip B (64 × 512)" src={result.artifacts.strip_b} />
                  <ArtifactRow title="IrisCode A" src={result.artifacts.code_a} />
                  <ArtifactRow title="IrisCode B" src={result.artifacts.code_b} />
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
