import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Eye, Aperture, Users, Cpu, Bot, FileText, BookOpen, Settings,
  ScanEye, ArrowUpRight, MoreHorizontal, BarChart3, ShieldCheck, GitMerge,
  HelpCircle, AlertTriangle, Activity, Layers,
} from "lucide-react";
import { api, fmtPct, fmtNum, qualityTone, type Quality, type TopSubject } from "@/lib/api";

export const Route = createFileRoute("/")({
  component: Dashboard,
  head: () => ({
    meta: [
      { title: "Iris Recognition Console" },
      { name: "description", content: "Daugman IrisCode + ResNet18/ArcFace iris recognition dashboard." },
    ],
  }),
});

// ───────────────────────────────────────────────────────────── sidebar
function SidebarLink({
  icon: Icon, label, to, exact = false,
}: { icon: any; label: string; to: string; exact?: boolean }) {
  return (
    <Link
      to={to}
      activeOptions={{ exact }}
      title={label}
      className="grid h-11 w-11 place-items-center rounded-2xl text-white/60 transition hover:bg-white/5 hover:text-white aria-[current=page]:bg-white/10 aria-[current=page]:text-white"
    >
      <Icon size={20} strokeWidth={1.75} />
    </Link>
  );
}

export function Sidebar() {
  return (
    <aside className="sticky top-4 flex h-[calc(100vh-2rem)] w-20 flex-col items-center justify-between rounded-[2rem] bg-ink py-5 text-white">
      <div className="flex flex-col items-center gap-2">
        <Link
          to="/"
          title="Iris Recognition Console"
          className="group relative grid h-11 w-11 place-items-center rounded-2xl bg-white/10 transition hover:bg-white/15"
        >
          <Aperture size={22} className="text-lime transition group-hover:rotate-12" strokeWidth={1.75} />
        </Link>
        <div className="mt-4 flex flex-col gap-1.5">
          <SidebarLink icon={Activity}  label="Dashboard"     to="/" exact />
          <SidebarLink icon={ScanEye}   label="Verify identity" to="/verify" />
          <SidebarLink icon={Users}     label="Subjects"      to="/subjects" />
          <SidebarLink icon={Layers}    label="System info"   to="/system" />
        </div>
      </div>
      <div className="flex flex-col items-center gap-1.5">
        <a href="https://github.com" title="GitHub" className="grid h-11 w-11 place-items-center rounded-2xl text-white/60 hover:bg-white/5 hover:text-white">
          <BookOpen size={20} strokeWidth={1.75} />
        </a>
        <button title="Settings" className="grid h-11 w-11 place-items-center rounded-2xl text-white/60 hover:bg-white/5 hover:text-white">
          <Settings size={20} strokeWidth={1.75} />
        </button>
        <div className="mt-2 grid h-11 w-11 place-items-center rounded-full bg-gradient-to-br from-amber-300 to-rose-400 text-ink ring-2 ring-white/20">
          <Eye size={20} strokeWidth={2} />
        </div>
      </div>
    </aside>
  );
}

// ───────────────────────────────────────────────────────────── building blocks
function ProgressDots({ filled, total = 7 }: { filled: number; total?: number }) {
  return (
    <div className="flex items-center gap-1.5">
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          className={`h-9 w-6 rounded-full ${i < filled ? "bg-ink" : "border border-dashed border-foreground/25"}`}
        />
      ))}
    </div>
  );
}

function StatCard({
  icon: Icon, title, value, sub, pct, accent = false, compact = false,
}: {
  icon: any; title: string; value: string; sub?: string; pct: number;
  accent?: boolean; compact?: boolean;
}) {
  const filled = Math.max(0, Math.min(7, Math.round((pct / 100) * 7)));
  return (
    <div
      className={`relative flex min-w-0 flex-col overflow-hidden rounded-[2rem] ${
        compact ? "gap-4 p-5" : "gap-6 p-6"
      } ${accent ? "bg-lime" : "bg-surface"}`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-3">
          <div className={`grid place-items-center rounded-2xl bg-background/80 ${compact ? "h-9 w-9" : "h-11 w-11"}`}>
            <Icon size={compact ? 16 : 18} strokeWidth={2} />
          </div>
          <span className={`truncate font-medium tracking-tight ${compact ? "text-sm" : "text-lg"}`}>
            {title}
          </span>
        </div>
        {!compact && (
          <button className="shrink-0 text-foreground/50 hover:text-foreground"><MoreHorizontal size={20} /></button>
        )}
      </div>

      <div className={`flex min-w-0 flex-wrap items-end ${compact ? "gap-2" : "gap-3"}`}>
        <span
          className={`min-w-0 font-display font-medium leading-none tracking-tighter ${
            compact ? "text-4xl" : "text-6xl"
          }`}
        >
          {value}
        </span>
        {sub && (
          <span className={`mb-1 truncate text-xs text-foreground/60 ${compact ? "" : "md:text-sm"}`}>
            {sub}
          </span>
        )}
        <div
          className={`mb-1 inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-[10px] font-semibold ${
            accent ? "bg-surface" : "bg-lime"
          } ${compact ? "" : "ml-1 gap-1.5 px-2.5 text-xs"}`}
        >
          {Math.round(pct)}%
          <span className="grid h-3 w-3 place-items-center rounded-full border border-ink/60">
            <span className="h-1 w-1 rounded-full bg-ink" />
          </span>
        </div>
      </div>

      <ProgressDots filled={filled} total={compact ? 5 : 7} />
    </div>
  );
}

function ModelHealthCard({ cnnAuc, cnnEer, daugmanEer }: { cnnAuc: number; cnnEer: number; daugmanEer: number }) {
  return (
    <div className="relative flex flex-col justify-between overflow-hidden rounded-[2rem] bg-ink p-6 text-white">
      <div className="absolute inset-y-0 right-0 w-1/2 bg-gradient-to-l from-fuchsia-500/30 via-indigo-500/20 to-transparent" />
      <div className="absolute right-2 top-2 bottom-2 w-1/2 rounded-[1.5rem] bg-[radial-gradient(circle_at_70%_40%,rgba(236,72,153,0.45),transparent_55%),radial-gradient(circle_at_50%_80%,rgba(99,102,241,0.4),transparent_60%),linear-gradient(135deg,#1a1a1f,#3a2740)]" />
      <div className="relative">
        <div className="text-xs uppercase tracking-widest text-white/60">Model health</div>
        <h3 className="mt-2 max-w-[14ch] font-display text-3xl font-medium leading-[1.05] tracking-tight">
          CNN AUC <span className="text-lime">{fmtNum(cnnAuc)}</span>
        </h3>
        <div className="mt-3 text-sm text-white/70">
          CNN EER {fmtPct(cnnEer)} · beats Daugman {fmtPct(daugmanEer)} baseline
        </div>
      </div>
      <Link
        to="/verify"
        className="relative inline-flex w-fit items-center gap-2 rounded-full bg-white px-5 py-3 text-sm font-semibold text-ink transition hover:bg-white/90"
      >
        Try verification
        <span className="grid h-6 w-6 place-items-center rounded-full bg-ink text-white">
          <ArrowUpRight size={11} />
        </span>
      </Link>
    </div>
  );
}

function RightCard({
  icon: Icon, title, sub, to, href, accent = false,
}: { icon: any; title: string; sub?: string; to?: string; href?: string; accent?: boolean }) {
  const inner = (
    <>
      <div className="flex items-start justify-between">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-background">
          <Icon size={18} strokeWidth={1.75} />
        </div>
        <ArrowUpRight size={18} className="text-foreground/40 transition group-hover:text-foreground" />
      </div>
      <div>
        <div className="font-medium tracking-tight">{title}</div>
        {sub && <div className="mt-0.5 line-clamp-2 text-xs text-foreground/50">{sub}</div>}
      </div>
    </>
  );
  const classes = `group flex flex-col gap-4 rounded-[1.75rem] p-5 text-left transition ${
    accent ? "bg-lime hover:bg-lime/90" : "bg-surface hover:bg-secondary"
  }`;
  if (to)  return <Link to={to}  className={classes}>{inner}</Link>;
  if (href) return <a   href={href} className={classes}>{inner}</a>;
  return <button className={classes}>{inner}</button>;
}

function TopSubjectsChart({ items }: { items: TopSubject[] }) {
  if (!items.length) return null;
  const max = Math.max(...items.map((i) => i.intra_hd), 0.001);
  return (
    <div className="rounded-[2rem] bg-surface p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-2xl bg-background"><BarChart3 size={18} /></div>
          <h3 className="font-display text-3xl font-medium tracking-tight">Most consistent subjects</h3>
          <div className="ml-4 flex items-center gap-5 text-sm">
            <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-ink" />Intra-subject HD</span>
            <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-lime" />Samples</span>
          </div>
        </div>
        <Link
          to="/subjects"
          className="inline-flex items-center gap-2 rounded-full bg-background px-4 py-2 text-sm font-medium hover:bg-secondary"
        >
          Browse all subjects <ArrowUpRight size={14} />
        </Link>
      </div>

      <div className="mt-8 flex gap-6 overflow-x-auto">
        <div className="flex flex-col justify-between py-2 text-xs text-foreground/50">
          {[max, max * 0.8, max * 0.6, max * 0.4, max * 0.2, 0].map((v, i) => (
            <span key={i}>{v.toFixed(2)}</span>
          ))}
        </div>
        <div className="relative min-w-[420px] flex-1">
          <div className="absolute inset-0 flex flex-col justify-between">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-px w-full bg-foreground/5" />
            ))}
          </div>
          <div
            className="relative grid h-[320px] items-end gap-3 px-2"
            style={{ gridTemplateColumns: `repeat(${items.length}, minmax(0, 1fr))` }}
          >
            {items.map((d) => {
              const hdH = (d.intra_hd / max) * 100;
              const cnt = Math.min(1, d.n_codes / 30);
              return (
                <Link
                  key={d.subject_id}
                  to="/subjects/$subjectId"
                  params={{ subjectId: d.subject_id }}
                  className="relative flex h-full items-end justify-center"
                >
                  <div className="relative w-10">
                    <div className="relative mx-auto w-10 rounded-full bg-ink" style={{ height: `${hdH * 2.8}px` }}>
                      <span className="absolute -top-1 left-1/2 h-2.5 w-2.5 -translate-x-1/2 rounded-full bg-white ring-2 ring-ink" />
                    </div>
                    <div className="absolute bottom-0 left-1/2 w-10 -translate-x-1/2 rounded-full bg-lime" style={{ height: `${cnt * 280}px` }}>
                      <span className="absolute -bottom-1 left-1/2 h-2.5 w-2.5 -translate-x-1/2 rounded-full bg-ink" />
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
          <div
            className="mt-4 grid gap-3 px-2 text-center text-xs text-foreground/60"
            style={{ gridTemplateColumns: `repeat(${items.length}, minmax(0, 1fr))` }}
          >
            {items.map((d) => (<div key={d.subject_id} className="truncate">#{d.subject_id}</div>))}
          </div>
        </div>
      </div>
    </div>
  );
}

function QualityChips({ buckets }: { buckets: Record<Quality, number> }) {
  const order: Quality[] = ["high", "medium", "low", "unknown"];
  const labels: Record<Quality, string> = {
    high: "High consistency", medium: "Medium consistency",
    low: "Low consistency", unknown: "Unknown",
  };
  return (
    <div className="flex flex-wrap items-center gap-2">
      {order.map((k) => {
        const tone = qualityTone(k);
        return (
          <span
            key={k}
            className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm ring-1 ${tone.bg} ${tone.text} ${tone.ring}`}
          >
            <span className="h-2 w-2 rounded-full bg-current" />
            {labels[k]}
            <span className="rounded-full bg-background px-2 py-0.5 text-xs font-semibold tabular-nums text-foreground">
              {buckets[k] ?? 0}
            </span>
          </span>
        );
      })}
    </div>
  );
}

// ───────────────────────────────────────────────────────────── dashboard
function Dashboard() {
  const summaryQ = useQuery({ queryKey: ["summary"], queryFn: api.summary });
  const s = summaryQ.data;

  const cnnAuc = s?.cnn?.auc ?? 0;
  const cnnEer = s?.cnn?.eer ?? 0;
  const daugmanEer = s?.daugman?.eer ?? 0;
  const cnnHealthPct = Math.round((cnnAuc || 0) * 100);
  const eerImprovementPct =
    daugmanEer && cnnEer
      ? Math.max(0, Math.round(((daugmanEer - cnnEer) / daugmanEer) * 100))
      : 0;

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="mx-auto flex max-w-[1500px] gap-6">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <h1 className="font-display text-3xl font-medium leading-[1.05] tracking-tighter sm:text-4xl md:text-5xl xl:text-[64px]">
              Iris{" "}
              <span className="inline-grid h-10 w-10 translate-y-0.5 place-items-center rounded-full bg-surface align-middle md:h-12 md:w-12">
                <Eye size={20} strokeWidth={1.75} />
              </span>{" "}
              Recognition{" "}
              <span className="inline-grid h-10 w-14 translate-y-0.5 place-items-center rounded-full bg-lime align-middle md:h-12 md:w-16">
                <ShieldCheck size={20} strokeWidth={1.75} />
              </span>{" "}
              Console
            </h1>
            <div className="flex items-center gap-3">
              <button className="grid h-12 w-12 place-items-center rounded-full bg-surface text-foreground hover:bg-secondary">
                <Settings size={18} />
              </button>
              <Link
                to="/verify"
                className="inline-flex items-center gap-2 rounded-full bg-ink px-5 py-3.5 text-sm font-semibold text-white hover:opacity-90"
              >
                <ScanEye size={18} /> Verify
              </Link>
            </div>
          </div>

          {summaryQ.isError && (
            <div className="mt-6 rounded-2xl border border-[color:var(--risk-high)]/40 bg-[color:var(--risk-high)]/10 p-4 text-sm">
              <div className="flex items-center gap-2 font-semibold text-[color:var(--risk-high)]">
                <AlertTriangle size={16} /> Backend not reachable
              </div>
              <div className="mt-1 text-foreground/70">
                Start the FastAPI server:{" "}
                <code className="rounded bg-background px-1.5 py-0.5">uvicorn backend.main:app --reload --port 8000</code>
              </div>
            </div>
          )}

          <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
            <div className="flex flex-col gap-6">
              <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
                <StatCard
                  icon={Users}
                  title="Subjects"
                  value={s ? s.n_subjects.toLocaleString() : "—"}
                  sub={s ? `${s.n_codes.toLocaleString()} codes` : undefined}
                  pct={s ? Math.min(100, (s.n_subjects / 1000) * 100) : 0}
                />
                <StatCard
                  icon={Cpu}
                  title="CNN EER"
                  value={s ? fmtPct(cnnEer, 1) : "—"}
                  sub={s ? `thr ${fmtNum(s.cnn.threshold)}` : undefined}
                  pct={Math.max(0, 100 - Math.round((cnnEer || 0) * 100 * 3))}
                  accent
                />
                <ModelHealthCard cnnAuc={cnnAuc} cnnEer={cnnEer} daugmanEer={daugmanEer} />
              </div>

              <div className="rounded-[2rem] bg-surface p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="grid h-11 w-11 place-items-center rounded-2xl bg-background">
                      <Cpu size={18} />
                    </div>
                    <h3 className="font-display text-2xl font-medium tracking-tight">
                      End-to-end pipeline
                    </h3>
                  </div>
                  <Link
                    to="/system"
                    className="hidden items-center gap-2 rounded-full bg-background px-4 py-2 text-sm font-medium hover:bg-secondary sm:inline-flex"
                  >
                    Algorithm details <ArrowUpRight size={14} />
                  </Link>
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

              <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
                <div className="rounded-[2rem] bg-surface p-6">
                  <div className="text-xs uppercase tracking-widest text-foreground/50">Daugman IrisCode</div>
                  <div className="mt-2 font-display text-4xl tracking-tight">{fmtPct(daugmanEer, 1)}</div>
                  <div className="mt-1 text-sm text-foreground/60">EER on {s?.daugman?.num_images ?? "—"} imgs</div>
                  <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-xl bg-background/60 p-3">
                      <div className="text-foreground/50 text-xs">AUC</div>
                      <div className="font-medium">{fmtNum(s?.daugman?.auc)}</div>
                    </div>
                    <div className="rounded-xl bg-background/60 p-3">
                      <div className="text-foreground/50 text-xs">Threshold</div>
                      <div className="font-medium">{fmtNum(s?.daugman?.threshold)}</div>
                    </div>
                  </div>
                </div>
                <div className="rounded-[2rem] bg-surface p-6">
                  <div className="text-xs uppercase tracking-widest text-foreground/50">ResNet18 + ArcFace</div>
                  <div className="mt-2 font-display text-4xl tracking-tight">{fmtPct(cnnEer, 1)}</div>
                  <div className="mt-1 text-sm text-foreground/60">CNN EER on held-out subjects</div>
                  <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-xl bg-background/60 p-3">
                      <div className="text-foreground/50 text-xs">AUC</div>
                      <div className="font-medium">{fmtNum(s?.cnn?.auc)}</div>
                    </div>
                    <div className="rounded-xl bg-background/60 p-3">
                      <div className="text-foreground/50 text-xs">Threshold</div>
                      <div className="font-medium">{fmtNum(s?.cnn?.threshold)}</div>
                    </div>
                  </div>
                </div>
                <div className="rounded-[2rem] bg-ink p-6 text-white">
                  <div className="text-xs uppercase tracking-widest text-white/60">EER improvement</div>
                  <div className="mt-2 font-display text-4xl tracking-tight text-lime">
                    {eerImprovementPct ? `−${eerImprovementPct}%` : "—"}
                  </div>
                  <div className="mt-1 text-sm text-white/70">CNN relative to Daugman baseline</div>
                  <Link
                    to="/system"
                    className="mt-4 inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm hover:bg-white/20"
                  >
                    Read algorithm <ArrowUpRight size={12} />
                  </Link>
                </div>
              </div>
            </div>

            <aside className="flex min-w-0 flex-col gap-4">
              <div className="grid grid-cols-2 gap-4">
                <StatCard
                  compact
                  icon={GitMerge}
                  title="Models"
                  value={s ? String((s.daugman.available ? 1 : 0) + (s.cnn.available ? 1 : 0)) : "—"}
                  sub="loaded"
                  pct={cnnHealthPct}
                />
                <StatCard
                  compact
                  icon={Layers}
                  title="Codes"
                  value={s ? `${(s.n_codes / 1000).toFixed(1)}k` : "—"}
                  sub="encoded"
                  pct={Math.min(100, ((s?.n_codes ?? 0) / 20000) * 100)}
                />
              </div>
              <RightCard
                icon={ScanEye}
                title="Verify two iris images"
                sub="Upload + match with Daugman, CNN, or both"
                to="/verify"
                accent
              />
              <RightCard
                icon={Users}
                title="Browse subjects"
                sub={s ? `${s.n_subjects} subjects · ${s.n_codes} codes` : "loading…"}
                to="/subjects"
              />
              <RightCard
                icon={Bot}
                title="LLM analysis (Türkçe)"
                sub={s?.has_llm_analysis ? "llama3.2 — cached report ready" : "regenerate from /system"}
                to="/system"
              />
              <RightCard
                icon={FileText}
                title="Algorithm walkthrough"
                sub="Daugman pipeline + CNN training recipe"
                to="/system"
              />
              <RightCard
                icon={HelpCircle}
                title="Source code"
                sub="utils/ + backend/ + this dashboard"
                href="https://github.com"
              />
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}
