import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  Users, Search, ArrowUpRight, Sparkles,
} from "lucide-react";
import { api, fmtNum, qualityTone } from "@/lib/api";
import { Sidebar } from "./index";

export const Route = createFileRoute("/subjects/")({
  component: SubjectsPage,
  head: () => ({ meta: [{ title: "Subjects · Iris Recognition Console" }] }),
});

const SORTS: { key: string; label: string }[] = [
  { key: "consistency_asc",  label: "Most consistent" },
  { key: "consistency_desc", label: "Least consistent" },
  { key: "codes_desc",       label: "Most samples" },
  { key: "id_asc",           label: "Subject ID" },
];

function SubjectsPage() {
  const [sort, setSort] = useState("consistency_asc");
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState(50);

  const subjQ = useQuery({
    queryKey: ["subjects", sort, q, limit],
    queryFn: () => api.subjects({ sort, q: q || undefined, limit }),
  });

  const rows = subjQ.data?.items ?? [];
  const total = subjQ.data?.count ?? 0;

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="mx-auto flex max-w-[1500px] gap-6">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <div className="flex flex-wrap items-end justify-between gap-6">
            <div>
              <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-foreground/50">
                <Users size={14} /> Explorer
              </div>
              <h1 className="mt-2 font-display text-3xl font-medium leading-[1.05] tracking-tighter sm:text-4xl md:text-5xl xl:text-[56px]">
                Subjects{" "}
                <span className="inline-grid h-10 w-14 translate-y-0.5 place-items-center rounded-full bg-lime align-middle md:h-12 md:w-16">
                  <Sparkles size={20} strokeWidth={1.75} />
                </span>
              </h1>
              <div className="mt-2 text-sm text-foreground/60">
                {subjQ.isPending ? "loading…" : `${rows.length} of ${total} subjects`}
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-2 rounded-full bg-surface px-4 py-2">
                <Search size={16} className="text-foreground/50" />
                <input
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="filter subject id…"
                  className="w-44 bg-transparent text-sm outline-none placeholder:text-foreground/40"
                />
              </div>
              <div className="flex items-center gap-1 rounded-full bg-surface p-1">
                {SORTS.map((s) => (
                  <button
                    key={s.key}
                    onClick={() => setSort(s.key)}
                    className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                      sort === s.key ? "bg-ink text-white" : "text-foreground/70 hover:bg-secondary"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-8 overflow-hidden rounded-[2rem] bg-surface">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-background/40 text-xs uppercase tracking-wider text-foreground/50">
                  <tr>
                    <th className="px-6 py-3">Subject</th>
                    <th className="px-3 py-3">Quality</th>
                    <th className="px-3 py-3">Intra-subject HD</th>
                    <th className="px-3 py-3">Samples</th>
                    <th className="px-6 py-3 text-right"></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const tone = qualityTone(r.quality);
                    return (
                      <tr key={r.subject_id} className="border-t border-foreground/5 transition hover:bg-background/30">
                        <td className="px-6 py-3 font-medium tracking-tight">#{r.subject_id}</td>
                        <td className="px-3 py-3">
                          <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${tone.bg} ${tone.text} ${tone.ring}`}>
                            <span className="h-1.5 w-1.5 rounded-full bg-current" />
                            {r.quality}
                          </span>
                        </td>
                        <td className="px-3 py-3 tabular-nums text-foreground/80">{fmtNum(r.intra_hd, 4)}</td>
                        <td className="px-3 py-3 tabular-nums text-foreground/70">{r.n_codes}</td>
                        <td className="px-6 py-3 text-right">
                          <Link
                            to="/subjects/$subjectId"
                            params={{ subjectId: r.subject_id }}
                            className="inline-flex items-center gap-1 rounded-full bg-background px-3 py-1.5 text-xs font-medium hover:bg-secondary"
                          >
                            Open <ArrowUpRight size={12} />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                  {!subjQ.isPending && rows.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-6 py-10 text-center text-sm text-foreground/50">
                        No subjects match the current filter.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            {total > limit && (
              <div className="border-t border-foreground/5 p-4 text-center text-sm">
                <button
                  onClick={() => setLimit((l) => l + 50)}
                  className="rounded-full bg-background px-4 py-2 hover:bg-secondary"
                >
                  Load 50 more
                </button>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
