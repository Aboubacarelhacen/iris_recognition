import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Users, ArrowLeft, Layers, Activity } from "lucide-react";
import { api, fmtNum, qualityTone } from "@/lib/api";
import { Sidebar } from "./index";

export const Route = createFileRoute("/subjects/$subjectId")({
  component: SubjectDetailPage,
  head: ({ params }) => ({
    meta: [{ title: `Subject #${params.subjectId} · Iris Recognition Console` }],
  }),
});

function SubjectDetailPage() {
  const { subjectId } = Route.useParams();
  const q = useQuery({
    queryKey: ["subject", subjectId],
    queryFn: () => api.subject(subjectId),
  });

  const d = q.data;
  const tone = d ? qualityTone(d.quality) : qualityTone("unknown");

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <div className="mx-auto flex max-w-[1500px] gap-6">
        <Sidebar />

        <main className="min-w-0 flex-1">
          <Link
            to="/subjects"
            className="inline-flex items-center gap-2 rounded-full bg-surface px-4 py-2 text-sm hover:bg-secondary"
          >
            <ArrowLeft size={14} /> All subjects
          </Link>

          <div className="mt-6 flex flex-wrap items-end justify-between gap-6">
            <div>
              <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-foreground/50">
                <Users size={14} /> Subject
              </div>
              <h1 className="mt-2 font-display text-4xl font-medium leading-[1.05] tracking-tighter sm:text-5xl xl:text-[64px]">
                #{subjectId}
              </h1>
            </div>
            {d && (
              <div className="flex items-center gap-3">
                <span className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ring-1 ${tone.bg} ${tone.text} ${tone.ring}`}>
                  <span className="h-1.5 w-1.5 rounded-full bg-current" />
                  {d.quality} consistency
                </span>
              </div>
            )}
          </div>

          {q.isError && (
            <div className="mt-6 rounded-2xl bg-[color:var(--risk-high)]/10 p-4 text-sm text-[color:var(--risk-high)]">
              {(q.error as Error).message}
            </div>
          )}

          {d && (
            <>
              <div className="mt-8 grid grid-cols-1 gap-5 md:grid-cols-3">
                <div className="rounded-[2rem] bg-surface p-6">
                  <div className="flex items-center gap-3">
                    <div className="grid h-11 w-11 place-items-center rounded-2xl bg-background"><Layers size={18} /></div>
                    <span className="text-lg font-medium">Samples</span>
                  </div>
                  <div className="mt-4 font-display text-5xl tracking-tighter">{d.n_codes}</div>
                  <div className="mt-1 text-sm text-foreground/60">IrisCodes encoded</div>
                </div>
                <div className="rounded-[2rem] bg-surface p-6">
                  <div className="flex items-center gap-3">
                    <div className="grid h-11 w-11 place-items-center rounded-2xl bg-background"><Activity size={18} /></div>
                    <span className="text-lg font-medium">Intra-subject HD</span>
                  </div>
                  <div className="mt-4 font-display text-5xl tracking-tighter">
                    {fmtNum(d.intra_hd, 4)}
                  </div>
                  <div className="mt-1 text-sm text-foreground/60">avg Hamming distance among own codes</div>
                </div>
                <div className="rounded-[2rem] bg-ink p-6 text-white">
                  <div className="text-xs uppercase tracking-widest text-white/60">Reading</div>
                  <div className="mt-2 font-display text-2xl tracking-tight">
                    Lower HD = a more <span className="text-lime">consistent</span> iris signal.
                  </div>
                  <div className="mt-3 text-sm text-white/70">
                    Subjects under 0.20 typically verify cleanly; above 0.25 indicates
                    segmentation noise or out-of-focus samples.
                  </div>
                </div>
              </div>

              <div className="mt-8 rounded-[2rem] bg-surface p-6">
                <div className="flex items-center justify-between">
                  <h3 className="font-display text-2xl font-medium tracking-tight">IrisCodes</h3>
                  <div className="text-xs text-foreground/50">showing first {d.codes.length}</div>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-6">
                  {d.codes.map((c, i) => (
                    <div key={i} className="rounded-2xl bg-background/60 p-3">
                      {c.code_png ? (
                        <img
                          src={c.code_png}
                          alt={`code ${i}`}
                          className="h-16 w-full rounded-lg object-fill"
                          style={{ imageRendering: "pixelated" }}
                        />
                      ) : (
                        <div className="h-16 w-full rounded-lg bg-background" />
                      )}
                      <div className="mt-2 line-clamp-1 text-[10px] tracking-tight text-foreground/50">
                        {c.image_path.split("/").slice(-1)[0]}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
