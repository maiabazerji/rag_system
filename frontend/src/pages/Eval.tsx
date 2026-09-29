/**
 * The Evaluation page: every recorded evaluation run in one place, as two tabs.
 * Overview compares the latest run of each configuration; Runs launches a run
 * and lists the history with per-run regression reports. The active tab lives
 * in the URL (`?tab=overview|runs`) so it can be linked and survives a reload.
 */
import { Suspense, lazy, useRef, type KeyboardEvent } from "react";
import { useSearchParams } from "react-router-dom";
import LoadingSpinner from "../components/LoadingSpinner";
import EvalRuns from "../components/eval/EvalRuns";
import { useI18n, type MessageKey } from "../i18n";

// Loaded on demand: the Overview is the only view that ships the charting library.
const EvalOverview = lazy(() => import("../components/eval/EvalOverview"));

const EVAL_TABS = ["overview", "runs"] as const;
type EvalTab = (typeof EVAL_TABS)[number];

const TAB_LABEL: Record<EvalTab, MessageKey> = {
  overview: "eval.tab.overview",
  runs: "eval.tab.runs",
};

/** The tab named in the URL, defaulting to the Overview for a missing or unknown value. */
function tabFromParam(value: string | null): EvalTab {
  return (EVAL_TABS as readonly string[]).includes(value ?? "") ? (value as EvalTab) : "overview";
}

export default function EvalPage() {
  const { t } = useI18n();
  const [params, setParams] = useSearchParams();
  const active = tabFromParam(params.get("tab"));
  const tabRefs = useRef<Record<EvalTab, HTMLButtonElement | null>>({ overview: null, runs: null });

  const select = (tab: EvalTab, focus = false) => {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set("tab", tab);
        return next;
      },
      { replace: true },
    );
    if (focus) tabRefs.current[tab]?.focus();
  };

  // Roving focus per the WAI-ARIA tabs pattern: arrows move and activate, Home/End jump.
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    const i = EVAL_TABS.indexOf(active);
    const last = EVAL_TABS.length - 1;
    const target =
      e.key === "ArrowRight" ? (i === last ? 0 : i + 1)
      : e.key === "ArrowLeft" ? (i === 0 ? last : i - 1)
      : e.key === "Home" ? 0
      : e.key === "End" ? last
      : null;
    if (target == null) return;
    e.preventDefault();
    select(EVAL_TABS[target], true);
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="display text-4xl font-semibold text-white">{t("eval.title")}</h1>
        <p className="text-zinc-400 max-w-2xl">{t("eval.subtitle")}</p>
      </header>

      <div role="tablist" aria-label={t("eval.tabs")} className="flex gap-1 border-b border-bg-border">
        {EVAL_TABS.map((tab) => {
          const selected = tab === active;
          return (
            <button
              key={tab}
              ref={(el) => {
                tabRefs.current[tab] = el;
              }}
              type="button"
              role="tab"
              id={`eval-tab-${tab}`}
              aria-selected={selected}
              aria-controls={`eval-panel-${tab}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => select(tab)}
              onKeyDown={onKeyDown}
              className={`px-4 py-2 -mb-px text-sm border-b-2 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
                selected ? "border-accent text-white" : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {t(TAB_LABEL[tab])}
            </button>
          );
        })}
      </div>

      <div role="tabpanel" id={`eval-panel-${active}`} aria-labelledby={`eval-tab-${active}`} tabIndex={0} className="focus:outline-none">
        {active === "overview" ? (
          <Suspense fallback={<LoadingSpinner />}>
            <EvalOverview />
          </Suspense>
        ) : (
          <EvalRuns />
        )}
      </div>
    </div>
  );
}
