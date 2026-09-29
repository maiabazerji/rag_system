import { Fragment, type ReactNode } from "react";
import { useI18n } from "../i18n";
import { handleNumber, sourceAnchor, splitCitations } from "../utils/citations";

/** A cited chunk as the API returns it. Citation fields are absent on old payloads. */
export type CitedSource = {
  chunk_id: string;
  quote: string;
  document?: string | null;
  handle?: string | null;
  title?: string | null;
  section?: string | null;
  page?: number | null;
  relevance_score?: number | null;
};

export type GroundingStatus = "answered" | "partial" | "insufficient_context";

/** The grounding fields shared by /ask answers and /compare/strategies rows. */
export type Grounding = {
  grounded?: boolean;
  status?: GroundingStatus | null;
  invalid_citations?: string[];
  citation_count?: number;
  unsupported_notes?: string | null;
};

/** Answer text with each `[S#]` marker rendered as a superscript link to its source. */
export function CitedText({
  text,
  anchorPrefix,
  sources,
}: {
  text: string;
  anchorPrefix: string;
  sources: CitedSource[];
}) {
  const { t } = useI18n();
  const known = new Set(sources.map((s) => s.handle).filter(Boolean));
  return (
    <>
      {splitCitations(text).map((seg, i) => {
        if (seg.kind === "text") return <Fragment key={i}>{seg.text}</Fragment>;
        if (!known.has(seg.handle)) return <Fragment key={i}>[{seg.handle}]</Fragment>;
        return (
          <sup key={i} className="ml-0.5">
            <a
              href={`#${sourceAnchor(anchorPrefix, seg.handle)}`}
              onClick={() => openSource(anchorPrefix, seg.handle)}
              className="text-accent hover:underline font-mono text-[0.7em]"
              aria-label={t("grounding.citeLink", { handle: seg.handle })}
            >
              [{handleNumber(seg.handle)}]
            </a>
          </sup>
        );
      })}
    </>
  );
}

/** Open the <details> holding a source so the anchor jump lands on visible content. */
function openSource(prefix: string, handle: string) {
  const el = document.getElementById(sourceAnchor(prefix, handle));
  const details = el?.closest("details");
  if (details && !details.open) details.open = true;
}

const BADGE_STYLES = {
  good: "!text-emerald-300 !border-emerald-500/40 !bg-emerald-500/10",
  warn: "!text-amber-300 !border-amber-500/40 !bg-amber-500/10",
} as const;

/** Grounded / partial / weakly grounded / insufficient context. Nothing for old payloads. */
export function GroundingBadge({ g, className = "" }: { g: Grounding; className?: string }) {
  const { t } = useI18n();
  if (!g.status) return null;
  const [label, tip, tone] =
    g.status === "insufficient_context"
      ? (["grounding.insufficient", "grounding.insufficientTip", "warn"] as const)
      : g.status === "partial"
        ? (["grounding.partial", "grounding.partialTip", "warn"] as const)
        : g.grounded
          ? (["grounding.grounded", "grounding.groundedTip", "good"] as const)
          : (["grounding.weak", "grounding.weakTip", "warn"] as const);
  return (
    <span className={`chip ${BADGE_STYLES[tone]} ${className}`} title={t(tip)}>
      {t(label)}
    </span>
  );
}

/** Warns that the model cited sources it was never given (they were removed). */
export function InvalidCitationsWarning({ invalid }: { invalid?: string[] }) {
  const { tp } = useI18n();
  if (!invalid || invalid.length === 0) return null;
  return (
    <div role="status" className="text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-2 py-1">
      ⚠ {tp("grounding.invalidCitations", invalid.length, { handles: invalid.join(", ") })}
    </div>
  );
}

/** "Employee Handbook · Leave · p. 12": the readable location of a source. */
export function SourceLocation({ s }: { s: CitedSource }): ReactNode {
  const { t } = useI18n();
  const parts = [s.title || s.document, s.section, s.page != null ? t("grounding.page", { page: s.page }) : null].filter(
    (p): p is string => Boolean(p),
  );
  if (parts.length === 0) return null;
  return <span className="text-zinc-400">{parts.join(" · ")}</span>;
}
