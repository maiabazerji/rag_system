/**
 * Inline citation markers. The backend validates every citation and rewrites
 * the survivors canonically as `[S1]`, `[S2]`...; each handle matches the
 * `handle` of one entry in the answer's `sources`.
 */
export type Segment = { kind: "text"; text: string } | { kind: "cite"; handle: string };

const MARKER = /\[(S\d+)\]/g;

/** Split answer text into plain text and citation markers, in order. */
export function splitCitations(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const start = m.index ?? 0;
    if (start > last) out.push({ kind: "text", text: text.slice(last, start) });
    out.push({ kind: "cite", handle: m[1] });
    last = start + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

/** The number shown for a handle: "S3" -> "3". */
export function handleNumber(handle: string): string {
  return handle.replace(/^S/, "");
}

/** A DOM id for a source, unique per answer thanks to `prefix`. */
export function sourceAnchor(prefix: string, handle: string): string {
  return `${prefix}-src-${handle}`;
}
