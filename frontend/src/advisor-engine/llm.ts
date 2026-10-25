/**
 * "Bring your own key" profile extraction for the static demo, mirroring
 * backend/app/advisor/profile.py `extract_with_llm`: same tool, same system
 * prompt, same validation. The request goes from the browser straight to
 * Anthropic; the key never reaches any server of ours.
 */
import type { ProjectProfile, QuestionMix } from "../api/advisor";
import engine from "./generated/engine.json";
import { corpusBucket, normalizeLevel, normalizeMix } from "./profile";

const BYOK = engine.byok;
const ENDPOINT = "https://api.anthropic.com/v1/messages";
const KEY_STORAGE = "veritas.anthropicKey";

/** The visitor's Anthropic key, for this browser tab only (sessionStorage). */
export function getBrowserKey(): string | null {
  try {
    return sessionStorage.getItem(KEY_STORAGE);
  } catch {
    return null;
  }
}

export function setBrowserKey(key: string | null): void {
  try {
    if (key && key.trim()) sessionStorage.setItem(KEY_STORAGE, key.trim());
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    // Storage blocked (private mode): the key simply is not kept.
  }
}

const SIZES = ["tiny", "small", "medium", "large", "xlarge"];
const LEVELS = ["low", "medium", "high"];
const FRESHNESS = ["static", "monthly", "weekly", "daily", "realtime"];

function stringList(v: unknown, field: string): string[] {
  if (!Array.isArray(v) || !v.every((x) => typeof x === "string")) throw new Error(`invalid ${field}`);
  return v;
}

function oneOf(v: unknown, allowed: string[], field: string): string {
  if (typeof v !== "string" || !allowed.includes(v)) throw new Error(`invalid ${field}`);
  return v;
}

function optionalInt(v: unknown, field: string): number | null {
  if (v == null) return null;
  if (typeof v !== "number" || !Number.isInteger(v)) throw new Error(`invalid ${field}`);
  return v;
}

/** ProjectProfile validation of the tool input, as pydantic does it in Python. */
export function profileFromToolInput(input: Record<string, unknown>): ProjectProfile {
  const data = Object.fromEntries(Object.entries(input).filter(([, v]) => v != null));
  const docs = optionalInt(data.corpus_size_docs, "corpus_size_docs");
  const mixIn = (data.question_mix ?? {}) as Partial<QuestionMix>;
  for (const k of ["single_fact", "relational_multi_hop", "exploratory_multi_step"] as const) {
    if (mixIn[k] !== undefined && (typeof mixIn[k] !== "number" || mixIn[k]! < 0)) throw new Error("invalid question_mix");
  }
  return {
    corpus_size: (docs !== null ? corpusBucket(docs) : oneOf(data.corpus_size ?? "small", SIZES, "corpus_size")) as ProjectProfile["corpus_size"],
    corpus_size_docs: docs,
    document_types: stringList(data.document_types ?? [], "document_types"),
    languages: stringList(data.languages ?? ["en"], "languages"),
    question_mix: normalizeMix(mixIn),
    latency_budget_ms: optionalInt(data.latency_budget_ms, "latency_budget_ms"),
    cost_sensitivity: oneOf(normalizeLevel(data.cost_sensitivity ?? "medium"), LEVELS, "cost_sensitivity") as ProjectProfile["cost_sensitivity"],
    data_freshness: oneOf(data.data_freshness ?? "weekly", FRESHNESS, "data_freshness") as ProjectProfile["data_freshness"],
    compliance: stringList(data.compliance ?? [], "compliance"),
    data_residency: typeof data.data_residency === "string" ? data.data_residency : null,
    entity_richness: oneOf(normalizeLevel(data.entity_richness ?? "medium"), LEVELS, "entity_richness") as ProjectProfile["entity_richness"],
    example_questions: stringList(data.example_questions ?? [], "example_questions").slice(0, 10),
    source: "llm",
    overridden_fields: [],
  };
}

/**
 * Ask Claude for the profile with the visitor's key.
 *
 * Throws on any failure (network, HTTP error, no tool call, invalid input);
 * the caller falls back to the heuristic.
 */
export async function extractWithBrowserKey(description: string, apiKey: string): Promise<ProjectProfile> {
  const r = await fetch(ENDPOINT, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: BYOK.model,
      max_tokens: BYOK.max_tokens,
      system: BYOK.system,
      tools: [BYOK.tool],
      messages: [
        {
          role: "user",
          content:
            `Project description:\n<description>\n${description}\n</description>\n\n` +
            `Call ${BYOK.tool_name} with the profile.`,
        },
      ],
      ...BYOK.request_options,
    }),
  });
  if (!r.ok) throw new Error(`Anthropic API ${r.status}`);
  const body = (await r.json()) as { content?: { type?: string; name?: string; input?: unknown }[] };
  const block = (body.content ?? []).find((b) => b.type === "tool_use" && b.name === BYOK.tool_name);
  if (!block || typeof block.input !== "object" || block.input === null) {
    throw new Error(`model did not call ${BYOK.tool_name}`);
  }
  return profileFromToolInput(block.input as Record<string, unknown>);
}
