# Structured Outputs

Structured outputs constrain an LLM to emit data that conforms to a schema —
typically a JSON object validated against a Pydantic model. This turns an
otherwise free-form response into something the rest of the system can parse
and act on deterministically.

## Enforcement Strategies

- **Tool use / function calling**: the provider exposes a tool whose
  parameter schema is the target Pydantic model. Claude and OpenAI both
  return well-formed JSON for tool arguments.
- **Constrained decoding**: libraries like `instructor`, `outlines`, and
  `guidance` constrain token sampling to only emit tokens consistent with the
  JSON schema. This eliminates most parse failures at generation time.
- **Validate and retry**: parse the response against Pydantic, and on failure
  feed the validation error back to the model for a single repair attempt.

## Why It Matters for RAG

A RAG answer schema typically includes the answer text, a list of citation
sources with chunk IDs and quoted spans, a confidence score, and a refusal
flag. Enforcing the schema makes citations machine-checkable against the
retrieved context — the basis for faithfulness evaluation.
