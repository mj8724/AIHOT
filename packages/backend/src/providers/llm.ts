// OpenAI-compatible chat calls, always through receipts. One model is enough: `default` is whatever the
// deployment names in LLM_BASE_URL / LLM_API_KEY / LLM_MODEL, and every capability uses it unless the
// site, an environment variable or the admin's model page picks one of the site's named presets
// (site/models.ts).
import type { z } from "zod";
import { PRESETS } from "@aihot/site/models";
import { config, credential } from "../config.ts";
import { sha256 } from "../lib/ids.ts";
import { assertAccepted, paidRequest, ProviderRejectedError, rejectReceivedResponse } from "./receipts.ts";

export interface ModelSpec {
  key: string;
  service: string;
  model: string;
  baseUrlEnv: string;
  apiKeyEnv: string;
  /** Extra request fields, e.g. switching reasoning off for short structured tasks. */
  extra?: Record<string, unknown>;
  /** Output tokens added to every call's own limit for a reasoning model, which reasons before it answers. */
  reasoningTokens?: number;
  jsonMode: boolean;
  vision?: boolean;
}

function extraFromEnv(value: string | undefined): Record<string, unknown> | undefined {
  if (!value) return undefined;
  try {
    return JSON.parse(value) as Record<string, unknown>;
  } catch {
    throw new Error("LLM_EXTRA_JSON must be a JSON object, e.g. {\"enable_thinking\": false}");
  }
}

function reasoningTokensFromEnv(value: string | undefined): number | undefined {
  if (!value) return undefined;
  if (!/^\d+$/.test(value)) throw new Error("LLM_REASONING_TOKENS must be a non-negative integer, e.g. 4000");
  return Number(value);
}

export const MODELS: Record<string, ModelSpec> = {
  // Read from the environment at call time.
  default: {
    key: "default", service: "llm", baseUrlEnv: "LLM_BASE_URL", apiKeyEnv: "LLM_API_KEY",
    get model() { return process.env.LLM_MODEL ?? ""; },
    get extra() { return extraFromEnv(process.env.LLM_EXTRA_JSON); },
    get reasoningTokens() { return reasoningTokensFromEnv(process.env.LLM_REASONING_TOKENS); },
    get jsonMode() { return process.env.LLM_JSON_MODE !== "false"; },
    get vision() { return process.env.LLM_VISION === "true"; },
  },
  // The pack's named presets, each with its own address and key.
  ...Object.fromEntries(Object.entries(PRESETS).map(([key, preset]) => [key, { key, ...preset }])),
};

export type ContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } };

export interface ChatJsonOptions<S extends z.ZodType> {
  model: string;
  purpose: string;
  subject: string;
  promptVersion: string;
  system: string;
  user: string | ContentPart[];
  schema: S;
  temperature?: number;
  maxTokens?: number;
  attemptTag?: string;
  timeoutMs?: number;
  /** false: the model answers in its own text format (no JSON mode); `parse` turns it into the schema's input. */
  json?: boolean;
  parse?: (content: string) => unknown;
}

export interface ChatJsonResult<T> {
  data: T;
  receiptId: number;
  reused: boolean;
  model: string;
  usage: Record<string, unknown> | null;
}

export class ModelOutputError extends Error {
  readonly receiptId: number | null;
  constructor(message: string, receiptId: number | null = null) {
    super(message);
    this.receiptId = receiptId;
  }
}

function extractJson(text: string): unknown {
  let t = text.trim();
  const fence = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(t);
  if (fence) t = fence[1]!;
  const start = t.indexOf("{");
  const end = t.lastIndexOf("}");
  if (start === -1 || end === -1) throw new ModelOutputError("No JSON object in model output");
  const body = t.slice(start, end + 1);
  try {
    return JSON.parse(body);
  } catch {
    return JSON.parse(escapeControlCharsInStrings(body));
  }
}

/** Models sometimes emit raw newlines or tabs inside JSON strings (multi-line posts); escape only those. */
export function escapeControlCharsInStrings(json: string): string {
  let out = "";
  let inString = false;
  let escaped = false;
  for (const ch of json) {
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      else if (ch < " ") {
        out += ch === "\n" ? "\\n" : ch === "\r" ? "\\r" : ch === "\t" ? "\\t" : `\\u${ch.charCodeAt(0).toString(16).padStart(4, "0")}`;
        continue;
      }
    } else if (ch === '"') inString = true;
    out += ch;
  }
  return out;
}

function isConnectFailure(error: unknown): boolean {
  const code = (error as { cause?: { code?: string } })?.cause?.code ?? (error as { code?: string })?.code;
  return ["ECONNREFUSED", "ENOTFOUND", "EAI_AGAIN", "UND_ERR_CONNECT_TIMEOUT", "CERT_HAS_EXPIRED"].includes(code ?? "");
}

// newapi muse-spark 通道只放行流式请求: stream:true 回来的是 SSE,
// 这里拼回 chat-completion 形状, 下游解析/usage 逻辑零改动.
function isSseStream(res: Response, text: string): boolean {
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("text/event-stream")) return true;
  const t = text.trimStart();
  return t.startsWith("data:") && (text.includes("chat.completion.chunk") || text.includes("[DONE]"));
}

function parseChatCompletionsSse(text: string): Record<string, unknown> {
  let id: string | undefined;
  let model: string | undefined;
  let created = Math.floor(Date.now() / 1000);
  let finishReason: string | null = null;
  let usage: Record<string, unknown> | null = null;
  const parts: string[] = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t.startsWith("data:")) continue;
    const payload = t.slice(5).trim();
    if (!payload || payload === "[DONE]") continue;
    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (typeof chunk.id === "string") id = chunk.id;
    if (typeof chunk.model === "string") model = chunk.model;
    if (typeof chunk.created === "number") created = chunk.created;
    const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
    const delta = choices?.[0]?.delta as Record<string, unknown> | undefined;
    if (typeof delta?.content === "string") parts.push(delta.content);
    if (typeof choices?.[0]?.finish_reason === "string") finishReason = choices[0].finish_reason as string;
    if (chunk.usage && typeof chunk.usage === "object") usage = chunk.usage as Record<string, unknown>;
  }
  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: parts.join("") }, finish_reason: finishReason ?? "stop" }],
    ...(usage ? { usage } : {}),
  };
}

export async function chatJson<S extends z.ZodType>(opts: ChatJsonOptions<S>): Promise<ChatJsonResult<z.infer<S>>> {
  const spec = MODELS[opts.model];
  if (!spec) throw new Error(`Unknown model ${opts.model}`);
  if (!config.modelCallsEnabled) throw new Error("Model calls are disabled (MODEL_CALLS_ENABLED=false)");
  const baseUrl = credential("models", spec.baseUrlEnv);
  const apiKey = credential("models", spec.apiKeyEnv);
  if (!baseUrl || !apiKey || !spec.model) throw new Error(`Model ${opts.model} is not configured (${spec.baseUrlEnv}, ${spec.apiKeyEnv}${spec.key === "default" ? ", LLM_MODEL" : ""})`);

  const temperature = opts.temperature ?? 0.2;
  const maxTokens = Math.max(opts.maxTokens ?? 1500, 512) + (spec.reasoningTokens ?? 0);
  const userText = typeof opts.user === "string" ? opts.user : JSON.stringify(opts.user);
  const body: Record<string, unknown> = {
    model: spec.model,
    messages: [
      // A prompt given as one user message (the title/summary prompts) has no system message.
      ...(opts.system ? [{ role: "system", content: opts.system }] : []),
      // Multimodal parts go through as parts; plain objects are sent as JSON text.
      { role: "user", content: typeof opts.user === "string" || Array.isArray(opts.user) ? opts.user : userText },
    ],
    temperature,
    max_tokens: maxTokens,
    ...(spec.jsonMode && opts.json !== false ? { response_format: { type: "json_object" } } : {}),
    ...(spec.extra ?? {}),
    // newapi muse-spark 通道只放行流式请求(非流式→403 FreeTierError); 下方把 SSE 拼回 chat-completion 对象.
    stream: true,
  };

  const receipt = await paidRequest(
    {
      service: spec.service,
      model: spec.model,
      purpose: opts.purpose,
      subject: opts.subject,
      identity: { model: spec.model, promptVersion: opts.promptVersion, system: sha256(opts.system), user: sha256(userText), temperature, maxTokens, extra: spec.extra ?? null },
      requestSummary: { promptVersion: opts.promptVersion, systemHash: sha256(opts.system), userHash: sha256(userText), userChars: userText.length, temperature, maxTokens },
      attemptTag: opts.attemptTag,
    },
    async () => {
      const started = Date.now();
      let res: Response;
      try {
        res = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(opts.timeoutMs ?? 120_000),
        });
      } catch (error) {
        if (isConnectFailure(error)) throw new ProviderRejectedError(`connect failed: ${String(error)}`, null, true);
        throw error;
      }
      const text = await res.text();
      assertAccepted(spec.service, res.status, text);
      let json: Record<string, unknown>;
      if (isSseStream(res, text)) {
        json = parseChatCompletionsSse(text);
      } else {
        try {
          json = JSON.parse(text);
          if (!json || typeof json !== "object" || Array.isArray(json)) throw new Error("Expected a response object");
        } catch {
          json = { unparsable: text.slice(0, 20000) };
        }
      }
      const usage = (json.usage as Record<string, unknown> | undefined) ?? null;
      return {
        response: { ...json, _latencyMs: Date.now() - started },
        requestId: (json.id as string | undefined) ?? res.headers.get("x-request-id"),
        usage,
        cost: null,
      };
    },
  );

  const response = receipt.response as { choices?: Array<{ message?: { content?: string }; finish_reason?: string }>; usage?: Record<string, unknown> };
  const content = response.choices?.[0]?.message?.content ?? "";
  let parsed: z.infer<S>;
  try {
    parsed = opts.schema.parse(opts.parse ? opts.parse(content) : extractJson(content));
  } catch (error) {
    // Unusable output: record it and let a later attempt pay for a fresh answer. A reasoning model that
    // reasoned up to the output limit leaves an empty or cut-off answer; say so, and where to give it room.
    const detail = response.choices?.[0]?.finish_reason === "length"
      ? `output token limit reached (finish_reason=length): a reasoning model may have spent it reasoning; give it room with ${spec.key === "default" ? "LLM_REASONING_TOKENS" : `reasoningTokens on preset ${spec.key}`}. ${String(error)}`
      : String(error);
    await rejectReceivedResponse(receipt.receiptId, `unusable output: ${detail.slice(0, 500)}`);
    throw new ModelOutputError(`Model ${opts.model} returned unusable output for ${opts.subject}: ${detail.slice(0, 300)}`, receipt.receiptId);
  }
  return { data: parsed, receiptId: receipt.receiptId, reused: receipt.reused, model: spec.key, usage: response.usage ?? null };
}
