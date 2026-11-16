// free-model-aggregator-02 — one server, two wires, real free models only.
//
// A client points at this server; every completion it answers comes from one
// of the fourteen free models in the trial brief, dialled through the
// grader's gateway on Chat Completions. The server never answers from
// itself: no canned text, no local model, no paraphrase of the request.
//
// Layout of this file, top to bottom:
//   1. the model table (read from the brief at runtime, hardcoded fallback)
//   2. cause-aware routing: only models meeting the prompt's needs, best
//      house rating first, ties broken by id so replays agree byte for byte
//   3. the gateway client with retry + failover and the failure trail
//   4. wire translation: Anthropic <-> Chat Completions, both directions
//   5. the HTTP server: /v1/chat/completions, /v1/messages, /v1/models
//   6. register with the env, then poll /work until it says done, exit 0
//
// Determinism: every choice comes from the brief, the request, or the
// gateway's answer. No wall clock, no Math.random, no UUIDs in anything
// sent. The only counter is a per-process response counter, which replays
// reproduce because the env sends the prompts in the same order.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

// ---------------------------------------------------------------- models

type FreeModel = {
  id: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
};

// The fourteen dialable ids, as the practice briefs list them. This table is
// only a fallback: at runtime the brief file wins, so a final brief with a
// different sealed catalog still routes from the brief, not from here.
const FALLBACK_MODELS: FreeModel[] = [
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", rating: 1150, context: 1024000, tools: true, vision: false, json: true },
  { id: "openrouter/cohere/north-mini-code:free", rating: 1200, context: 256000, tools: true, vision: false, json: false },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", rating: 1150, context: 512000, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", rating: 1150, context: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-31b-it:free", rating: 1210, context: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", rating: 1150, context: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", rating: 1150, context: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", rating: 1150, context: 256000, tools: true, vision: true, json: false },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", rating: 1285, context: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", rating: 1221, context: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", rating: 1303, context: 1000000, tools: true, vision: true, json: true },
  { id: "openrouter/particle-ai/particle-1:free", rating: 1150, context: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/recursive/flint-v1.0:free", rating: 1150, context: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/thinkingmachines/inkling-2-small:free", rating: 1475, context: 1048576, tools: true, vision: true, json: false },
];

const MODELS_SCHEMA = "free-model-aggregator-02/models/v1";

// Pull the sealed model list out of the brief text. The brief carries it as
// one JSON object; find its opening brace and scan for the matching close.
// Returns null when the brief holds no such object.
function briefModels(text: string): FreeModel[] | null {
  const start = text.indexOf(`{"schema":"${MODELS_SCHEMA}"`);
  if (start < 0) return null;
  let depth = 0;
  let end = -1;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        end = i + 1;
        break;
      }
    }
  }
  if (end < 0) return null;
  let doc: any;
  try {
    doc = JSON.parse(text.slice(start, end));
  } catch {
    return null;
  }
  if (!doc || !Array.isArray(doc.models) || doc.models.length === 0) return null;
  const out: FreeModel[] = [];
  for (const m of doc.models) {
    if (!m || typeof m.id !== "string" || m.free !== true) continue;
    out.push({
      id: m.id,
      rating: typeof m.rating_milli === "number" ? m.rating_milli : 0,
      context: typeof m.context_length === "number" ? m.context_length : null,
      tools: m.tools === true,
      vision: m.vision === true,
      json: m.json === true,
    });
  }
  return out.length > 0 ? out : null;
}

// Read the brief from MAKEFUN_PROMPT_FILE when the trial sets it.
function loadModels(): FreeModel[] {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file) {
    try {
      const parsed = briefModels(readFileSync(file, "utf8"));
      if (parsed !== null) return parsed;
    } catch {
      // fall through to the fallback table
    }
  }
  return FALLBACK_MODELS;
}

// The catalog document, byte for byte as registered and as --list prints it.
function buildCatalog(models: FreeModel[]): any {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.context })),
      },
    ],
  };
}

// ---------------------------------------------------------------- routing

type PromptNeeds = {
  tools: boolean;
  json: boolean;
  vision: boolean;
  // rough input size in tokens (a token per 3 bytes) plus the output cap:
  // the model's window must cover both.
  budget: number;
};

// Read what a prompt needs straight from its request body: declared tools,
// a JSON-mode response_format, image parts, and the output cap.
function needsOf(body: any): PromptNeeds {
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  const fmt = body.response_format;
  const json = !!fmt && (fmt.type === "json_object" || fmt.type === "json_schema");
  let vision = false;
  let bytes = 0;
  const countContent = (c: any): void => {
    if (typeof c === "string") {
      bytes += c.length;
      return;
    }
    if (!Array.isArray(c)) return;
    for (const p of c) {
      if (!p || typeof p !== "object") continue;
      if (p.type === "image_url" || p.type === "image") vision = true;
      if (typeof p.text === "string") bytes += p.text.length;
    }
  };
  if (typeof body.system === "string") bytes += body.system.length;
  if (Array.isArray(body.system)) countContent(body.system);
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  for (const m of msgs) {
    if (m && typeof m === "object") countContent(m.content);
  }
  const maxTokens = typeof body.max_tokens === "number" && body.max_tokens > 0 ? body.max_tokens : 1024;
  return { tools, json, vision, budget: Math.ceil(bytes / 3) + maxTokens };
}

// Eligible models only, best house rating first, id ascending on a tie so
// the choice is the same on the live pass and the replay.
function ranked(models: FreeModel[], needs: PromptNeeds): FreeModel[] {
  return models
    .filter(
      (m) =>
        (!needs.tools || m.tools) &&
        (!needs.json || m.json) &&
        (!needs.vision || m.vision) &&
        (m.context === null || m.context >= needs.budget),
    )
    .sort((a, b) => b.rating - a.rating || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

// The provider's answer names the model without the gateway prefix; the
// dialled id is what every answer and chunk must report.
function upstreamId(dialed: string): string {
  return dialed.startsWith("openrouter/") ? dialed.slice("openrouter/".length) : dialed;
}

// ---------------------------------------------------------------- failures

type FailureRow = { status: number; code: string; model: string | null };

function errorCodeOf(payload: any): string {
  const err = payload && typeof payload === "object" ? payload.error : null;
  if (err && typeof err === "object") {
    if (typeof err.code === "string" && err.code !== "") return err.code.slice(0, 64);
    if (typeof err.type === "string" && err.type !== "") return err.type.slice(0, 64);
    if (typeof err.message === "string" && err.message !== "") return err.message.slice(0, 64);
  }
  if (typeof payload?.message === "string" && payload.message !== "") return payload.message.slice(0, 64);
  return "upstream_error";
}

function failuresHeader(rows: FailureRow[]): string | null {
  if (rows.length === 0) return null;
  return JSON.stringify(rows.slice(0, 8));
}

// ---------------------------------------------------------------- gateway

type UpstreamOk = {
  ok: true;
  text: string;
  finish: string;
  usage: { input: number; output: number };
  toolCalls: Array<{ id: string; name: string; args: string }>;
  rawModel: string | null;
};

type UpstreamOut = UpstreamOk | { ok: false; status: number; code: string };

const GATEWAY_TIMEOUT_MS = 25000;

// One Chat Completions call. Non-streaming upstream always: the server
// re-emits the answer in the wire the client asked for, streamed or not,
// so the upstream shape never leaks into the client's wire.
async function callUpstream(modelId: string, upstreamBody: any): Promise<UpstreamOut> {
  const base = (process.env.OPENAI_BASE_URL || "https://openrouter.ai/api/v1").replace(/\/$/, "");
  const key = process.env.OPENAI_API_KEY || "";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GATEWAY_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${key}`,
      },
      body: JSON.stringify(upstreamBody),
      signal: controller.signal,
    });
  } catch {
    clearTimeout(timer);
    return { ok: false, status: 502, code: "gateway_unreachable" };
  } finally {
    clearTimeout(timer);
  }
  let payload: any = null;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }
  if (!res.ok) {
    return { ok: false, status: res.status, code: errorCodeOf(payload) };
  }
  const choice = payload && Array.isArray(payload.choices) ? payload.choices[0] : null;
  const message = choice && typeof choice.message === "object" ? choice.message : null;
  if (!message) return { ok: false, status: 502, code: "empty_upstream_answer" };
  let text = "";
  if (typeof message.content === "string") text = message.content;
  else if (Array.isArray(message.content)) {
    for (const p of message.content) {
      if (p && typeof p === "object" && p.type === "text" && typeof p.text === "string") text += p.text;
    }
  }
  const toolCalls: UpstreamOk["toolCalls"] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const c of message.tool_calls) {
      if (!c || typeof c !== "object" || !c.function || typeof c.function.name !== "string") continue;
      toolCalls.push({
        id: typeof c.id === "string" ? c.id : `call-${toolCalls.length}`,
        name: c.function.name,
        args: typeof c.function.arguments === "string" ? c.function.arguments : "{}",
      });
    }
  }
  const usage = payload.usage && typeof payload.usage === "object" ? payload.usage : {};
  const input = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0;
  const outputRaw = typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0;
  return {
    ok: true,
    text,
    finish: typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
    usage: { input: Math.max(0, input), output: Math.max(1, outputRaw) },
    toolCalls,
    rawModel: typeof payload.model === "string" ? payload.model : null,
  };
}

// ---------------------------------------------------------------- translate

// Anthropic message content -> Chat Completions messages. System becomes a
// leading system message; image parts become image_url parts with a data URL.
function anthropicToChat(body: any): any[] {
  const messages: any[] = [];
  if (typeof body.system === "string" && body.system !== "") {
    messages.push({ role: "system", content: body.system });
  } else if (Array.isArray(body.system)) {
    const text = body.system
      .filter((p: any) => p && p.type === "text" && typeof p.text === "string")
      .map((p: any) => p.text)
      .join("");
    if (text !== "") messages.push({ role: "system", content: text });
  }
  for (const m of Array.isArray(body.messages) ? body.messages : []) {
    if (!m || typeof m !== "object") continue;
    const role = m.role === "assistant" ? "assistant" : "user";
    const content = m.content;
    if (typeof content === "string") {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: any[] = [];
    for (const p of content) {
      if (!p || typeof p !== "object") continue;
      if (p.type === "text" && typeof p.text === "string") parts.push({ type: "text", text: p.text });
      else if (p.type === "image" && p.source && typeof p.source === "object") {
        const src = p.source;
        if (src.type === "url" && typeof src.url === "string") {
          parts.push({ type: "image_url", image_url: { url: src.url } });
        } else if (src.type === "base64" && typeof src.data === "string") {
          parts.push({ type: "image_url", image_url: { url: `data:${src.media_type || "image/png"};base64,${src.data}` } });
        }
      } else if (p.type === "tool_use" && typeof p.name === "string") {
        parts.push({ type: "text", text: `[tool ${p.name} called]` });
      } else if (p.type === "tool_result") {
        const t = typeof p.content === "string" ? p.content : "";
        if (t !== "") parts.push({ type: "text", text: t });
      }
    }
    messages.push({ role, content: parts });
  }
  return messages;
}

// Anthropic tool declarations -> OpenAI function tools.
function anthropicToolsToChat(tools: any): any[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  const out: any[] = [];
  for (const t of tools) {
    if (!t || typeof t !== "object" || typeof t.name !== "string") continue;
    out.push({
      type: "function",
      function: {
        name: t.name,
        description: typeof t.description === "string" ? t.description : "",
        parameters: t.input_schema && typeof t.input_schema === "object" ? t.input_schema : { type: "object", properties: {} },
      },
    });
  }
  return out.length > 0 ? out : undefined;
}

// Build the upstream Chat Completions body for one client request, on the
// model chosen. Only the fields the gateway needs are carried over; nothing
// stateful (store, previous ids) ever goes out.
function upstreamBodyFor(api: string, body: any, modelId: string): any {
  const maxTokens =
    typeof body.max_tokens === "number" && body.max_tokens > 0
      ? Math.min(body.max_tokens, 8192)
      : typeof body.max_completion_tokens === "number" && body.max_completion_tokens > 0
        ? Math.min(body.max_completion_tokens, 8192)
        : 1024;
  const out: any = { model: modelId, max_tokens: maxTokens, stream: false };
  if (api === "openai") {
    out.messages = Array.isArray(body.messages) ? body.messages : [];
    if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = body.tools;
    if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
    if (body.response_format && typeof body.response_format === "object") out.response_format = body.response_format;
  } else {
    out.messages = anthropicToChat(body);
    const tools = anthropicToolsToChat(body.tools);
    if (tools) out.tools = tools;
  }
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop)) out.stop = body.stop;
  else if (typeof body.stop === "string") out.stop = body.stop;
  return out;
}

// ---------------------------------------------------------------- serve one prompt

type Served = {
  status: number;
  dialed: string | null;
  answer: UpstreamOk | null;
  failures: FailureRow[];
};

// Dial, retrying the same model once, then failing over down the ranked
// list. At most four upstream calls per prompt, so a bad model cannot eat
// the trial's call budget. Every refusal is recorded for the trail header.
async function servePrompt(api: string, body: any, models: FreeModel[]): Promise<Served> {
  const needs = needsOf(body);
  const order = ranked(models, needs);
  const failures: FailureRow[] = [];
  if (order.length === 0) {
    return { status: 502, dialed: null, answer: null, failures };
  }
  const attempts: string[] = [];
  attempts.push(order[0].id, order[0].id);
  for (let i = 1; i < order.length && attempts.length < 4; i++) attempts.push(order[i].id);
  for (const modelId of attempts) {
    const upstream = await callUpstream(modelId, upstreamBodyFor(api, body, modelId));
    if (upstream.ok) {
      return { status: 200, dialed: modelId, answer: upstream, failures };
    }
    failures.push({ status: upstream.status, code: upstream.code, model: modelId });
    // A 400 the gateway refused before forwarding (bad model, no cap) will
    // not be fixed by another model of the same shape; stop spending calls.
    if (upstream.status === 400) break;
  }
  return { status: 502, dialed: null, answer: null, failures };
}

// Stop reasons the two wires know, mapped from the upstream finish reason.
function openaiFinish(finish: string): string {
  if (finish === "length") return "length";
  if (finish === "tool_calls" || finish === "function_call") return "tool_calls";
  if (finish === "content_filter") return "content_filter";
  return "stop";
}

function anthropicStop(finish: string, toolCalls: number): string {
  if (toolCalls > 0) return "tool_use";
  if (finish === "length") return "max_tokens";
  return "end_turn";
}

// ---------------------------------------------------------------- wire out

let seq = 0;

function trialCreated(): number {
  const raw = process.env.MAKEFUN_TRIAL_NOW;
  if (!raw) return 0;
  if (/^\d+$/.test(raw)) return Number(raw);
  const ms = Date.parse(raw);
  return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0;
}

function splitText(text: string, size: number): string[] {
  if (text === "") return [];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

function sendJson(res: ServerResponse, status: number, body: any, failures: FailureRow[]): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(text)),
  };
  const trail = failuresHeader(failures);
  if (trail !== null) headers["x-makefun-upstream-failures"] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendSse(res: ServerResponse, frames: string[], failures: FailureRow[]): void {
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  };
  const trail = failuresHeader(failures);
  if (trail !== null) headers["x-makefun-upstream-failures"] = trail;
  res.writeHead(200, headers);
  for (const f of frames) res.write(f);
  res.end();
}

// OpenAI JSON completion: the upstream text whole, the dialled id, usage
// and a finish reason the API knows.
function openaiJsonBody(dialed: string, answer: UpstreamOk, stream: boolean): any {
  const finish = openaiFinish(answer.finish);
  const message: any = { role: "assistant", content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: `chatcmpl-agg-${++seq}`,
    object: "chat.completion",
    created: trialCreated(),
    model: dialed,
    choices: [{ index: 0, message, finish_reason: finish }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

// OpenAI SSE: content deltas, one finish chunk carrying the usage, [DONE]
// last. The usage rides the finish chunk, as stream_options.include_usage
// does on the gateway.
function openaiSseFrames(dialed: string, answer: UpstreamOk): string[] {
  const id = `chatcmpl-agg-${++seq}`;
  const created = trialCreated();
  const frames: string[] = [];
  const finish = openaiFinish(answer.finish);
  for (const piece of splitText(answer.text, 500)) {
    frames.push(
      `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: dialed, choices: [{ index: 0, delta: { role: "assistant", content: piece }, finish_reason: null }] })}\n\n`,
    );
  }
  if (answer.toolCalls.length > 0) {
    frames.push(
      `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: dialed, choices: [{ index: 0, delta: { tool_calls: answer.toolCalls.map((c, i) => ({ index: i, id: c.id, type: "function", function: { name: c.name, arguments: c.args } })) }, finish_reason: null }] })}\n\n`,
    );
  }
  frames.push(
    `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: dialed, choices: [{ index: 0, delta: {}, finish_reason: finish }], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return frames;
}

// Anthropic JSON message: text blocks (plus tool_use blocks when the
// provider called one), the dialled id, usage, a stop reason the API knows.
function anthropicJsonBody(dialed: string, answer: UpstreamOk): any {
  const content: any[] = [];
  if (answer.text !== "") content.push({ type: "text", text: answer.text });
  for (const c of answer.toolCalls) {
    let input: any = {};
    try {
      input = JSON.parse(c.args);
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: c.id, name: c.name, input });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  return {
    id: `msg-agg-${++seq}`,
    type: "message",
    role: "assistant",
    content,
    model: dialed,
    stop_reason: anthropicStop(answer.finish, answer.toolCalls.length),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

// Anthropic SSE, in the order the API requires: message_start first, then
// one text block (start, deltas, stop), tool_use blocks after it when the
// provider called any, message_delta carrying the stop reason and the
// output count, message_stop last and nothing after it.
function anthropicSseFrames(dialed: string, answer: UpstreamOk): string[] {
  const id = `msg-agg-${++seq}`;
  const frames: string[] = [];
  const stop = anthropicStop(answer.finish, answer.toolCalls.length);
  frames.push(
    `event: message_start\ndata: ${JSON.stringify({ type: "message_start", message: { id, type: "message", role: "assistant", content: [], model: dialed, stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } } })}\n\n`,
  );
  frames.push(
    `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`,
  );
  for (const piece of splitText(answer.text, 500)) {
    frames.push(
      `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } })}\n\n`,
    );
  }
  frames.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
  let index = 1;
  for (const c of answer.toolCalls) {
    frames.push(
      `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index, content_block: { type: "tool_use", id: c.id, name: c.name, input: {} } })}\n\n`,
    );
    frames.push(
      `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: c.args } })}\n\n`,
    );
    frames.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index })}\n\n`);
    index += 1;
  }
  frames.push(
    `event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.usage.output } })}\n\n`,
  );
  frames.push(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
  return frames;
}

// ---------------------------------------------------------------- server

function modelsListBody(models: FreeModel[], anthropic: boolean): any {
  const data = models.map((m) => {
    if (anthropic) return { id: m.id, type: "model", display_name: m.id, created: trialCreated(), free: true };
    return { id: m.id, object: "model", created: trialCreated(), owned_by: "openrouter", free: true };
  });
  if (anthropic) return { data };
  return { object: "list", data };
}

async function readJsonBody(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 1024 * 1024) throw new Error("body_too_large");
    chunks.push(chunk as Buffer);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function handleCompletions(
  api: string,
  req: IncomingMessage,
  res: ServerResponse,
  models: FreeModel[],
): Promise<void> {
  let body: any;
  try {
    body = await readJsonBody(req);
  } catch {
    sendJson(res, 400, { error: { message: "the request body is not JSON", type: "invalid_request_error" } }, []);
    return;
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    sendJson(res, 400, { error: { message: "the request body is not one JSON object", type: "invalid_request_error" } }, []);
    return;
  }
  const stream = body.stream === true;
  const served = await servePrompt(api, body, models);
  if (served.status !== 200 || !served.answer || !served.dialed) {
    sendJson(
      res,
      502,
      { error: { message: "every free model refused this prompt", type: "server_error" } },
      served.failures,
    );
    return;
  }
  const dialed = served.dialed;
  const answer = served.answer;
  if (api === "openai") {
    if (stream) sendSse(res, openaiSseFrames(dialed, answer), served.failures);
    else sendJson(res, 200, openaiJsonBody(dialed, answer, false), served.failures);
  } else {
    if (stream) sendSse(res, anthropicSseFrames(dialed, answer), served.failures);
    else sendJson(res, 200, anthropicJsonBody(dialed, answer), served.failures);
  }
}

// ---------------------------------------------------------------- main

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

async function registerWithEnv(envUrl: string, port: number, catalog: any): Promise<boolean> {
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, "")}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
      });
      if (res.ok) return true;
    } catch {
      // the env may not be listening yet; retry below
    }
    await sleep(500);
  }
  return false;
}

async function waitForDone(envUrl: string): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, "")}/work`);
      if (res.ok) {
        const data: any = await res.json();
        if (data && data.done === true) return;
      }
    } catch {
      // keep polling
    }
    await sleep(500);
  }
}

async function main(): Promise<void> {
  const models = loadModels();
  const catalog = buildCatalog(models);
  if (process.argv.includes("--list")) {
    // Outside a trial there is no brief to read, so there is no catalog to
    // print: an empty document, as the brief's starter does. Inside a trial
    // the catalog above comes from the brief and matches the registration.
    if (!process.env.MAKEFUN_PROMPT_FILE) {
      process.stdout.write(JSON.stringify({ providers: [] }));
    } else {
      process.stdout.write(JSON.stringify(catalog));
    }
    return;
  }
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (!envUrl) {
    process.stderr.write("free-model-aggregator-02: MAKEFUN_ENV_URL is not set; nothing to register with\n");
    process.exitCode = 1;
    return;
  }
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://server");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      const anthropic = req.headers["x-api-key"] !== undefined;
      sendJson(res, 200, modelsListBody(models, anthropic), []);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      handleCompletions("openai", req, res, models).catch(() => {
        try {
          sendJson(res, 500, { error: { message: "internal error", type: "server_error" } }, []);
        } catch {
          // the response is already gone
        }
      });
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/messages") {
      handleCompletions("anthropic", req, res, models).catch(() => {
        try {
          sendJson(res, 500, { error: { message: "internal error", type: "server_error" } }, []);
        } catch {
          // the response is already gone
        }
      });
      return;
    }
    sendJson(res, 404, { error: { message: `no route ${req.method} ${url.pathname}`, type: "invalid_request_error" } }, []);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  const ok = await registerWithEnv(envUrl, port, catalog);
  if (!ok) {
    process.stderr.write("free-model-aggregator-02: the env never accepted the registration\n");
    process.exitCode = 1;
    server.close();
    return;
  }
  await waitForDone(envUrl);
  server.close();
}

process.on("SIGTERM", () => process.exit(0));
main().catch((err) => {
  process.stderr.write(`free-model-aggregator-02: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
