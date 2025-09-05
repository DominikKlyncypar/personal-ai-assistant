// Minimal browser-side client for the worker API.
// Centralizes fetch, base URL handling, and response checks.

const API_BASE = "http://127.0.0.1:8000";
import {
  StartStopResponse,
  CaptureStatusResponse,
  LevelResponse,
  DumpWavResponse,
  TranscribeWavResponse,
  TranscribeUploadResponse,
} from "./schemas";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const ct = res.headers.get("content-type") || "";
  const isJson = ct.includes("application/json");
  const body = isJson ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = isJson && body && body.error ? body.error : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

export const worker = {
  // Meetings
  createMeeting: async (title = "Untitled") =>
    request(`/v1/meeting/new?title=${encodeURIComponent(title)}`, { method: "POST" }),

  deleteMeeting: async (id, cascade = true) =>
    request(`/v1/meeting/${id}?cascade=${cascade ? "true" : "false"}`, { method: "DELETE" }),

  exportJson: async (id, { speakers = "0", maxSpeakers } = {}) => {
    const p = new URLSearchParams({ speakers });
    if (speakers === "auto" && maxSpeakers) p.set("max_speakers", String(maxSpeakers));
    return request(`/v1/meeting/${id}/export.json?${p.toString()}`);
  },

  exportMarkdown: async (id, { speakers = "0", maxSpeakers } = {}) => {
    const p = new URLSearchParams({ speakers });
    if (speakers === "auto" && maxSpeakers) p.set("max_speakers", String(maxSpeakers));
    return request(`/v1/meeting/${id}/export.md?${p.toString()}`);
  },

  // Capture (legacy routes for now)
  startCapture: async (payload) => StartStopResponse.parse(await request(`/v1/start_capture`, { method: "POST", body: JSON.stringify(payload) })),
  stopCapture: async () => StartStopResponse.parse(await request(`/v1/stop_capture`, { method: "POST" })),
  captureStatus: async () => CaptureStatusResponse.parse(await request(`/v1/capture_status`)),
  level: async () => LevelResponse.parse(await request(`/v1/level`)),
  dumpWav: async (seconds = 5, label) => DumpWavResponse.parse(await request(`/v1/dump_wav?seconds=${encodeURIComponent(seconds)}${label ? `&label=${encodeURIComponent(label)}` : ""}`)),

  transcribeWav: async (path, opts={}) => TranscribeWavResponse.parse(await request(`/v1/transcribe_wav?path=${encodeURIComponent(path)}${opts.language ? `&language=${encodeURIComponent(opts.language)}` : ""}${opts.beam_size ? `&beam_size=${encodeURIComponent(opts.beam_size)}` : ""}`)),
  transcribeUpload: async (formData) => TranscribeUploadResponse.parse(await fetch(`${API_BASE}/v1/transcribe_upload`, { method: "POST", body: formData }).then(async (r)=>{ const ct=r.headers.get("content-type")||""; const b=ct.includes("application/json")? await r.json(): await r.text(); if(!r.ok) throw new Error(ct.includes("application/json")&&b&&b.error? b.error:`HTTP ${r.status}`); return b; })),
};
