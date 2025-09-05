// Zod schemas for worker responses. Keep in sync with worker/src/models/*.
import { z } from "zod";

export const StartStopResponse = z.object({
  ok: z.boolean(),
  message: z.string(),
  running: z.boolean(),
});

export const CaptureStatusResponse = z.object({ running: z.boolean() });
export const LevelResponse = z.object({ running: z.boolean(), rms: z.number() });

export const DumpWavResponse = z.object({
  ok: z.boolean(),
  path: z.string(),
  filename: z.string(),
  seconds: z.number(),
  samplerate: z.number(),
  samples: z.number(),
});

export const Segment = z.object({ start: z.number(), end: z.number(), text: z.string() });

export const TranscribeWavResponse = z.object({
  ok: z.boolean(),
  language: z.string().optional(),
  duration: z.number().optional(),
  segments: z.array(Segment),
  confidence: z.number().nullable().optional(),
});

export const TranscribeUploadResponse = z.object({
  ok: z.boolean(),
  meeting_id: z.number(),
  filename: z.string(),
  segments: z.array(Segment),
  text: z.string(),
});

export const VadEvents = z.object({ started: z.boolean(), ended: z.boolean() });
export const VadCheckResponse = z.object({
  ok: z.boolean(),
  running: z.boolean(),
  rms: z.number(),
  db: z.number(),
  speech: z.boolean(),
  raw_is_speech_frame: z.boolean(),
  speech_frames: z.number(),
  silence_frames: z.number(),
  ms_since_voice: z.number(),
  events: VadEvents,
});

