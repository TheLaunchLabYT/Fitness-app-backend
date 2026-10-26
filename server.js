// Tracker AI backend: a tiny Node server that sits between the app and Groq.
// The Groq key lives only here (Render environment variables), never in the app.
//
// Routes (all POST routes need the header  x-app-token: <APP_TOKEN>):
//   GET  /health          -> { ok, models }            (no token, used to wake the server)
//   POST /coach           -> today's suggestion from a check-in + recent training
//   POST /food/suggest    -> what to eat next, from your targets and saved meals
//   POST /food/estimate   -> macros for a meal you describe in words
//   POST /food/scan       -> macros from a photo of your food
//
// No dependencies: needs Node 18+ (Render uses Node 22 by default).

import http from 'node:http';

const PORT = Number(process.env.PORT) || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const APP_TOKEN = process.env.APP_TOKEN || '';
const DEFAULT_TEXT_MODEL = 'openai/gpt-oss-120b'; // Groq retired llama-3.3-70b-versatile in Aug 2026
const DEFAULT_VISION_MODEL = 'qwen/qwen3.8-27b';
const TEXT_MODEL = (process.env.TEXT_MODEL || DEFAULT_TEXT_MODEL).trim();
const VISION_MODEL = (process.env.VISION_MODEL || DEFAULT_VISION_MODEL).trim();
const MAX_BODY = 8 * 1024 * 1024; // photos are resized in the app, this is just a ceiling
const GROQ_URL = process.env.GROQ_URL || 'https://api.groq.com/openai/v1/chat/completions';

if (!GROQ_API_KEY) console.warn('GROQ_API_KEY is not set: AI routes will fail until you add it on Render.');
if (!APP_TOKEN) console.warn('APP_TOKEN is not set: anyone who finds this URL can use your Groq quota.');

/* ---------------- helpers ---------------- */

const num = (v, lo = 0, hi = 100000, d = 0) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  const p = 10 ** d;
  return Math.round(Math.min(hi, Math.max(lo, n)) * p) / p;
};
const str = (v, max = 400) => (typeof v === 'string' ? v.trim().slice(0, max) : '');
const oneOf = (v, list, fallback) => (list.includes(v) ? v : fallback);
const arr = (v, max = 20) => (Array.isArray(v) ? v.slice(0, max) : []);

const macros = (m = {}) => ({
  calories: num(m.calories, 0, 10000),
  protein: num(m.protein, 0, 1000, 1),
  carbs: num(m.carbs, 0, 2000, 1),
  fat: num(m.fat, 0, 1000, 1),
});

const sumItems = (items) =>
  items.reduce(
    (t, i) => ({
      calories: t.calories + i.calories,
      protein: Math.round((t.protein + i.protein) * 10) / 10,
      carbs: Math.round((t.carbs + i.carbs) * 10) / 10,
      fat: Math.round((t.fat + i.fat) * 10) / 10,
    }),
    { calories: 0, protein: 0, carbs: 0, fat: 0 },
  );

const cleanItems = (items) =>
  arr(items, 25)
    .map((i) => ({ name: str(i?.name, 80), grams: num(i?.grams, 0, 5000), ...macros(i) }))
    .filter((i) => i.name);

/** Pull the JSON object out of a model reply (drops <think> blocks and code fences). */
function parseJson(text) {
  const t = String(text || '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/```(?:json)?/gi, '');
  const a = t.indexOf('{');
  const b = t.lastIndexOf('}');
  if (a < 0 || b <= a) throw new Error('The AI did not return JSON');
  return JSON.parse(t.slice(a, b + 1));
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function groq({ model, system, user, temperature = 0.4, maxTokens = 1400 }) {
  // gpt-oss models think before answering; that thinking counts toward the token limit.
  const reasoning = /gpt-oss/i.test(model);
  if (!GROQ_API_KEY) throw new HttpError(500, 'GROQ_API_KEY is missing on the server');
  const body = {
    model,
    temperature,
    max_completion_tokens: reasoning ? maxTokens + 2500 : maxTokens,
    ...(reasoning ? { reasoning_effort: 'low' } : {}),
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
  };
  const call = async (b) => {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 45000);
    try {
      const res = await fetch(GROQ_URL, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${GROQ_API_KEY}` },
        body: JSON.stringify(b),
        signal: ctrl.signal,
      });
      const json = await res.json().catch(() => ({}));
      return { res, json };
    } finally {
      clearTimeout(timer);
    }
  };

  let { res, json } = await call(body);
  // JSON mode sometimes rejects an almost-valid answer; try once more without it and parse ourselves.
  if (res.status === 400 && /json/i.test(JSON.stringify(json))) {
    const { response_format, ...rest } = body;
    ({ res, json } = await call(rest));
  }
  // A mistyped or retired model name in Render's env vars: fall back to the built-in default.
  const fallback = model === VISION_MODEL ? DEFAULT_VISION_MODEL : DEFAULT_TEXT_MODEL;
  if ((res.status === 404 || /does not exist|model_not_found|decommissioned/i.test(JSON.stringify(json))) && model !== fallback) {
    console.warn(`Model "${model}" was rejected by Groq, using "${fallback}". Fix TEXT_MODEL / VISION_MODEL on Render.`);
    body.model = fallback;
    if (/gpt-oss/i.test(fallback)) {
      body.reasoning_effort = 'low';
      body.max_completion_tokens = maxTokens + 2500;
    }
    ({ res, json } = await call(body));
  }
  if (res.status === 429) throw new HttpError(429, 'Groq rate limit reached, try again in a minute');
  if (!res.ok) {
    console.error('Groq error', res.status, JSON.stringify(json).slice(0, 500));
    throw new HttpError(502, json?.error?.message || `Groq error ${res.status}`);
  }
  return parseJson(json?.choices?.[0]?.message?.content);
}

/* ---------------- routes ---------------- */

const RUN_TYPES = ['Easy', 'Tempo', 'Long', 'Intervals', 'Race'];

async function coach(body) {
  const system = `You are the coach inside a personal lifting and running tracker. Each day the user checks in with how they feel and you decide ONE thing for today.

Pick exactly one "kind":
- "rest": full day off.
- "recovery": light movement only (walk, mobility, stretching, easy cycling), 15-40 min.
- "run": one run. Give a run type (Easy, Tempo, Long, Intervals), distance in km and a target pace or effort.
- "workout": one strength session with 4-7 exercises.

Rules:
- Read the check-in first. Poor sleep, low energy or high soreness means go easier. Pain or illness means rest or recovery, and suggest seeing a professional if pain is sharp or lasts.
- Use the history: avoid training the same sore muscles two days in a row, balance lifting and running toward their weekly goal, and keep hard days apart.
- If a workout is already planned for today, keep it if the check-in allows (repeat its name and exercises), or swap it for something easier and say why.
- For workouts, prefer exercise names the user has done before. weightKg is the working weight; use 0 when unsure so the app fills in last time's weight.
- Keep the run distance realistic for this user: base it on their recent runs (new runners: 2-5 km).
- Plain, friendly, short. No emojis. Lowercase title, max 5 words.

Reply with JSON only:
{
  "kind": "rest" | "recovery" | "run" | "workout",
  "title": string,
  "summary": string (1-2 sentences, what to do today),
  "why": string (1 sentence, refer to their check-in or history),
  "intensity": "low" | "moderate" | "high",
  "durationMin": number,
  "run": null | { "type": "Easy"|"Tempo"|"Long"|"Intervals", "distanceKm": number, "target": string },
  "workout": null | { "name": string, "exercises": [ { "name": string, "sets": number, "reps": number, "weightKg": number } ] },
  "steps": [string] (2-6 short steps: warm-up, main part, cool-down, or recovery routine)
}`;
  const out = await groq({ model: TEXT_MODEL, system, user: JSON.stringify(body).slice(0, 24000), temperature: 0.5 });

  const kind = oneOf(out.kind, ['rest', 'recovery', 'run', 'workout'], 'recovery');
  const run =
    kind === 'run' && out.run
      ? { type: oneOf(out.run.type, RUN_TYPES, 'Easy'), distanceKm: num(out.run.distanceKm, 0.5, 60, 1), target: str(out.run.target, 120) }
      : null;
  const exercises = arr(out.workout?.exercises, 10)
    .map((e) => ({ name: str(e?.name, 60), sets: num(e?.sets, 1, 10), reps: num(e?.reps, 1, 50), weightKg: num(e?.weightKg, 0, 500, 1) }))
    .filter((e) => e.name);
  const workout = kind === 'workout' && exercises.length ? { name: str(out.workout?.name, 60) || 'Workout', exercises } : null;
  return {
    kind: kind === 'workout' && !workout ? 'recovery' : kind === 'run' && !run ? 'recovery' : kind,
    title: str(out.title, 60) || kind,
    summary: str(out.summary, 400),
    why: str(out.why, 300),
    intensity: oneOf(out.intensity, ['low', 'moderate', 'high'], 'low'),
    durationMin: num(out.durationMin, 0, 240),
    run,
    workout,
    steps: arr(out.steps, 8).map((s) => str(s, 200)).filter(Boolean),
  };
}

async function foodSuggest(body) {
  const system = `You are the nutrition helper inside a fitness tracker. The user has a calorie and macro target for the day, has already eaten some of it, and keeps a list of meals they like or can make.

Suggest up to 3 options for their next meal that fit what is LEFT of today's target.
- Strongly prefer meals from their list. When you use one, copy its "id" into "mealId" and scale the portion if needed (say so in "portion").
- You may add one new idea built from ingredients that appear in their meals, marked with "mealId": null.
- Respect their goal (cut = filling and high protein, bulk = enough calories, maintain = balanced) and the time of day.
- If they wrote a request, follow it.
- Macro numbers must be realistic for the portion. Calories should roughly equal protein*4 + carbs*4 + fat*9.
- No emojis. Short.

Reply with JSON only:
{
  "suggestions": [ { "name": string, "mealId": string | null, "portion": string, "calories": number, "protein": number, "carbs": number, "fat": number, "why": string } ],
  "tip": string (one short sentence about the rest of the day)
}`;
  const out = await groq({ model: TEXT_MODEL, system, user: JSON.stringify(body).slice(0, 24000), temperature: 0.6 });
  const ids = new Set(arr(body?.meals, 200).map((m) => m?.id));
  return {
    suggestions: arr(out.suggestions, 3)
      .map((s) => ({
        name: str(s?.name, 80),
        mealId: ids.has(s?.mealId) ? s.mealId : null,
        portion: str(s?.portion, 120),
        why: str(s?.why, 240),
        ...macros(s),
      }))
      .filter((s) => s.name),
    tip: str(out.tip, 240),
  };
}

const FOOD_JSON = `{
  "name": string (short name for the whole meal),
  "items": [ { "name": string, "grams": number, "calories": number, "protein": number, "carbs": number, "fat": number } ],
  "confidence": "low" | "medium" | "high",
  "notes": string (one short sentence: what you assumed, e.g. cooking oil or portion size)
}`;

function foodResult(out) {
  const items = cleanItems(out.items);
  return {
    name: str(out.name, 80) || items[0]?.name || 'Meal',
    items,
    total: sumItems(items),
    confidence: oneOf(out.confidence, ['low', 'medium', 'high'], 'medium'),
    notes: str(out.notes, 240),
  };
}

async function foodEstimate(body) {
  const text = str(body?.text, 1000);
  if (!text) throw new HttpError(400, 'Describe the meal first');
  const system = `You estimate nutrition. The user describes a meal in words. Break it into items with realistic grams and macros (use typical values from standard food databases; assume a normal home portion when no amount is given). Reply with JSON only:\n${FOOD_JSON}`;
  return foodResult(await groq({ model: TEXT_MODEL, system, user: text, temperature: 0.2, maxTokens: 1000 }));
}

async function foodScan(body) {
  const image = typeof body?.image === 'string' ? body.image : '';
  if (!/^data:image\/(jpeg|png|webp);base64,/.test(image)) throw new HttpError(400, 'Send the photo as a base64 data URL');
  const note = str(body?.note, 300);
  const system = `You look at a photo of food and estimate its nutrition. Identify each food on the plate, estimate the portion in grams from what you see (plate size, utensils, packaging), and give realistic macros from standard food databases. Include likely hidden calories such as oil, butter or sauce. If the photo is not food, return an empty items list and say so in notes. Reply with JSON only:\n${FOOD_JSON}`;
  const user = [
    { type: 'text', text: note ? `Extra info from the user: ${note}` : 'Estimate the nutrition of this meal.' },
    { type: 'image_url', image_url: { url: image } },
  ];
  return foodResult(await groq({ model: VISION_MODEL, system, user, temperature: 0.2, maxTokens: 1400 }));
}

const ROUTES = {
  '/coach': coach,
  '/food/suggest': foodSuggest,
  '/food/estimate': foodEstimate,
  '/food/scan': foodScan,
};

/* ---------------- server ---------------- */

// Simple per-IP limit so a leaked token can't drain your Groq quota.
const hits = new Map();
const limited = (ip) => {
  const now = Date.now();
  const list = (hits.get(ip) || []).filter((t) => now - t < 60000);
  list.push(now);
  hits.set(ip, list);
  return list.length > 30;
};

const send = (res, status, data) => {
  res.writeHead(status, {
    'content-type': 'application/json',
    'access-control-allow-origin': '*',
    'access-control-allow-headers': 'content-type, x-app-token',
    'access-control-allow-methods': 'GET, POST, OPTIONS',
  });
  res.end(JSON.stringify(data));
};

const readBody = (req) =>
  new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(new HttpError(413, 'Request too large'));
        req.destroy();
      } else chunks.push(c);
    });
    req.on('end', () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {});
      } catch {
        reject(new HttpError(400, 'Invalid JSON'));
      }
    });
    req.on('error', reject);
  });

const server = http.createServer(async (req, res) => {
  const path = (req.url || '/').split('?')[0].replace(/\/+$/, '') || '/';
  try {
    if (req.method === 'OPTIONS') return send(res, 204, {});
    if (req.method === 'GET' && (path === '/' || path === '/health')) {
      const info = { ok: true, service: 'tracker-ai', version: 2, models: { text: TEXT_MODEL, vision: VISION_MODEL }, keySet: !!GROQ_API_KEY };
      // /health?check=1 makes one tiny real call so the app can show whether the AI actually answers.
      if (/[?&]check=1/.test(req.url || '')) {
        try {
          await groq({ model: TEXT_MODEL, system: 'Reply with JSON only: {"ok": true}', user: 'ping', temperature: 0, maxTokens: 50 });
          info.aiOk = true; // the model answered with JSON
        } catch (e) {
          info.aiOk = false;
          info.aiError = e.message;
        }
      }
      return send(res, 200, info);
    }
    const handler = ROUTES[path];
    if (!handler || req.method !== 'POST') return send(res, 404, { error: 'Not found' });
    if (APP_TOKEN && req.headers['x-app-token'] !== APP_TOKEN) return send(res, 401, { error: 'Wrong app token' });
    const ip = String(req.headers['x-forwarded-for'] || req.socket.remoteAddress || '').split(',')[0].trim();
    if (limited(ip)) return send(res, 429, { error: 'Too many requests, slow down a little' });
    const body = await readBody(req);
    const started = Date.now();
    const out = await handler(body);
    console.log(`${path} ok in ${Date.now() - started} ms`);
    send(res, 200, out);
  } catch (e) {
    const status = e instanceof HttpError ? e.status : e instanceof SyntaxError ? 502 : 500;
    if (status >= 500) console.error(path, e);
    send(res, status, { error: e.message || 'Server error' });
  }
});

server.listen(PORT, () => console.log(`tracker-ai listening on ${PORT} (text: ${TEXT_MODEL}, vision: ${VISION_MODEL})`));
