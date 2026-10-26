// Tracker AI backend: a tiny Node server that sits between the app and Groq.
// The Groq key lives only here (Render environment variables), never in the app.
//
// Routes (all POST routes need the header  x-app-token: <APP_TOKEN>):
//   GET  /health          -> { ok, models }            (no token, used to wake the server)
//   POST /coach           -> today's suggestion from a check-in + recent training
//   POST /food/suggest    -> what to eat next, from your targets and saved meals
//   POST /food/estimate   -> macros for a meal you describe in words
//   POST /food/scan       -> macros from a photo of your food
//   POST /split/build     -> a weekly training split + nutrition advice from the split builder
//   POST /assistant       -> a chat reply for Train › ask ai, using the user's own data
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
- "workout": one strength session with 3-8 exercises.

1. Start from the split.
- "plannedToday" is today's workout. When plannedToday.fromSplit is true it comes from the user's split ("split" has the split, today's day and the next few days). That is the starting point: they expect to do it.
- Then tailor it. You may keep it exactly; adjust it (add an exercise, remove one, swap one for another, change sets or reps); or replace it entirely with a different session, a run, recovery or rest. Keep as much of the split day as makes sense: a good reason is needed for each change (soreness, an injury or note, time, a weak point in their goal, what they did recently, how busy their day is).
- Add exercises only when the day has room and they fit the day's focus and the user's goal (goals.training).
- If you replace or skip the split day, say where it could go instead (e.g. tomorrow, or swap with the next rest day) so the week stays balanced.
- With no plannedToday: plan from the split's today, their history and weekly goal as usual.
- List every change in "changes", in order. Keep exercise names exactly as in plannedToday for the ones you keep.

2. Weigh their schedule ("schedule", from their Quill planner; null means unknown, then ignore this part).
- today / tomorrow / dayAfter: level (light, normal, busy, packed), busyMin (fixed events + planned work), freeMin (free time left in their usual windows), due (tasks due that day with priority), tests (exams that day). behind: whether they are behind on their tasks overall. trainingTimeToday: when Quill scheduled today's session.
- Decide how important training is today compared with what else needs the time. Training matters: consistency over the week, their goal, and it helps with stress and focus, so don't drop it for ordinary busyness or low-priority tasks.
- But when today is packed or busy AND something important is close (a test today or tomorrow, an urgent or high-priority task due today or tomorrow, or behind.status "behind"), protect that time: shorten the session (about 30-45 min, keep the main lifts, drop accessories), move it to a better time, or skip it for a short walk and say what the time is better used for.
- Light day and feeling good: train as planned; adding one accessory is fine.
- Also consider how many sessions they did this week (thisWeek) versus goals.sessionsPerWeek: if they are behind on training, it matters more today.
- Put the decision in "schedule": call "train" (as planned), "shorter", "move" (with "at": a suggested start time "HH:MM" if useful) or "skip". "note": one short sentence naming the reason from their day (e.g. "Physics test tomorrow and 1h free: 35 min, main lifts only."). Never list their other tasks beyond what is needed to explain.

3. General rules.
- Read the check-in first. Poor sleep, low energy or high soreness means go easier. Pain or illness means rest or recovery, and suggest seeing a professional if pain is sharp or lasts.
- Avoid training the same sore muscles two days in a row and keep hard days apart.
- checkIn.timeAvailableMin is a hard limit when given: about 10 min warm-up plus 2-3 min per set.
- Prefer exercise names the user has done before. weightKg is the working weight; use 0 when unsure (the app fills in last time's weight). Keep the planned weight for exercises you keep.
- Keep the run distance realistic for this user: base it on their recent runs (new runners: 2-5 km).
- Plain, friendly, short. No emojis. Lowercase title, max 5 words.

Reply with JSON only:
{
  "kind": "rest" | "recovery" | "run" | "workout",
  "title": string,
  "summary": string (1-2 sentences, what to do today),
  "why": string (1 sentence, refer to their check-in, split, history or day),
  "intensity": "low" | "moderate" | "high",
  "durationMin": number,
  "run": null | { "type": "Easy"|"Tempo"|"Long"|"Intervals", "distanceKm": number, "target": string },
  "workout": null | { "name": string, "exercises": [ { "name": string, "sets": number, "reps": number, "weightKg": number } ] } (the final list, after your changes),
  "basedOn": string | null (name of the split day you started from),
  "changes": [ { "kind": "add" | "remove" | "swap" | "change", "name": string (the exercise; for swap the one taken out), "to": string (swap only: the replacement), "detail": string (e.g. "3 x 12" or "4 -> 3 sets"), "why": string (short) } ],
  "schedule": null | { "call": "train" | "shorter" | "move" | "skip", "note": string, "at": string | null },
  "steps": [string] (2-6 short steps: warm-up, main part, cool-down, or recovery routine)
}`;
  const out = await groq({ model: TEXT_MODEL, system, user: JSON.stringify(body).slice(0, 24000), temperature: 0.5, maxTokens: 1800 });

  const kind = oneOf(out.kind, ['rest', 'recovery', 'run', 'workout'], 'recovery');
  const run =
    kind === 'run' && out.run
      ? { type: oneOf(out.run.type, RUN_TYPES, 'Easy'), distanceKm: num(out.run.distanceKm, 0.5, 60, 1), target: str(out.run.target, 120) }
      : null;
  const exercises = arr(out.workout?.exercises, 10)
    .map((e) => ({ name: str(e?.name, 60), sets: num(e?.sets, 1, 10), reps: num(e?.reps, 1, 50), weightKg: num(e?.weightKg, 0, 500, 1) }))
    .filter((e) => e.name);
  const workout = kind === 'workout' && exercises.length ? { name: str(out.workout?.name, 60) || 'Workout', exercises } : null;
  const finalKind = kind === 'workout' && !workout ? 'recovery' : kind === 'run' && !run ? 'recovery' : kind;
  const planned = body?.plannedToday && Array.isArray(body.plannedToday.exercises) ? body.plannedToday : null;
  const { changes, replaced } = coachChanges(planned, finalKind === 'workout' ? workout : null, arr(out.changes, 12));
  const sc = out.schedule && typeof out.schedule === 'object' && body?.schedule ? out.schedule : null;
  return {
    kind: finalKind,
    title: str(out.title, 60) || finalKind,
    summary: str(out.summary, 400),
    why: str(out.why, 300),
    intensity: oneOf(out.intensity, ['low', 'moderate', 'high'], 'low'),
    durationMin: num(out.durationMin, 0, 240),
    run,
    workout,
    steps: arr(out.steps, 8).map((s) => str(s, 200)).filter(Boolean),
    basedOn: planned ? str(out.basedOn, 60) || str(planned.name, 60) || null : null,
    changes,
    replaced,
    schedule: sc
      ? {
          call: oneOf(sc.call, ['train', 'shorter', 'move', 'skip'], 'train'),
          note: str(sc.note, 240),
          ...(/^\d{1,2}:\d{2}$/.test(String(sc.at || '')) ? { at: String(sc.at) } : {}),
        }
      : null,
  };
}

/** The real difference between today's plan and the coach's workout. The model's own list only supplies the reasons
 *  (and which removed/added pair is a swap), so what the app shows always matches the exercises you'll do. */
function coachChanges(planned, workout, said) {
  if (!planned || !planned.exercises.length) return { changes: [], replaced: false };
  const key = (n) => String(n || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const before = new Map(planned.exercises.map((e) => [key(e.name), e]));
  const after = workout ? new Map(workout.exercises.map((e) => [key(e.name), e])) : new Map();
  const kept = [...before.keys()].filter((k) => after.has(k));
  // Not a workout any more, or nothing in common: the split day was replaced entirely.
  if (!workout || kept.length === 0) return { changes: [], replaced: true };
  const notes = said.filter((c) => c && typeof c === 'object').map((c) => ({ kind: String(c.kind || ''), name: key(c.name), to: key(c.to), why: str(c.why, 160) }));
  const whyFor = (k, kind) => (notes.find((c) => (c.name === k || c.to === k) && (!kind || c.kind === kind)) || notes.find((c) => c.name === k || c.to === k))?.why || '';
  const sr = (e) => `${e.sets} × ${e.reps}`;
  const removed = [...before.keys()].filter((k) => !after.has(k));
  const added = [...after.keys()].filter((k) => !before.has(k));
  const changes = [];
  for (const c of notes) {
    if (c.kind !== 'swap' || !removed.includes(c.name) || !added.includes(c.to)) continue;
    removed.splice(removed.indexOf(c.name), 1);
    added.splice(added.indexOf(c.to), 1);
    const e = after.get(c.to);
    changes.push({ kind: 'swap', name: before.get(c.name).name, to: e.name, detail: sr(e), why: c.why });
  }
  for (const k of removed) changes.push({ kind: 'remove', name: before.get(k).name, why: whyFor(k, 'remove') });
  for (const k of added) { const e = after.get(k); changes.push({ kind: 'add', name: e.name, detail: sr(e), why: whyFor(k, 'add') }); }
  for (const k of kept) {
    const a = before.get(k), b = after.get(k);
    if (num(a.sets, 0, 99) !== b.sets || num(a.reps, 0, 999) !== b.reps) changes.push({ kind: 'change', name: b.name, detail: `${sr({ sets: num(a.sets, 0, 99), reps: num(a.reps, 0, 999) })} → ${sr(b)}`, why: whyFor(k, 'change') });
  }
  return { changes: changes.slice(0, 12), replaced: false };
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

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

async function splitBuild(body) {
  const system = `You are a strength and conditioning coach inside a lifting and running tracker. The user filled in a split builder: their goal, where they are now (experience, body, current working weights, injuries) and what they do now (days available, session length, equipment, runs, daily activity, current routine). You also get their recent logged training.

Design ONE weekly training split (Monday to Sunday) that fits all of it, plus nutrition direction.

Training rules:
- Lifting days: exactly answers.daysPerWeek (max 6). If answers.weekdays is not empty, lift ONLY on those weekdays. Otherwise spread hard days out.
- Pick the structure that suits the days and experience: 2-3 days full body; 3 days PPL only for experienced lifters chasing muscle; 4 days upper/lower; 5 days upper/lower + push/pull/legs; 6 days push/pull/legs twice. Change it when their goal or current routine clearly calls for something else, and say why.
- Each muscle group about twice a week where possible. Big compound lifts first, then accessories.
- Fit the session length: about 1 exercise per 10-12 minutes (30 min = 3, 60 min = 5-6, 90 min = 7).
- Reps by goal: strength 3-6 on main lifts; muscle 6-12; fat loss 8-15 and keep lifting heavy-ish to hold muscle; fitness 8-12; endurance 12-15 and fewer lifting days.
- Sets: beginners 2-3, others 3-4.
- Equipment: "gym" any; "dumbbells" only dumbbells, a bench and a pull-up bar; "bodyweight" no equipment except a pull-up bar.
- Respect injuries and limits in answers.limits: leave out or swap exercises that load that area.
- Prefer exercise names the user already uses (history.knownExercises). Otherwise use standard names like "Bench Press", "Romanian Deadlift", "Lat Pulldown".
- weightKg: a sensible starting working weight for the given reps, based on answers.lifts or history.bestLifts. Use 0 when you have nothing to go on; the app then fills in last time's weight.
- Runs: answers.runsPerWeek runs. Put them on non-lifting days first, never a hard run the day before heavy legs, at most one long run. Base distances on their recent runs (new runners 2-5 km). A day may have both a lift and an easy run.
- Rest days: rest true, no exercises, run null (or a run if it is a run-only day; then rest is false).
- Day names short, e.g. "Push", "Upper A", "Lower", "Full body", "Easy run", "Rest".

Nutrition:
- goal: "cut", "maintain" or "bulk". Follow answers.foodGoal unless it clearly conflicts with their goal; then pick the better one and explain in the tips.
- pace: "gentle", "standard" or "fast". Beginners bulking: standard. Experienced lifters bulking: gentle. Cutting with lots to lose: standard or fast.
- proteinPerKg: 1.6-2.4 (higher when cutting), or null for the default.
- tips: 2-4 short, practical tips tied to their training (pre/post training food, run days, protein spread).

Style: plain, friendly, short. No emojis. Sentence case.

Reply with JSON only:
{
  "name": string (short name for the split, e.g. "4-day upper lower"),
  "summary": string (2 sentences: why this split fits their goal and schedule),
  "week": [ 7 items, Monday first: { "day": "Mon".."Sun", "rest": boolean, "name": string, "focus": string (muscles or aim), "exercises": [ { "name": string, "sets": number, "reps": number, "weightKg": number } ], "run": null | { "type": "Easy"|"Tempo"|"Long"|"Intervals", "distanceKm": number } } ],
  "progression": [string] (2-4 short rules for adding weight or reps over the weeks),
  "nutrition": { "goal": "cut"|"maintain"|"bulk", "pace": "gentle"|"standard"|"fast", "proteinPerKg": number | null, "tips": [string] }
}`;
  const out = await groq({ model: TEXT_MODEL, system, user: JSON.stringify(body).slice(0, 24000), temperature: 0.4, maxTokens: 3000 });

  const raw = arr(out.week, 7);
  const week = DAYS.map((d, i) => {
    const w = raw.find((x) => x?.day === d) ?? raw[i] ?? {};
    const exercises = arr(w.exercises, 9)
      .map((e) => ({ name: str(e?.name, 60), sets: num(e?.sets, 1, 10), reps: num(e?.reps, 1, 50), weightKg: num(e?.weightKg, 0, 500, 1) }))
      .filter((e) => e.name && e.sets > 0 && e.reps > 0);
    const run = w.run && typeof w.run === 'object' && num(w.run.distanceKm, 0, 60, 1) > 0
      ? { type: oneOf(w.run.type, RUN_TYPES, 'Easy'), distanceKm: num(w.run.distanceKm, 0.5, 60, 1) }
      : null;
    const rest = !exercises.length && !run;
    return { day: d, rest, name: rest ? 'Rest' : str(w.name, 40) || 'Workout', focus: str(w.focus, 80), exercises, run };
  });
  if (!week.some((d) => d.exercises.length)) throw new HttpError(502, 'The AI did not return a usable split, try again');
  const n = out.nutrition || {};
  const ppk = Number(n.proteinPerKg);
  return {
    name: str(out.name, 60) || 'My split',
    summary: str(out.summary, 500),
    week,
    progression: arr(out.progression, 5).map((x) => str(x, 200)).filter(Boolean),
    nutrition: {
      goal: oneOf(n.goal, ['cut', 'maintain', 'bulk'], oneOf(body?.answers?.foodGoal, ['cut', 'maintain', 'bulk'], 'maintain')),
      pace: oneOf(n.pace, ['gentle', 'standard', 'fast'], 'standard'),
      proteinPerKg: Number.isFinite(ppk) && ppk >= 1.2 && ppk <= 3 ? Math.round(ppk * 10) / 10 : null,
      tips: arr(n.tips, 5).map((x) => str(x, 200)).filter(Boolean),
    },
  };
}

async function assistant(body) {
  const msgs = arr(body?.messages, 12)
    .map((m) => ({ role: m?.role === 'assistant' ? 'assistant' : 'user', content: str(m?.content, 2000) }))
    .filter((m) => m.content);
  if (!msgs.length || msgs[msgs.length - 1].role !== 'user') throw new HttpError(400, 'Ask something first');
  const system = `You are the assistant inside "stack", a personal lifting, running and nutrition tracker. You talk with the user about their training, food and progress.

You get their data in "context" (all weights in kg, distances in km; show them in the units they use if they mention them): recent workouts and runs, their split, today's plan and check-ins, food targets and what they ate today, body data and weight trend, sets per week for each muscle over the last 4 weeks, goals, their Quill schedule if connected, and their own reflections.

How to answer:
- Use their actual numbers. Point to specific workouts, lifts, muscles or days when relevant. Don't invent data that isn't there; say what's missing and how to log it.
- Be practical: give a clear recommendation first, then a short reason. Concrete sets, reps, weights, foods or amounts where useful.
- Evidence-based and safe: no extreme diets, no crash cuts, no PED advice. For pain, injury, illness or medical questions, give general guidance and suggest seeing a professional; never diagnose.
- If they ask the app to change something (their split, a plan, targets), explain how to do it in the app: Home (today's workout and check-in), Train (pages picked from the menu at the top: split, workouts, runs, food with targets and saved meals, ask ai (this chat), trends and personal records), Progress (body map, your details, reflections), the + button (log a workout, run or food), Settings.
- Friendly, plain and short: usually 2-6 sentences, or a short list with "- " bullets. No markdown headings, no tables, no emojis.

Reply with JSON only: { "reply": string }`;
  const transcript = msgs.map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n\n');
  const user = `context: ${JSON.stringify(body?.context ?? {}).slice(0, 16000)}\n\nConversation so far:\n${transcript}\n\nWrite the assistant's next reply.`;
  const out = await groq({ model: TEXT_MODEL, system, user, temperature: 0.6, maxTokens: 1200 });
  return { reply: str(out.reply, 4000) };
}

const ROUTES = {
  '/coach': coach,
  '/food/suggest': foodSuggest,
  '/food/estimate': foodEstimate,
  '/food/scan': foodScan,
  '/split/build': splitBuild,
  '/assistant': assistant,
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
      const info = { ok: true, service: 'tracker-ai', version: 5, models: { text: TEXT_MODEL, vision: VISION_MODEL }, keySet: !!GROQ_API_KEY };
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
