# tracker-ai (backend)

Small Node server that the Tracker app calls for AI. It holds your Groq key so the app never does.

| Route | What it does | Groq model |
|---|---|---|
| `GET /health` | Wake-up / status check | none |
| `POST /coach` | Check-in + recent training → rest, recovery, run or workout | `TEXT_MODEL` |
| `POST /food/suggest` | What to eat next, from targets, today's log and your meals | `TEXT_MODEL` |
| `POST /food/estimate` | Macros for a meal described in words | `TEXT_MODEL` |
| `POST /food/scan` | Macros from a food photo | `VISION_MODEL` |

All POST routes need the header `x-app-token` to match `APP_TOKEN`.

## Environment variables (set on Render)

| Key | Value |
|---|---|
| `GROQ_API_KEY` | your key from console.groq.com (starts with `gsk_`) |
| `APP_TOKEN` | any long random string; the same string goes in the app's `src/config.ts` |
| `TEXT_MODEL` | optional, default `openai/gpt-oss-120b` |
| `VISION_MODEL` | optional, default `qwen/qwen3.8-27b` (Groq's image model; it is a preview model, swap it here if Groq retires it) |

## Run locally

```
set GROQ_API_KEY=gsk_...
set APP_TOKEN=test
npm start
```
Then open http://localhost:3000/health.

No npm packages needed; Node 18 or newer.
