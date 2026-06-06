# FitLife Gym WhatsApp Agent

A FastAPI-based WhatsApp chatbot for a gym, powered by Groq, LangGraph, LangChain tools, SQLite, and a local RAG knowledge base.

The agent can answer gym FAQs, explain membership plans, handle trial-class interest, show trainer/class information, and send replies through the WhatsApp Cloud API.

## Features

- WhatsApp Cloud API webhook receiver
- Groq LLM integration through `langchain-groq`
- LangGraph tool-use loop
- Local RAG over `knowledge.txt` using Sentence Transformers
- SQLite database for members, class schedules, trial bookings, and class registrations
- Conversation memory per phone number
- Deterministic WhatsApp flows for common gym conversations
- Cleaner error logging for WhatsApp API failures

## Project Structure

```text
Gym Whatsapp agent/
|-- agent.py          # FastAPI app, LangGraph agent, RAG, webhook routes
|-- db.py             # SQLite schema, seed data, DB helper functions
|-- whatsapp.py       # WhatsApp Cloud API send helpers
|-- knowledge.txt     # Gym knowledge base for plans, timings, offers, trainers
|-- gym.db            # Local SQLite database, created automatically
|-- requirements.txt  # Python dependencies
|-- .env.example      # Example environment variables
`-- README.md         # Setup and usage guide
```

## Prerequisites

- Python 3.10 or newer
- A Groq API key
- A Meta Developer app with WhatsApp Cloud API enabled
- A WhatsApp test number or production WhatsApp Business phone number
- `cloudflared`, ngrok, or another HTTPS tunnel for webhook testing

## 1. Clone The Project

```powershell
git clone <your-repository-url>
cd "Gym Whatsapp agent"
```

If you already have the folder locally, open a terminal in:

```powershell
D:\Project\Gym Whatsapp agent
```

## 2. Create A Virtual Environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, use:

```powershell
venv\Scripts\activate.bat
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

The first startup may download the Sentence Transformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This can take a little time on the first run.

## 4. Configure Environment Variables

Create `.env` from the example:

```powershell
copy .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key
MODEL=llama-3.3-70b-versatile

WHATSAPP_TOKEN=your_meta_whatsapp_access_token
WHATSAPP_PHONE_NUMBER_ID=your_whatsapp_phone_number_id
WHATSAPP_WABA_ID=your_whatsapp_business_account_id
WHATSAPP_VERIFY_TOKEN=choose_any_secret_verify_token
GRAPH_API_VERSION=v25.0

DEMO_MEMBER_PHONE=919876543210
```

Important notes:

- `WHATSAPP_TOKEN` must be valid. Temporary Meta tokens expire.
- `WHATSAPP_PHONE_NUMBER_ID` is not your phone number. It is the Phone Number ID from Meta.
- `DEMO_MEMBER_PHONE` should be the test member phone number without `+`.
- Restart Uvicorn after every `.env` change.

## 5. Run The App Locally

```powershell
venv\Scripts\uvicorn.exe agent:app --host 127.0.0.1 --port 8000
```

Expected output:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Open the health check:

```powershell
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "llama-3.3-70b-versatile",
  "corpus_chunks": 54
}
```

## 6. Expose Localhost With HTTPS

Meta webhooks require a public HTTPS URL.

Using Cloudflare Tunnel:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Copy the generated URL, for example:

```text
https://example.trycloudflare.com
```

Your webhook callback URL will be:

```text
https://example.trycloudflare.com/webhook
```

## 7. Configure Meta WhatsApp Webhook

In Meta Developer Dashboard:

1. Open your app.
2. Go to WhatsApp > Configuration.
3. Set Callback URL:

```text
https://your-public-url/webhook
```

4. Set Verify Token to the same value as:

```env
WHATSAPP_VERIFY_TOKEN=...
```

5. Click Verify and Save.
6. Subscribe to the `messages` webhook field.

## 8. Subscribe The App To WhatsApp Events

Run this once after webhook verification:

```powershell
curl -X POST "https://graph.facebook.com/v25.0/<WHATSAPP_WABA_ID>/subscribed_apps" `
  -H "Authorization: Bearer <WHATSAPP_TOKEN>"
```

Replace:

- `<WHATSAPP_WABA_ID>` with your WhatsApp Business Account ID
- `<WHATSAPP_TOKEN>` with your active Meta access token

## 9. Add Test Recipients

If you are using Meta's test WhatsApp number:

1. Go to WhatsApp > API Setup.
2. Add your personal WhatsApp number as a recipient.
3. Verify the OTP on WhatsApp.
4. Send a message to the test business number.

If the recipient is not added or verified, Meta may reject sends.

## 10. Try A Conversation

Example messages:

```text
Hello
I want to see all plans
Add 1499 plan for me
Rajat Paliwal
Main Alwar Branch
Yes
2026-07-07 at 12:30
```

The bot should guide the user through:

- Plan discovery
- Plan selection
- Name capture
- Branch selection
- Joining instructions
- Optional free trainer assessment booking

## Database Behavior

The app uses local SQLite:

```text
gym.db
```

Tables:

```text
members
class_schedule
trial_bookings
class_registrations
```

What gets inserted automatically:

- `class_schedule` is seeded by `db.init_db()`.
- `members` gets a demo row only if `DEMO_MEMBER_PHONE` is set.
- `trial_bookings` gets rows only when a trial or assessment booking is actually saved.
- `class_registrations` gets rows only when a class registration succeeds.

To inspect bookings:

```powershell
venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('gym.db'); c.row_factory=sqlite3.Row; rows=c.execute('select * from trial_bookings order by id desc').fetchall(); [print(dict(r)) for r in rows]"
```

## Important Best Practices

### Keep Secrets Out Of Git

Never commit `.env`.

Your `.gitignore` should include:

```text
.env
venv/
__pycache__/
*.pyc
gym.db
```

For production, use environment variables from your hosting platform instead of storing secrets in files.

### Use A Permanent Meta Token For Production

Temporary WhatsApp tokens expire and cause:

```text
401 Unauthorized
OAuthException code 190
Authentication Error
```

For production:

1. Create a Meta Business System User.
2. Assign WhatsApp permissions.
3. Generate a long-lived/permanent token.
4. Give it `whatsapp_business_messaging`.

### Restart After Environment Changes

The app reads `.env` at startup. After changing token, phone number ID, model, or verify token:

```powershell
CTRL+C
venv\Scripts\uvicorn.exe agent:app --host 127.0.0.1 --port 8000
```

### Do Not Trust LLM Text Alone For Bookings

The best pattern is:

- Bot says booked only after `db.book_trial(...)` returns `success`.
- Log the database insert.
- Notify the gym owner after a confirmed booking.

If you want owner notifications, add an owner phone number in `.env`:

```env
OWNER_WHATSAPP_PHONE=919999999999
```

Then send an internal WhatsApp message after successful booking.

### Keep Knowledge Updated

Edit `knowledge.txt` for:

- Prices
- Offers
- Trainers
- Class schedules
- Policies
- Branch details

Restart the server after editing `knowledge.txt`, because embeddings are rebuilt at startup.

## Troubleshooting

### `401 Unauthorized`, `OAuthException code 190`

Your WhatsApp token is invalid or expired.

Fix:

1. Generate a new token in Meta Developer Dashboard.
2. Update `WHATSAPP_TOKEN` in `.env`.
3. Restart Uvicorn.

### `403 Forbidden`

Common causes:

- Recipient is not added as a test recipient.
- Token lacks WhatsApp messaging permission.
- Wrong `WHATSAPP_PHONE_NUMBER_ID`.
- App is not subscribed to webhook events.
- You are outside the allowed messaging window.

Check the log line:

```text
[WHATSAPP] send_text failed: status=... body=...
```

Meta's JSON body usually explains the exact issue.

### Webhook Verification Fails

Check:

- Callback URL ends with `/webhook`.
- Tunnel is running.
- `WHATSAPP_VERIFY_TOKEN` in `.env` matches Meta exactly.
- Uvicorn is running on port `8000`.

### Bot Replies But No WhatsApp Message Arrives

Check server logs for:

```text
[WHATSAPP] send_text failed
```

If you see `HTTP/1.1 200 OK`, Meta accepted the message.

### `gym.db` Has Empty Tables

This can be normal:

- `trial_bookings` is empty until a booking is saved.
- `class_registrations` is empty until a class registration succeeds.
- `members` is empty unless `DEMO_MEMBER_PHONE` is configured.

### First Startup Is Slow

The embedding model may download/load on first startup. Later starts are faster if the model is cached.

## Useful Local Commands

Run server:

```powershell
venv\Scripts\uvicorn.exe agent:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Reset one user's memory:

```powershell
curl -X POST http://127.0.0.1:8000/reset/918290406024
```

Inspect trial bookings:

```powershell
venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('gym.db'); c.row_factory=sqlite3.Row; [print(dict(r)) for r in c.execute('select * from trial_bookings order by id desc')]"
```

## Production Checklist

- Use a permanent Meta access token.
- Deploy behind HTTPS.
- Store secrets in environment variables.
- Add owner/admin notification for confirmed bookings.
- Add persistent conversation state if running multiple app workers.
- Back up `gym.db` or use a managed database.
- Add structured logging.
- Add tests for booking and signup flows.
- Monitor WhatsApp API errors.

## License

Private project. Add your license here if you plan to distribute it.
