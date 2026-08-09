# INTEGRATION CHANGELOG

## Architecture Overview

### Frontend: Multi-Folder Design Files (from AI UI Generator)
The 10 `frontend html_css/*/code.html` files are the primary frontend, served at clean URLs:

| URL | Design File | Description |
|-----|-------------|-------------|
| `/` | `secure_portal_login/code.html` | Login page (no auth required) |
| `/dashboard` | `sovereign_rag_dashboard/code.html` | Admin dashboard |
| `/chat` | `chat_assistant_explorer/code.html` | Chat assistant (admin & user) |
| `/chat/user` | `ai_knowledge_assistant_user_view/code.html` | Simplified user chat view |
| `/knowledge` | `knowledge_base_management/code.html` | Knowledge base management |
| `/logs` | `system_logs_infrastructure_monitoring/code.html` | System logs |
| `/users` | `user_management_access_control/code.html` | User management |
| `/security` | `security_performance_audit/code.html` | Security & audit |
| `/settings` | `platform_configuration_connectivity_status/code.html` | Platform settings |

### Shared Library
`static/shared.js` provides to all pages:
- `API` object (get/post/patch/del/upload with Bearer auth)
- `checkAuth()` — redirects to login if no valid session; enforces RBAC
- `setupNav()` — wires sidebar `href` attributes to clean URLs
- `goHome()` — redirects to dashboard (admin) or chat (user) based on role
- `escapeHtml()` — XSS-safe text rendering
- `formatDate()` — timestamp formatting
- Navigation helpers (`goDashboard`, `goChat`, `goLogs`, `goLogin`, `logout`, etc.)

### Backend API (`app/main.py`)
40 routes under FastAPI:

| Method | Path | Auth | Role | Description |
|--------|------|------|------|-------------|
| GET | `/`, `/dashboard`, `/chat`, etc. | — | — | Serve design pages (auth in JS) |
| POST | `/api/login` | Public | — | Authenticate, returns Bearer token + role |
| POST | `/api/logout` | Bearer | — | Invalidate session |
| GET | `/api/me` | Bearer | — | Current user info |
| GET | `/api/users` | Bearer | admin | List all users |
| POST | `/api/users` | Bearer | admin | Create user |
| DELETE | `/api/users/{id}` | Bearer | admin | Delete user |
| PATCH | `/api/users/{id}/password` | Bearer | self/admin | Change password |
| GET | `/api/conversations` | Bearer | — | List user conversations |
| POST | `/api/conversations` | Bearer | — | Create conversation |
| PATCH | `/api/conversations/{id}` | Bearer | — | Rename conversation |
| DELETE | `/api/conversations/{id}` | Bearer | — | Delete conversation |
| GET | `/api/conversations/{id}/messages` | Bearer | — | Get chat messages |
| POST | `/api/conversations/{id}/messages` | Bearer | — | Add chat message |
| POST | `/ingest` | Bearer | admin | Upload & index document |
| POST | `/query` | Bearer | — | RAG question answering |
| GET | `/status` | Bearer | — | System health status |
| GET | `/metrics` | Bearer | — | Usage metrics |
| GET | `/documents` | Bearer | — | Indexed documents list |
| DELETE | `/index` | Bearer | admin | Purge vector collection |
| GET | `/api/logs/system` | Bearer | admin | Structured system logs |
| GET | `/api/logs/audit` | Bearer | admin | Security audit trail |
| POST | `/api/settings/test-connection` | Bearer | — | Test Ollama/Chroma/Embeddings |
| POST | `/api/evaluate` | Bearer | admin | Trigger RAGAS evaluation (async) |
| GET | `/api/evaluation/latest` | Bearer | — | Latest evaluation report |
| GET | `/health` | Public | — | Docker healthcheck |

## RBAC Implementation

### Backend
- `require_auth(request)` — FastAPI dependency. Extracts Bearer token or X-API-Key. Raises 401 if missing/invalid. Whitelists: `/health`, `/api/login`, `/docs`, `/openapi.json`.
- `require_admin(session)` — Helper that raises 403 if `session["role"] != "admin"`.
- **Admin-only endpoints**: `GET/POST/DELETE /api/users`, `POST /ingest`, `DELETE /index`, `GET /api/logs/system`, `GET /api/logs/audit`, `POST /api/evaluate`.

### Frontend (`static/shared.js`)
- `checkAuth()` called on every page load:
  - If no valid session token → redirect to `/` (login)
  - If role is not admin and page is admin-only → redirect to `/chat`
- `ADMIN_PAGES` = all routes except `/` and `/chat/user`
- `USER_PAGES` = `/chat` only
- Login page uses `goHome()` which redirects admin to `/dashboard` and user to `/chat`

## Per-Page Integration

Each design page has inline JS (after `<script src="/static/shared.js">`) that:
1. Calls `checkAuth()` and `setupNav()` on DOMContentLoaded
2. Fetches data from backend API
3. Updates the design page's existing HTML elements with real data via CSS selectors

### Pages and their API calls:

| Page | API Calls |
|------|-----------|
| Login | `POST /api/login` |
| Dashboard | `GET /status`, `GET /metrics`, `GET /documents` |
| Chat | `POST /query`, `GET /status` |
| Knowledge Base | `POST /ingest`, `GET /documents`, `DELETE /index` |
| System Logs | `GET /status`, `GET /api/logs/system` |
| User Management | `GET /api/users`, `POST /api/users`, `DELETE /api/users/{id}` |
| Security Audit | `GET /status`, `GET /api/logs/audit` |
| Settings | `GET /status`, `POST /api/settings/test-connection` |

## Missing Backend Functions Implemented

The following endpoints did not exist in the original Streamlit backend and were created from scratch:

1. **`POST /api/conversations/{conv_id}/messages`** — Save a chat message to a conversation.
2. **`PATCH /api/users/{user_id}/password`** — Change user password (self or admin).
3. **`GET /api/logs/system`** — Reads structured JSON log file (`data/api.log`).
4. **`GET /api/logs/audit`** — Reads `data/audit.log` for security events. Created `_log_audit()` helper integrated into login.
5. **`POST /api/settings/test-connection`** — Tests Ollama, ChromaDB, and embedding model.
6. **`POST /api/evaluate`** — Triggers async RAGAS evaluation.
7. **`GET /api/evaluation/latest`** — Reads CSV evaluation report and returns aggregated metrics.

## Auth Enforcement Fix

Original `verify_auth()` returned `None` when no API key was configured, making all endpoints public. Replaced with `require_auth()` which always raises 401 unless valid Bearer token or matching X-API-Key is provided.

## Files Created/Modified

- `static/shared.js` — **NEW** — API client, auth, RBAC, navigation, formatting helpers
- `frontend html_css/*/code.html` (10 files) — Added `<script src="/static/shared.js">` and fixed page-specific JS selectors
- `app/main.py` — Added 7 new API routes + 9 page routes + fixed auth enforcement + RBAC
- `static/index.html` — **REMOVED** (superseded by design files)
- `app/__init__.py` — Not modified
- `INTEGRATION_CHANGELOG.md` — This file

## Testing

To run:
1. `uvicorn app.main:app --reload --port 8000`
2. Open `http://localhost:8000` (login page)
3. Login as `admin`/`admin` → redirected to `/dashboard`
4. Navigate sidebar links → each page loads its design file with live data
5. Create user with role "user" → login as that user → redirected to `/chat` only
