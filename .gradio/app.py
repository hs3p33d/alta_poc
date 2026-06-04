"""
ALTA AI Test Intelligence Agent — POC with RAG
===============================================
Solves 504 timeout by using RAG (Retrieval Augmented Generation):
- Splits 1500+ line SRS into chunks at startup
- Embeds every chunk using Gemini embedding model (free)
- On each query, finds top-5 most relevant chunks using cosine similarity
- Sends ONLY those 5 chunks to Gemini — not the full document
- Result: fast, accurate, no timeouts, scales to any document size

Requirements:
    pip install gradio google-genai openpyxl python-dotenv numpy
"""

import os, sys, json, time, math
import numpy as np
import gradio as gr
from google import genai
from google.genai import types
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════
# STARTUP — API KEY
# ═══════════════════════════════════════════════════════════════
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not API_KEY or "paste_your" in API_KEY:
    print("\n" + "="*60)
    print("  ERROR: GEMINI_API_KEY not set in .env file")
    print("  1. Open .env in this folder")
    print("  2. Set GEMINI_API_KEY=your_actual_key")
    print("  3. Save and re-run")
    print("  Free key: https://aistudio.google.com")
    print("="*60 + "\n")
    sys.exit(1)

AI           = genai.Client(api_key=API_KEY)
GEN_MODEL    = "gemini-2.5-flash"
EMBED_MODEL  = "gemini-embedding-001"

print(f"✅  Connected — gen:{GEN_MODEL}  embed:{EMBED_MODEL}")

# ═══════════════════════════════════════════════════════════════
# LOAD FILES
# ═══════════════════════════════════════════════════════════════
def read_file(path, label):
    try:
        with open(path, encoding="utf-8") as f:
            t = f.read().strip()
        print(f"✅  {label}: {len(t):,} chars  ({path})")
        return t
    except FileNotFoundError:
        print(f"⚠️   {label} not found ({path}) — using built-in sample")
        return None

_srs   = read_file("my_srs.txt",      "SRS document")
_proto = read_file("my_protocol.txt", "Protocol template")

FALLBACK_SRS = """
SRS-001: CO displayed in real time. Range: 2.0-8.0 L/min. Alert if outside range for 5s.
SRS-002: SpO2 alarm triggers below 92% for 10+ seconds in standard mode. Audio + visual.
SRS-003: HPI alarm triggers within 5s when HPI < 85 in high-acuity mode.
SRS-004: Three modes: standard, low-acuity, high-acuity. Mode changes logged with timestamp+userID.
SRS-005: Trend data stored 72h minimum. CSV export in 30s.
SRS-006: SVV displayed as percentage. Warning when SVV > 15%.
SRS-007: CAI updated every 5s. Alert if CAI outside 0.0-1.0.
SRS-008: Supports Swan-Ganz, FloTrac, ClearSight simultaneously.
SRS-009: Unacknowledged critical alarm escalates at 60s — louder audio + red flash.
SRS-010: Screen locks after 5min inactivity. PIN or badge to unlock. All unlocks logged.
"""

FALLBACK_PROTO = """
Test Case ID: TC-001
SRS Reference: SRS-002
Title: SpO2 Alarm Trigger Verification — Standard Mode
Testing Technique: Boundary Value Analysis
Preconditions:
- ALTA simulation launched in CTA
- Mode: STANDARD
- BAIT connected, SpO2 baseline at 96%
Step 1: Set BAIT SpO2 to 93%. Expected: No alarm. Pass/Fail: [ ]
Step 2: Set BAIT SpO2 to 92%, start timer. Expected: No alarm first 9s. Pass/Fail: [ ]
Step 3: Hold 92% for 11s. Expected: Alarm triggers 10-12s, audio+visual. Pass/Fail: [ ]
Step 4: Set BAIT SpO2 to 96%. Expected: Alarm clears within 5s. Pass/Fail: [ ]
Post Conditions: Reset simulator. Log result. Raise defect if any step fails.
Notes: Repeat 3 times to confirm timing consistency.
"""

SRS_RAW       = _srs   or FALLBACK_SRS
PROTO_TEMPLATE = _proto or FALLBACK_PROTO

# ═══════════════════════════════════════════════════════════════
# RAG — CHUNKING + INDEXING
# ═══════════════════════════════════════════════════════════════
CHUNK_SIZE    = 400   # chars per chunk — keeps well within embedding token limit
CHUNK_OVERLAP = 80    # overlap to avoid cutting context mid-sentence
TOP_K         = 6     # number of chunks to retrieve per query

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Batches to stay within API limits."""
    embeddings = []
    batch_size = 20  # API limit per call
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = AI.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
            )
            for emb in resp.embeddings:
                embeddings.append(emb.values)
            time.sleep(0.3)  # stay within free-tier rate limit
        except Exception as e:
            print(f"⚠️  Embedding batch {i//batch_size} error: {e}")
            # Fallback: zero vectors so app still runs
            for _ in batch:
                embeddings.append([0.0] * 768)
    return embeddings

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def retrieve(query: str, chunks: list[str],
             chunk_embs: list[list[float]], k: int = TOP_K) -> str:
    """Find top-k most relevant chunks for a query."""
    try:
        q_resp = AI.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
        )
        q_emb = q_resp.embeddings[0].values
    except Exception as e:
        print(f"⚠️  Query embedding error: {e} — returning full doc sample")
        return "\n\n".join(chunks[:k])

    scores = [(cosine_similarity(q_emb, ce), i)
              for i, ce in enumerate(chunk_embs)]
    scores.sort(reverse=True)
    top = [chunks[i] for _, i in scores[:k]]
    return "\n\n---\n\n".join(top)

# ── Build the index at startup ────────────────────────────────
print("\n📚  Building SRS index...")
SRS_CHUNKS = chunk_text(SRS_RAW)
print(f"   {len(SRS_CHUNKS)} chunks from {len(SRS_RAW):,} chars")
print(f"   Embedding {len(SRS_CHUNKS)} chunks (this runs once at startup)...")
SRS_EMBEDDINGS = embed_batch(SRS_CHUNKS)
print(f"✅  Index ready — {len(SRS_EMBEDDINGS)} vectors\n")

# ═══════════════════════════════════════════════════════════════
# GEMINI GENERATE
# ═══════════════════════════════════════════════════════════════
def generate(prompt: str, max_tokens: int = 2048) -> str:
    try:
        resp = AI.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )
        text = (resp.text or "").strip()
        return text or "The model returned an empty response. Please rephrase and try again."
    except Exception as e:
        err = str(e)
        if "504" in err or "timeout" in err.lower():
            return "⚠️ Request timed out. Please try a shorter or more specific question."
        if "429" in err:
            return "⚠️ Rate limit hit. Please wait 30 seconds and try again."
        return f"⚠️ API error: {err}"

# ═══════════════════════════════════════════════════════════════
# FEATURE 1 — Q&A
# ═══════════════════════════════════════════════════════════════
def qa_answer(question: str) -> str:
    if not question or not question.strip():
        return "⚠️ Please type a question first."

    context = retrieve(question, SRS_CHUNKS, SRS_EMBEDDINGS)

    prompt = f"""You are an expert AI assistant for a medical device QA team
testing the HemoSphere Alta hemodynamic monitoring platform by BD Medical.

The following are the MOST RELEVANT sections from the SRS requirements
document, selected specifically for this question:

━━━ RELEVANT SRS SECTIONS ━━━
{context}

━━━ PROTOCOL EXAMPLE FOR REFERENCE ━━━
{PROTO_TEMPLATE[:600]}

ANSWER RULES:
- Answer from the SRS sections provided above
- If the exact value is in the context, state it precisely and cite the SRS ID
- If the context does not fully cover the question, say what you know
  and note that the full document may have more detail
- Be clear, professional, and specific
- Format with bullet points or numbered lists where helpful

QUESTION: {question}

Answer:"""

    return generate(prompt, max_tokens=1200)

# ═══════════════════════════════════════════════════════════════
# FEATURE 2 — PROTOCOL WRITER
# ═══════════════════════════════════════════════════════════════
def proto_generate(srs_id: str, description: str, tc_id: str):
    errors = []
    if not srs_id.strip():  errors.append("SRS ID required")
    if not description.strip(): errors.append("Description required")
    if not tc_id.strip():   errors.append("Test Case ID required")
    if errors:
        return "⚠️ " + " | ".join(errors), None

    # Get relevant SRS context for this specific requirement
    query   = f"{srs_id} {description}"
    context = retrieve(query, SRS_CHUNKS, SRS_EMBEDDINGS)

    prompt = f"""You are a senior QA engineer writing a formal test protocol
for the HemoSphere Alta medical device QA team.

━━━ RELATED SRS CONTEXT ━━━
{context}

━━━ PROTOCOL FORMAT — FOLLOW EXACTLY ━━━
{PROTO_TEMPLATE}

REQUIREMENT TO TEST:
SRS ID: {srs_id}
Test Case ID: {tc_id}
Description: {description}

RULES:
- Follow the EXACT format of the example above — same field names and layout
- Auto-select the best technique:
    Numeric thresholds → Boundary Value Analysis
    Mode/state changes → State Transition Testing
    Multiple conditions → Decision Table Testing
    Input groups → Equivalence Partitioning
- Write EVERY step: numbered action + specific expected result + Pass/Fail: [ ]
- Include: Preconditions, Steps, Post Conditions, Notes
- Be specific about simulator inputs (e.g. "Set BAIT HPI input to 84")
- Protocol must be complete and immediately executable by a tester

Write the complete protocol now:"""

    text = generate(prompt, max_tokens=2200)
    if text.startswith("⚠️"):
        return text, None

    try:
        fp = build_excel(text, tc_id, srs_id)
        return text, fp
    except Exception as e:
        return text + f"\n\n⚠️ Excel export error: {e}", None


def build_excel(protocol_text: str, tc_id: str, srs_id: str) -> str:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Protocol"

    NAVY, TEAL, WHITE = "1E2761", "028090", "FFFFFF"
    LIGHT, ALT  = "EEF4FF", "F8FAFF"
    META, RED   = "F0F4FF", "DC2626"
    WARN        = "FFF4F4"

    def S(cell, bold=False, color=NAVY, bg=None, sz=10,
          ha="left", va="top", wrap=False, italic=False):
        cell.font      = Font(name="Calibri", size=sz, bold=bold,
                              color=color, italic=italic)
        cell.alignment = Alignment(horizontal=ha, vertical=va, wrap_text=wrap)
        if bg:
            cell.fill = PatternFill("solid", fgColor=bg)

    for col, w in zip("ABCDE", [8, 52, 40, 9, 9]):
        ws.column_dimensions[col].width = w

    ws.merge_cells("A1:E1")
    ws["A1"].value = "ALTA AI — Generated Test Protocol"
    S(ws["A1"], bold=True, color=WHITE, bg=NAVY, sz=13,
      ha="center", va="center")
    ws.row_dimensions[1].height = 32

    ws.merge_cells("A2:E2")
    ws["A2"].value = (f"TC: {tc_id}  |  SRS: {srs_id}  |  "
                      "AI Generated  |  ⚠ Review before formal use")
    S(ws["A2"], italic=True, color=WHITE, bg=TEAL, sz=10,
      ha="center", va="center")
    ws.row_dimensions[2].height = 18

    for c, h in enumerate(
            ["Step", "Action", "Expected Result", "Pass", "Fail"], 1):
        cell = ws.cell(row=3, column=c, value=h)
        S(cell, bold=True, color=WHITE, bg=NAVY, sz=11,
          ha="center", va="center")
    ws.row_dimensions[3].height = 22

    row, step = 4, 1
    for line in protocol_text.split("\n"):
        ln    = line.strip()
        if not ln: continue
        shade = LIGHT if row % 2 == 0 else ALT
        is_step = (ln.lower().startswith("step ") or
                   (len(ln) > 2 and ln[0].isdigit() and ln[1] in ".:"))
        if is_step:
            parts  = ln.split("Expected Result:", 1)
            action = (parts[0].split(":", 1)[-1].strip()
                      if ":" in parts[0] else parts[0])
            exp    = parts[1].strip() if len(parts) > 1 else ""
            for c, v in enumerate([step, action, exp, "[ ]", "[ ]"], 1):
                cell = ws.cell(row=row, column=c, value=v)
                S(cell, bg=shade,
                  ha="center" if c in (1, 4, 5) else "left",
                  bold=(c == 1), wrap=(c in (2, 3)))
            ws.row_dimensions[row].height = 36
            step += 1
        else:
            ws.merge_cells(f"A{row}:E{row}")
            cell = ws.cell(row=row, column=1, value=ln)
            S(cell, bold=(":" in ln[:30] and len(ln) < 60),
              color=NAVY, bg=META, wrap=True, va="center")
            ws.row_dimensions[row].height = 17
        row += 1

    ws.merge_cells(f"A{row+1}:E{row+1}")
    fw = ws.cell(row=row+1, column=1,
                 value=("⚠ AI GENERATED — Mandatory human review "
                        "before use in formal testing"))
    S(fw, bold=True, italic=True, color=RED, bg=WARN,
      sz=9, ha="center", va="center")

    fname = f"{tc_id}_Protocol.xlsx"
    wb.save(fname)
    return fname

# ═══════════════════════════════════════════════════════════════
# FEATURE 3 — TECHNIQUE ADVISOR
# ═══════════════════════════════════════════════════════════════
def tech_advise(scenario: str) -> str:
    if not scenario.strip():
        return "⚠️ Please describe your test scenario."

    context = retrieve(scenario, SRS_CHUNKS, SRS_EMBEDDINGS)

    prompt = f"""You are a world-class software testing expert for medical devices.

━━━ RELEVANT SRS CONTEXT ━━━
{context}

SCENARIO: {scenario}

Provide this exact structure:

**RECOMMENDED TECHNIQUE**
Name + why it fits in 2-3 sentences.

**HOW TO APPLY IT — ALTA SPECIFIC**
Step-by-step using CTA/BAIT simulators. Include specific input values.

**4 READY-TO-USE TEST CASES**
Each: Action → Expected Result. Numbered.

**BOUNDARY VALUES / EDGE CASES**
Specific values or conditions required for thorough coverage.

**COMMON MISTAKES**
Two mistakes teams make for this scenario and how to avoid them."""

    return generate(prompt, max_tokens=1600)

# ═══════════════════════════════════════════════════════════════
# FEATURE 4 — COVERAGE CHECKER
# ═══════════════════════════════════════════════════════════════
def cov_check(tc_list: str) -> str:
    if not tc_list.strip():
        return "⚠️ Please paste your test case list."

    # Only retrieve relevant SRS chunks based on the test list provided
    context = retrieve(tc_list, SRS_CHUNKS, SRS_EMBEDDINGS, k=10)

    # Also include a summary of all SRS IDs found
    srs_ids = []
    for line in SRS_RAW.split("\n"):
        for word in line.split():
            if word.startswith("SRS-") and word.replace("SRS-", "").replace(":", "").isdigit():
                sid = word.rstrip(":")
                if sid not in srs_ids:
                    srs_ids.append(sid)

    prompt = f"""You are a QA coverage analyst for the HemoSphere Alta device.

━━━ ALL SRS IDs FOUND IN DOCUMENT ━━━
{", ".join(srs_ids) if srs_ids else "See context below"}

━━━ RELEVANT SRS CONTENT ━━━
{context}

━━━ TEST CASES PROVIDED ━━━
{tc_list}

Provide:

**COVERAGE TABLE**
For each SRS ID: ✅ Covered / ❌ Not covered / ⚠️ Partially covered

**OVERALL COVERAGE**
X of Y SRS requirements covered = Z%

**CRITICAL GAPS**
List uncovered SRS IDs, their requirement title, and patient safety risk.

**TOP 3 PRIORITY NEW TEST CASES**
Write 3 specific test cases for the highest-risk gaps. Include SRS ID, title, key steps.

**QUALITY OBSERVATIONS**
Any SRS with only weak or single coverage that needs more test cases."""

    return generate(prompt, max_tokens=2000)

# ═══════════════════════════════════════════════════════════════
# UI CSS
# ═══════════════════════════════════════════════════════════════
CSS = """
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background: #f0f4ff !important;
}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* Header */
.alta-header {
    background: linear-gradient(135deg, #0b1740 0%, #1a3a8f 55%, #028090 100%);
    padding: 30px 36px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 10px 40px rgba(11,23,64,0.35);
    position: relative; overflow: hidden;
}
.alta-header::before {
    content: '';
    position: absolute; top: -60%; right: -5%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 65%);
    border-radius: 50%;
}

/* Nav */
.nav-wrap {
    background: white;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 8px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    display: flex; gap: 8px;
}
.nav-btn button {
    border-radius: 9px !important;
    height: 46px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
}

/* Feature wrapper */
.feat-wrap {
    background: white;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    margin-bottom: 12px;
}
.feat-head {
    background: linear-gradient(90deg, #0b1740 0%, #1a3a8f 100%);
    padding: 18px 24px;
}
.feat-body { padding: 24px; }

/* Inputs */
.gr-textbox textarea, .gr-textbox input {
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 9px !important;
    font-size: 14px !important;
    background: #fafbff !important;
    transition: all 0.2s !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #1a3a8f !important;
    box-shadow: 0 0 0 3px rgba(26,58,143,0.12) !important;
    background: white !important;
}

/* Output */
.out-box textarea {
    background: #f7f9ff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 9px !important;
    font-size: 13.5px !important;
    line-height: 1.8 !important;
    color: #1e293b !important;
}

/* Buttons */
.btn-primary button {
    background: linear-gradient(135deg, #0b1740, #1a3a8f) !important;
    border: none !important; color: white !important;
    border-radius: 9px !important; font-weight: 700 !important;
    font-size: 14px !important; height: 46px !important;
    box-shadow: 0 3px 12px rgba(11,23,64,0.28) !important;
    transition: all 0.2s !important;
}
.btn-primary button:hover {
    background: linear-gradient(135deg, #1a3a8f, #028090) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(11,23,64,0.38) !important;
}
.btn-secondary button {
    background: #f8faff !important;
    border: 1.5px solid #c7d2fe !important;
    color: #1a3a8f !important; border-radius: 8px !important;
    font-size: 12px !important; font-weight: 600 !important;
    transition: all 0.15s !important;
}
.btn-secondary button:hover {
    background: #1a3a8f !important; color: white !important;
    border-color: #1a3a8f !important;
}
.btn-danger button {
    background: white !important;
    border: 1.5px solid #fca5a5 !important;
    color: #dc2626 !important; border-radius: 9px !important;
    font-weight: 600 !important;
}

/* Side cards */
.side-card {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 4px solid #1a3a8f;
    border-radius: 9px; padding: 14px 16px;
    margin-bottom: 10px; font-size: 13px; color: #1e3a5f;
}
.side-warn {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-left: 4px solid #ea580c;
    border-radius: 9px; padding: 14px 16px;
    margin-bottom: 10px; font-size: 13px; color: #7c2d12;
}
.side-ok {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 9px; padding: 14px 16px;
    margin-bottom: 10px; font-size: 13px; color: #14532d;
}
label span { font-weight: 700 !important; color: #1e2761 !important; font-size: 13px !important; }
"""

# ═══════════════════════════════════════════════════════════════
# BUILD UI
# ═══════════════════════════════════════════════════════════════
srs_count = sum(1 for l in SRS_RAW.split("\n")
                if any(w.startswith("SRS-") for w in l.split()))

with gr.Blocks(css=CSS, title="ALTA AI Agent") as demo:

    # ── HEADER ───────────────────────────────────────────────
    gr.HTML(f"""
    <div class="alta-header">
      <div style="position:relative;z-index:1;display:flex;
                  justify-content:space-between;align-items:center;
                  flex-wrap:wrap;gap:16px">
        <div style="display:flex;align-items:center;gap:18px">
          <div style="font-size:52px;line-height:1;filter:drop-shadow(0 3px 8px rgba(0,0,0,0.3))">🏥</div>
          <div>
            <h1 style="color:white;margin:0;font-size:27px;font-weight:800;
                       letter-spacing:-0.6px;text-shadow:0 2px 12px rgba(0,0,0,0.25)">
              ALTA AI Test Intelligence Agent
            </h1>
            <p style="color:#93c5fd;margin:5px 0 0 0;font-size:13.5px;font-weight:400">
              HemoSphere Alta &nbsp;·&nbsp; BD Medical &nbsp;·&nbsp;
              Google Gemini 2.5 Flash &nbsp;·&nbsp; RAG-Powered &nbsp;·&nbsp; POC v2.0
            </p>
          </div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.18);
                      border-radius:10px;padding:12px 18px;text-align:center;min-width:80px">
            <div style="color:#7dd3fc;font-size:24px;font-weight:800">{len(SRS_CHUNKS)}</div>
            <div style="color:#bfdbfe;font-size:11px;margin-top:2px">SRS Chunks Indexed</div>
          </div>
          <div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.18);
                      border-radius:10px;padding:12px 18px;text-align:center;min-width:80px">
            <div style="color:#86efac;font-size:24px;font-weight:800">4</div>
            <div style="color:#bfdbfe;font-size:11px;margin-top:2px">AI Features</div>
          </div>
          <div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.18);
                      border-radius:10px;padding:12px 18px;text-align:center;min-width:80px">
            <div style="color:#fde68a;font-size:24px;font-weight:800">RAG</div>
            <div style="color:#bfdbfe;font-size:11px;margin-top:2px">No Timeouts</div>
          </div>
        </div>
      </div>
      <div style="position:relative;z-index:1;margin-top:14px;padding-top:12px;
                  border-top:1px solid rgba(255,255,255,0.14)">
        <p style="color:#93c5fd;font-size:12px;margin:0">
          ⚡ RAG mode — your full SRS is indexed into {len(SRS_CHUNKS)} searchable chunks.
          Only the most relevant sections are sent to AI per query — no timeouts, fast responses.
          &nbsp;&nbsp;⚠️ All outputs require human review before formal testing use.
        </p>
      </div>
    </div>
    """)

    # ── NAV ──────────────────────────────────────────────────
    with gr.Row(elem_classes=["nav-wrap"]):
        nav_qa    = gr.Button("💬  Q&A Assistant",    variant="primary",   scale=1, elem_classes=["nav-btn"])
        nav_proto = gr.Button("📋  Protocol Writer",  variant="secondary", scale=1, elem_classes=["nav-btn"])
        nav_tech  = gr.Button("🎯  Technique Advisor",variant="secondary", scale=1, elem_classes=["nav-btn"])
        nav_cov   = gr.Button("📊  Coverage Checker", variant="secondary", scale=1, elem_classes=["nav-btn"])

    # ── PAGE 1 — Q&A ─────────────────────────────────────────
    with gr.Column(visible=True) as pg_qa:
        gr.HTML("""
        <div class="feat-wrap"><div class="feat-head">
          <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:30px">💬</span>
            <div>
              <h2 style="color:white;margin:0;font-size:19px;font-weight:700">Knowledge Base Q&A</h2>
              <p style="color:#93c5fd;margin:3px 0 0;font-size:12.5px">
                Ask anything about ALTA specs, alarms, parameters, or test requirements.
                AI searches your full SRS document and answers with citations.
              </p>
            </div>
          </div>
        </div><div class="feat-body">""")

        with gr.Row():
            with gr.Column(scale=3):
                qa_in = gr.Textbox(
                    label="Your Question",
                    placeholder=(
                        "e.g. What is the SpO2 alarm threshold?\n"
                        "e.g. How does alarm escalation work?\n"
                        "e.g. What monitoring modes are supported?"
                    ),
                    lines=3,
                )
                with gr.Row():
                    qa_btn   = gr.Button("🔍  Ask", variant="primary", scale=4, elem_classes=["btn-primary"])
                    qa_clear = gr.Button("✕ Clear", scale=1, elem_classes=["btn-danger"])

                gr.HTML("<p style='color:#64748b;font-size:12px;margin:10px 0 5px;font-weight:600'>⚡ Quick examples — click to load:</p>")

                qa_ex = [
                    "What is the SpO2 alarm threshold in standard mode and how long before it triggers?",
                    "What is the HPI alarm response time in high-acuity mode?",
                    "How does alarm escalation work when a critical alarm is not acknowledged?",
                    "What monitoring modes does the device support and what is logged on mode change?",
                    "What sensors does the device support simultaneously?",
                    "How does screen lock and user authentication work?",
                ]
                with gr.Row():
                    for ex in qa_ex[:3]:
                        b = gr.Button(ex[:48]+"…", elem_classes=["btn-secondary"], size="sm")
                        b.click(fn=lambda x=ex: x, inputs=[], outputs=[qa_in])
                with gr.Row():
                    for ex in qa_ex[3:]:
                        b = gr.Button(ex[:48]+"…", elem_classes=["btn-secondary"], size="sm")
                        b.click(fn=lambda x=ex: x, inputs=[], outputs=[qa_in])

            with gr.Column(scale=1):
                gr.HTML("""
                <div class="side-card">
                  <strong>📚 How RAG works here</strong><br><br>
                  Your 1500+ line SRS is split into chunks and indexed.
                  Each question retrieves only the most relevant sections
                  — so every answer is fast and accurate regardless of document size.
                </div>
                <div class="side-card" style="margin-top:0">
                  <strong>💡 Best results tip</strong><br><br>
                  Be specific. Include the parameter name (HPI, SpO2, CO),
                  the mode (standard, high-acuity), or the SRS ID.
                </div>
                <div class="side-warn">
                  <strong>⚠️ Verify values</strong><br>
                  Always check cited values against official Polarion SRS before formal testing.
                </div>""")

        qa_out = gr.Textbox(
            label="Answer",
            lines=13,
            interactive=False,
            placeholder="Answer with citations will appear here...",
            elem_classes=["out-box"],
        )
        gr.HTML("</div></div>")

        qa_btn.click(fn=qa_answer,  inputs=[qa_in], outputs=[qa_out])
        qa_clear.click(fn=lambda: ("",""), inputs=[], outputs=[qa_in, qa_out])

    # ── PAGE 2 — PROTOCOL WRITER ──────────────────────────────
    with gr.Column(visible=False) as pg_proto:
        gr.HTML("""
        <div class="feat-wrap"><div class="feat-head">
          <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:30px">📋</span>
            <div>
              <h2 style="color:white;margin:0;font-size:19px;font-weight:700">Protocol Writer</h2>
              <p style="color:#93c5fd;margin:3px 0 0;font-size:12.5px">
                Enter a new SRS requirement. AI generates a complete, ready-to-execute
                test protocol in your exact format with correct testing technique. Downloads as Excel.
              </p>
            </div>
          </div>
        </div><div class="feat-body">""")

        with gr.Row():
            proto_srs = gr.Textbox(label="SRS ID", placeholder="e.g. SRS-003", scale=1)
            proto_tc  = gr.Textbox(label="Test Case ID", placeholder="e.g. TC-002", scale=1)

        proto_desc = gr.Textbox(
            label="Full SRS Requirement Description",
            placeholder="Paste the complete requirement text. More detail = better protocol...",
            lines=5,
        )

        gr.HTML("<p style='color:#64748b;font-size:12px;margin:8px 0 5px;font-weight:600'>⚡ Load an example:</p>")
        proto_ex = [
            ("SRS-003","TC-002",
             "The HPI alarm shall trigger within 5 seconds when the Haemodynamic Predictive Index falls below the configured threshold of 85 in high-acuity monitoring mode. Alarm shall include both audio and visual indicators. Alarm shall clear within 5 seconds when HPI returns above threshold."),
            ("SRS-004","TC-003",
             "The device shall support three monitoring modes: standard, low-acuity, and high-acuity. All mode changes shall be logged with a timestamp and the user ID of the person who made the change. The log shall be stored in tamper-proof audit storage."),
            ("SRS-009","TC-009",
             "If a critical alarm is not acknowledged within 60 seconds, the device shall escalate the alarm with increased audio volume and a flashing red visual indicator. The escalation shall persist until acknowledged by an authenticated user."),
            ("SRS-010","TC-010",
             "The device shall lock the screen after 5 minutes of inactivity. A valid user PIN or badge scan shall be required to unlock. All unlock events shall be logged with user ID and timestamp in the audit log."),
        ]
        with gr.Row():
            for ex in proto_ex:
                lbl = f"{ex[0]}: {ex[2][:38]}…"
                b = gr.Button(lbl, elem_classes=["btn-secondary"], size="sm")
                b.click(fn=lambda e=ex: (e[0],e[1],e[2]),
                        inputs=[], outputs=[proto_srs, proto_tc, proto_desc])

        with gr.Row():
            proto_btn   = gr.Button("⚡  Generate Protocol + Excel",
                                    variant="primary", scale=4, elem_classes=["btn-primary"])
            proto_clear = gr.Button("✕ Clear All", scale=1, elem_classes=["btn-danger"])

        with gr.Row():
            with gr.Column(scale=3):
                proto_out = gr.Textbox(
                    label="Generated Protocol — Review before use",
                    lines=20, interactive=False,
                    placeholder="Protocol will appear here...",
                    elem_classes=["out-box"],
                )
            with gr.Column(scale=1):
                proto_file = gr.File(label="📥 Excel Download", interactive=False)
                gr.HTML("""
                <div class="side-ok">
                  <strong>✅ Excel includes</strong><br>
                  Step table · Pass/Fail columns · All protocol fields · Warning footer
                </div>
                <div class="side-warn">
                  <strong>⚠️ Before use</strong><br>
                  Review every step. Add firmware version and tester name before executing.
                </div>""")

        gr.HTML("</div></div>")

        proto_btn.click(fn=proto_generate,
                        inputs=[proto_srs, proto_desc, proto_tc],
                        outputs=[proto_out, proto_file])
        proto_clear.click(fn=lambda: ("","","","",None),
                          inputs=[],
                          outputs=[proto_srs, proto_tc, proto_desc, proto_out, proto_file])

    # ── PAGE 3 — TECHNIQUE ADVISOR ────────────────────────────
    with gr.Column(visible=False) as pg_tech:
        gr.HTML("""
        <div class="feat-wrap"><div class="feat-head">
          <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:30px">🎯</span>
            <div>
              <h2 style="color:white;margin:0;font-size:19px;font-weight:700">Testing Technique Advisor</h2>
              <p style="color:#93c5fd;margin:3px 0 0;font-size:12.5px">
                Describe what you are testing. AI recommends the right technique
                with ALTA-specific examples, 4 test cases, and boundary values.
              </p>
            </div>
          </div>
        </div><div class="feat-body">""")

        with gr.Row():
            with gr.Column(scale=3):
                tech_in = gr.Textbox(
                    label="Describe your test scenario",
                    placeholder="e.g. Testing the HPI alarm when value crosses 85 threshold in high-acuity mode...",
                    lines=4,
                )
                gr.HTML("<p style='color:#64748b;font-size:12px;margin:8px 0 5px;font-weight:600'>⚡ Quick examples:</p>")
                tech_ex = [
                    "Testing the HPI alarm when the value crosses the 85 threshold in high-acuity mode",
                    "Testing device switching between standard, low-acuity, and high-acuity modes",
                    "Testing SpO2 alarm response time around the 92% saturation threshold",
                    "Testing alarm escalation when alarm is not acknowledged for 60 seconds",
                    "Testing trend data storage and CSV export after 72 hours",
                    "Testing screen lock after 5 minutes of user inactivity",
                ]
                with gr.Row():
                    for ex in tech_ex[:3]:
                        b = gr.Button(ex[:46]+"…", elem_classes=["btn-secondary"], size="sm")
                        b.click(fn=lambda x=ex: x, inputs=[], outputs=[tech_in])
                with gr.Row():
                    for ex in tech_ex[3:]:
                        b = gr.Button(ex[:46]+"…", elem_classes=["btn-secondary"], size="sm")
                        b.click(fn=lambda x=ex: x, inputs=[], outputs=[tech_in])

            with gr.Column(scale=1):
                gr.HTML("""
                <div class="side-card">
                  <strong>📖 Techniques covered</strong><br><br>
                  <b>BVA</b> — Boundary Value Analysis<br>
                  <b>EP</b> — Equivalence Partitioning<br>
                  <b>ST</b> — State Transition Testing<br>
                  <b>DT</b> — Decision Table Testing<br>
                  <b>EG</b> — Error Guessing<br>
                  <b>ET</b> — Exploratory Testing
                </div>""")

        with gr.Row():
            tech_btn   = gr.Button("🎯  Get Recommendation", variant="primary",
                                   scale=4, elem_classes=["btn-primary"])
            tech_clear = gr.Button("✕ Clear", scale=1, elem_classes=["btn-danger"])

        tech_out = gr.Textbox(
            label="Recommendation",
            lines=18, interactive=False,
            placeholder="Recommendation will appear here...",
            elem_classes=["out-box"],
        )
        gr.HTML("</div></div>")

        tech_btn.click(fn=tech_advise, inputs=[tech_in], outputs=[tech_out])
        tech_clear.click(fn=lambda: ("",""), inputs=[], outputs=[tech_in, tech_out])

    # ── PAGE 4 — COVERAGE CHECKER ─────────────────────────────
    with gr.Column(visible=False) as pg_cov:
        gr.HTML("""
        <div class="feat-wrap"><div class="feat-head">
          <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:30px">📊</span>
            <div>
              <h2 style="color:white;margin:0;font-size:19px;font-weight:700">Coverage Checker</h2>
              <p style="color:#93c5fd;margin:3px 0 0;font-size:12.5px">
                Paste your test case list. AI analyses SRS coverage, identifies critical gaps,
                calculates coverage %, and recommends priority new test cases.
              </p>
            </div>
          </div>
        </div><div class="feat-body">""")

        with gr.Row():
            with gr.Column(scale=3):
                cov_in = gr.Textbox(
                    label="Paste your existing test cases here",
                    placeholder=(
                        "TC-001: SpO2 Alarm — Standard Mode (covers SRS-002)\n"
                        "TC-002: HPI Alarm — High Acuity (covers SRS-003)\n"
                        "TC-003: Mode Selection (covers SRS-004)\n"
                        "...\n\nMore detail = more accurate analysis."
                    ),
                    lines=10,
                )
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="side-ok">
                  <strong>📊 You get</strong><br><br>
                  ✅ Per-SRS coverage table<br>
                  ✅ Overall % coverage<br>
                  ✅ Gaps with risk rating<br>
                  ✅ 3 priority new TCs<br>
                  ✅ Quality observations
                </div>
                <div class="side-card">
                  <strong>💡 Tip</strong><br>
                  Include "(covers SRS-XXX)" in each entry for the most accurate analysis.
                </div>""")

        sample_tcs = (
            "TC-001: SpO2 Alarm Trigger Verification - Standard Mode (covers SRS-002)\n"
            "TC-002: HPI Alarm Response Time - High Acuity Mode (covers SRS-003)\n"
            "TC-003: Monitoring Mode Selection and Switching (covers SRS-004)\n"
            "TC-004: Cardiac Output Display and Alert (covers SRS-001)"
        )

        with gr.Row():
            cov_btn    = gr.Button("📊  Analyse Coverage", variant="primary",
                                   scale=3, elem_classes=["btn-primary"])
            cov_sample = gr.Button("📝 Load Sample", scale=1, elem_classes=["btn-secondary"])
            cov_clear  = gr.Button("✕ Clear",        scale=1, elem_classes=["btn-danger"])

        cov_out = gr.Textbox(
            label="Coverage Analysis",
            lines=18, interactive=False,
            placeholder="Analysis will appear here...",
            elem_classes=["out-box"],
        )
        gr.HTML("</div></div>")

        cov_btn.click(fn=cov_check,     inputs=[cov_in], outputs=[cov_out])
        cov_sample.click(fn=lambda: sample_tcs, inputs=[], outputs=[cov_in])
        cov_clear.click(fn=lambda: ("",""), inputs=[], outputs=[cov_in, cov_out])

    # ── NAVIGATION LOGIC ──────────────────────────────────────
    pages = [pg_qa, pg_proto, pg_tech, pg_cov]
    navs  = [nav_qa, nav_proto, nav_tech, nav_cov]

    def switch(i):
        return (
            [gr.update(visible=(j == i)) for j in range(4)] +
            [gr.update(variant="primary" if j == i else "secondary") for j in range(4)]
        )

    nav_qa.click(   fn=lambda: switch(0), inputs=[], outputs=pages+navs)
    nav_proto.click(fn=lambda: switch(1), inputs=[], outputs=pages+navs)
    nav_tech.click( fn=lambda: switch(2), inputs=[], outputs=pages+navs)
    nav_cov.click(  fn=lambda: switch(3), inputs=[], outputs=pages+navs)

    # ── FOOTER ────────────────────────────────────────────────
    gr.HTML("""
    <div style="margin-top:20px;padding:14px 24px;
                background:white;border-radius:12px;
                border:1.5px solid #e2e8f0;
                display:flex;justify-content:space-between;
                align-items:center;flex-wrap:wrap;gap:10px;
                box-shadow:0 2px 10px rgba(0,0,0,0.05)">
      <span style="color:#1e2761;font-weight:700;font-size:13px">
        🏥 ALTA AI Test Intelligence Agent &nbsp;·&nbsp;
        <span style="color:#64748b;font-weight:400">POC v2.0 — RAG Powered</span>
      </span>
      <span style="color:#64748b;font-size:12px">
        In production: auto-syncs from Polarion · 2000+ documents
      </span>
      <span style="color:#dc2626;font-size:12px;font-weight:700">
        ⚠️ All outputs require human review
      </span>
    </div>
    """)

# ═══════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  ALTA AI Agent — RAG Mode — Starting server...")
    print("="*60)
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True,
        show_error=True,
    )