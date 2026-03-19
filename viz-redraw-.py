"""
viz_redraw_app.py  —  Professional Football Viz Redraw
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upload a screenshot → Claude extracts data → redrawn at 1920×1080
Inspired by The Athletic's clean, professional style.

pip install streamlit matplotlib mplsoccer numpy pillow requests scipy
streamlit run viz_redraw_app.py
"""

import io, json, base64, re, textwrap
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from PIL import Image
import requests

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VIZ REDRAW", layout="wide", page_icon="⚽")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif!important;}
.stApp{background:#07090f!important;}
section[data-testid="stSidebar"]{background:#040608!important;border-right:1px solid #111827!important;}
section[data-testid="stSidebar"] *{color:#fff!important;}
.stButton>button{background:#ef4444!important;color:#fff!important;font-weight:800!important;
  border:none!important;border-radius:4px!important;letter-spacing:.06em!important;
  text-transform:uppercase!important;padding:10px 20px!important;}
.stSelectbox>div>div,.stMultiSelect>div>div{background:#0d1424!important;border:1px solid #1e2d4a!important;}
div[data-baseweb="select"]*{background:#0d1424!important;color:#fff!important;}
div[data-baseweb="popover"]*{background:#0d1424!important;color:#fff!important;}
.stTextInput>div>div>input,.stTextArea textarea{background:#0d1424!important;
  border:1px solid #1e2d4a!important;color:#fff!important;}
.stNumberInput input{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important;}
label{color:#6b7280!important;font-size:9px!important;letter-spacing:.14em!important;text-transform:uppercase!important;}
h1,h2,h3{color:#fff!important;} footer{display:none!important;}
.stTabs [data-baseweb="tab"]{color:#6b7280!important;font-weight:700!important;}
.stTabs [aria-selected="true"]{color:#ef4444!important;border-bottom-color:#ef4444!important;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THEMES
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    "Dark Navy": {
        "bg": "#0a0f1c", "pitch": "#0a0f1c", "line": "#1e3a5f",
        "text": "#e2e8f0", "subtext": "#64748b",
        "accent": "#ef4444", "dot_ok": "#ef4444", "dot_bad": "#4b5563",
    },
    "Light (Athletic)": {
        "bg": "#f8f5f0", "pitch": "#f8f5f0", "line": "#9ca3af",
        "text": "#111827", "subtext": "#6b7280",
        "accent": "#ef4444", "dot_ok": "#ef4444", "dot_bad": "#d1d5db",
    },
    "Pure Black": {
        "bg": "#000000", "pitch": "#000000", "line": "#1f2937",
        "text": "#ffffff", "subtext": "#6b7280",
        "accent": "#ef4444", "dot_ok": "#ef4444", "dot_bad": "#374151",
    },
    "Dark Green": {
        "bg": "#0a1a0a", "pitch": "#0d1f0d", "line": "#1a4a1a",
        "text": "#f0fdf4", "subtext": "#6b7280",
        "accent": "#22c55e", "dot_ok": "#22c55e", "dot_bad": "#374151",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## VIZ REDRAW")
    st.markdown("---")
    api_key = st.text_input("Anthropic API Key", type="password", key="api_key")
    st.markdown("---")
    theme_name = st.selectbox("Theme", list(THEMES.keys()), key="theme")
    T = THEMES[theme_name]
    st.markdown("---")
    st.markdown("**LABELS**")
    title_in    = st.text_input("Title",    placeholder="e.g. Matt Grimes", key="title")
    subtitle_in = st.text_input("Subtitle", placeholder="e.g. 2025-26 Season", key="subtitle")
    brand_in    = st.text_input("Brand",    value="@MDUFFY", key="brand")

# ─────────────────────────────────────────────────────────────────────────────
# CANVAS — exact 1920x1080
# ─────────────────────────────────────────────────────────────────────────────
W, H, DPI = 19.2, 10.8, 100

def _canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(T["bg"])
    return fig

def _labels(fig, title, subtitle, brand, ty=0.94, sy=0.89):
    if title:
        fig.text(0.5, ty, title.upper(), fontsize=30, fontweight="900",
                 color=T["text"], ha="center", va="bottom",
                 fontfamily="Montserrat",
                 path_effects=[mpe.withStroke(linewidth=3, foreground=T["bg"])])
    if subtitle:
        fig.text(0.5, sy, subtitle, fontsize=15, fontweight="500",
                 color=T["subtext"], ha="center", va="bottom", fontfamily="Montserrat")
    if brand:
        fig.text(0.97, 0.02, brand, fontsize=12, fontweight="700",
                 color=T["subtext"], ha="right", va="bottom", fontfamily="Montserrat")

def _pitch_ax(fig, left=0.06, bot=0.10, w=0.88, h=0.76):
    ax = fig.add_axes([left, bot, w, h])
    ax.set_facecolor(T["pitch"])
    ax.set_xlim(-2, 107); ax.set_ylim(-2, 70)
    ax.set_aspect("equal"); ax.axis("off")
    return ax

def _draw_pitch(ax, lw=1.6):
    c = T["line"]
    def R(x, y, w, h):
        ax.add_patch(mpatches.Rectangle((x,y),w,h,fill=False,edgecolor=c,lw=lw,zorder=1))
    R(0,0,105,68)
    ax.plot([52.5,52.5],[0,68],color=c,lw=lw,zorder=1)
    ax.add_patch(plt.Circle((52.5,34),9.15,fill=False,edgecolor=c,lw=lw,zorder=1))
    ax.scatter([52.5],[34],s=14,color=c,zorder=1)
    R(0,13.84,16.5,40.32); R(88.5,13.84,16.5,40.32)
    R(0,24.84,5.5,18.32);  R(99.5,24.84,5.5,18.32)
    ax.add_patch(mpatches.Rectangle((-2,29.84),2,8.32,fill=False,edgecolor=c,lw=lw,zorder=1))
    ax.add_patch(mpatches.Rectangle((105,29.84),2,8.32,fill=False,edgecolor=c,lw=lw,zorder=1))
    ax.scatter([11,94],[34,34],s=14,color=c,zorder=1)

def _draw_half_pitch(ax, lw=1.6):
    c = T["line"]
    def R(x,y,w,h):
        ax.add_patch(mpatches.Rectangle((x,y),w,h,fill=False,edgecolor=c,lw=lw,zorder=1))
    ax.plot([52.5,105],[0,0],color=c,lw=lw,zorder=1)
    ax.plot([52.5,105],[68,68],color=c,lw=lw,zorder=1)
    ax.plot([105,105],[0,68],color=c,lw=lw,zorder=1)
    ax.plot([52.5,52.5],[0,68],color=c,lw=lw,zorder=1)
    ax.add_patch(plt.Circle((52.5,34),9.15,fill=False,edgecolor=c,lw=lw,zorder=1,clip_on=True))
    R(88.5,13.84,16.5,40.32)
    R(99.5,24.84,5.5,18.32)
    ax.add_patch(mpatches.Rectangle((105,29.84),2,8.32,fill=False,edgecolor=c,lw=lw,zorder=1))
    ax.scatter([94],[34],s=14,color=c,zorder=1)

def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=T["bg"], bbox_inches=None)
    plt.close(fig)
    return buf.getvalue()

def _cx(v): return float(v) * 105 / 100
def _cy(v): return float(v) * 68  / 100

# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE VISION — separate prompt per viz type
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = {
    "touch_map": """You are a football data extraction engine.
Extract all touch/dot positions from this football pitch screenshot.
Return ONLY valid JSON, no markdown, no explanation.

The pitch is shown landscape (wider than tall). 
Coordinate system: x=0 is LEFT edge, x=100 is RIGHT edge, y=0 is BOTTOM edge, y=100 is TOP edge.
All dots must be INSIDE the pitch: x between 3 and 97, y between 3 and 97.

Return:
{
  "viz_type": "touch_map",
  "team": "...",
  "opponent": "...",
  "competition": "...",
  "date": "...",
  "touch_map": {
    "successful": [{"x": <0-100>, "y": <0-100>}],
    "unsuccessful": [{"x": <0-100>, "y": <0-100>}]
  }
}

Map every visible dot. Successful touches are typically coloured (pink/red). Unsuccessful are grey/white.""",

    "shot_map": """You are a football data extraction engine.
Extract all shot positions from this xG/shot map screenshot.
Return ONLY valid JSON, no markdown, no explanation.

The team is ALWAYS shooting toward the RIGHT side (x=100).
Coordinate system: x=0 LEFT, x=100 RIGHT, y=0 BOTTOM, y=100 TOP.
ALL shots must have x > 52. If you see shots on the left side of the image, the pitch may be flipped — mirror them.
Goal mouth is at x=100, y between 37 and 63.
Penalty spot is approximately x=89, y=50.
Circle/dot SIZE = xG value. Bigger = higher xG (typical range 0.02 to 0.45).
FILLED solid circle = GOAL. EMPTY outline circle = not a goal (saved/blocked/off target).

Return:
{
  "viz_type": "shot_map",
  "team": "...",
  "opponent": "...",
  "competition": "...",
  "date": "...",
  "shot_map": {
    "shots": [{"x": <52-100>, "y": <0-100>, "xg": <0.01-0.5>, "outcome": "goal|saved|blocked|off_target"}],
    "total_xg": <float>,
    "goals": <int>,
    "total_shots": <int>
  }
}""",

    "pass_network": """You are a football data extraction engine.
Extract pass network data from this screenshot. Return ONLY valid JSON, no explanation, no markdown.

STEP 1 — NODES
List every numbered circle ON the pitch. For each:
- id: number inside circle
- x: position left→right within pitch rectangle (0=left edge, 100=right edge)
- y: position bottom→top within pitch rectangle (0=bottom, 100=top)
- name: surname written near the circle — read carefully, it is always there

STEP 2 — EDGES (CRITICAL — read this carefully)
Look at the image. You will see coloured lines drawn between some circles.
Go through the lines ONE BY ONE. For each line:
  - Follow it from one end to the other
  - Which circle does it START at? (shirt number)
  - Which circle does it END at? (shirt number)
  - Add ONLY that pair

DO NOT add an edge between two circles just because they are close together.
DO NOT add an edge unless you can physically trace a drawn line between them.
Some nodes will have ZERO edges — that is fine and correct.
Some nodes will have MANY edges — trace every line from them carefully.

Example: if node 19 has 4 lines drawn from it, you must return 4 edges involving id 19.
Example: if nodes 22, 15, 5 have NO lines between them, return NO edges between them.

Edge thickness → count: thin=4, medium=9, thick=15

STEP 3 — SUBS
Any circles shown outside/below the pitch go in subs_bench.

Return:
{
  "viz_type": "pass_network",
  "team": "...",
  "opponent": "...",
  "competition": "...",
  "date": "...",
  "pass_network": {
    "nodes": [{"id": <int>, "x": <0-100>, "y": <0-100>, "name": "<surname>"}],
    "edges": [{"from": <int>, "to": <int>, "count": <4-15>}],
    "formation": "...",
    "subs_bench": [<int>, ...]
  }
}""",

    "avg_positions": """You are a football data extraction engine.
Extract player positions from this lineup screenshot. Return ONLY valid JSON. No explanation. No markdown. No text before or after the JSON.

The image shows a VERTICAL (portrait) pitch. You must output LANDSCAPE coordinates.
In the output: x=0 is the GK end (left), x=100 is the attacking end (right). y=0 is bottom of pitch, y=100 is top.

STEP 1 — Find the GK (different coloured circle, in goal area). Is GK at TOP or BOTTOM of image?

STEP 2 — For every player measure two things within the pitch rectangle:
  A) How far are they from the GK end? (0%=same end as GK, 100%=opposite end) → this becomes x, scaled to 5-92
  B) How far across the pitch left-to-right are they? (0%=left edge, 100%=right edge) → this becomes y, scaled to 8-92

STEP 3 — Apply the correct y mapping:
  IF GK IS AT BOTTOM: left side of image = HIGH y (y=85), right side = LOW y (y=15)
  IF GK IS AT TOP:    left side of image = LOW y (y=15),  right side = HIGH y (y=85)

STEP 4 — Verify spread. Your x values MUST use the full range:
  GK: x=5-10
  Defenders: x=20-32
  Midfielders: x=38-62  
  Forwards/Wingers: x=68-88
  Strikers: x=78-92
  If your defenders and midfielders have similar x values, you have compressed the range — fix it.

Return ONLY this JSON:
{
  "viz_type": "avg_positions",
  "team": "...",
  "opponent": "...",
  "competition": "...",
  "date": "...",
  "avg_positions": {
    "formation": "...",
    "players": [{"id": <shirt_no>, "x": <5-92>, "y": <8-92>, "name": "...", "position": "GK|CB|LB|RB|DM|CM|AM|LW|RW|ST"}]
  }
}"""
}

def _parse_response(raw):
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    # Find first { and last } in case Claude added explanation text
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response. Claude returned: {raw[:200]}")
    raw = raw[start:end+1]
    return json.loads(raw)

def call_claude(image_bytes, viz_hint, key):
    b64 = base64.standard_b64encode(image_bytes).decode()

    # Pick the right prompt
    hint_lower = viz_hint.lower()
    if "touch" in hint_lower or "heat" in hint_lower:
        prompt = PROMPTS["touch_map"]
    elif "shot" in hint_lower or "xg" in hint_lower:
        prompt = PROMPTS["shot_map"]
    elif "pass" in hint_lower or "network" in hint_lower:
        prompt = PROMPTS["pass_network"]
    elif "position" in hint_lower or "lineup" in hint_lower or "formation" in hint_lower or "avg" in hint_lower:
        prompt = PROMPTS["avg_positions"]
    else:
        prompt = PROMPTS["touch_map"]  # fallback

    r = requests.post("https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": "claude-opus-4-6", "max_tokens": 4096,
              "messages": [{"role": "user", "content": [
                  {"type": "image", "source": {"type": "base64",
                   "media_type": "image/png", "data": b64}},
                  {"type": "text", "text": prompt}
              ]}]}, timeout=90)
    r.raise_for_status()
    raw = r.json()["content"][0]["text"].strip()
    return _parse_response(raw)

# ─────────────────────────────────────────────────────────────────────────────
# DRAW: TOUCH MAP──────────
# DRAW: TOUCH MAP
# ─────────────────────────────────────────────────────────────────────────────
def draw_touch_map(data, title, subtitle, brand):
    tm     = data.get("touch_map", {})
    succ   = tm.get("successful", [])
    unsucc = tm.get("unsuccessful", [])

    fig = _canvas()
    ax  = _pitch_ax(fig)
    _draw_pitch(ax)

    if succ:
        ax.scatter([_cx(p["x"]) for p in succ],
                   [_cy(p["y"]) for p in succ],
                   s=130, color=T["dot_ok"], alpha=0.88, zorder=4, edgecolors="none")
    if unsucc:
        ax.scatter([_cx(p["x"]) for p in unsucc],
                   [_cy(p["y"]) for p in unsucc],
                   s=100, color=T["dot_bad"], alpha=0.70, zorder=3, edgecolors="none")

    handles = []
    if succ:
        handles.append(mpatches.Patch(color=T["dot_ok"],
                        label=f"{len(succ)} Successful Touches"))
    if unsucc:
        handles.append(mpatches.Patch(color=T["dot_bad"],
                        label=f"{len(unsucc)} Unsuccessful Touches"))
    if handles:
        leg = fig.legend(handles=handles, loc="lower center",
                         bbox_to_anchor=(0.5, 0.022), ncol=2, frameon=False,
                         fontsize=14, labelcolor=T["text"],
                         handlelength=1.4, columnspacing=2.0)
        plt.setp(leg.get_texts(), fontfamily="Montserrat", fontweight="700")

    _labels(fig, title, subtitle, brand)
    return _to_png(fig)

# ─────────────────────────────────────────────────────────────────────────────
# DRAW: SHOT MAP  (Athletic-style)
# ─────────────────────────────────────────────────────────────────────────────
def draw_shot_map(data, title, subtitle, brand,
                  manual_goals=None, manual_xg=None, manual_shots=None,
                  manual_right=None, manual_left=None, manual_head=None):
    sm    = data.get("shot_map", {})
    shots = sm.get("shots", [])

    goals_val = manual_goals if manual_goals is not None else sm.get("goals", sum(1 for s in shots if s.get("outcome")=="goal"))
    xg_val    = manual_xg   if manual_xg    is not None else sm.get("total_xg", sum(s.get("xg",0) for s in shots))
    shots_val = manual_shots if manual_shots is not None else sm.get("total_shots", len(shots))
    right_val = manual_right
    left_val  = manual_left
    head_val  = manual_head

    fig = _canvas()

    # Half pitch — left 58% of canvas
    ax = fig.add_axes([0.03, 0.10, 0.58, 0.76])
    ax.set_facecolor(T["pitch"])
    ax.set_xlim(50, 107); ax.set_ylim(-2, 70)
    ax.set_aspect("equal"); ax.axis("off")
    _draw_half_pitch(ax)

    for s in shots:
        x   = _cx(s.get("x", 75))
        y   = _cy(s.get("y", 34))
        xg  = float(s.get("xg", 0.05))
        out = str(s.get("outcome", "saved")).lower()
        sz  = max(60, min(1200, 60 + xg * 2000))

        if out == "goal":
            ax.scatter(x, y, s=sz, color=T["dot_ok"], zorder=6,
                       edgecolors=T["text"], linewidths=1.5)
        else:
            ax.scatter(x, y, s=sz, facecolors="none", zorder=5,
                       edgecolors=T["dot_bad"], linewidths=1.5)

    # xG scale legend
    fig.text(0.04, 0.055, "Low xG", fontsize=11, color=T["subtext"],
             fontfamily="Montserrat", fontweight="600", va="bottom")
    for i, sz in enumerate([25, 70, 140, 280, 480]):
        r = 0.006 * (sz**0.42) / 3.2
        cx_ = 0.13 + i * 0.028
        fig.add_artist(plt.Circle((cx_, 0.062), r, transform=fig.transFigure,
                                  facecolor="none", edgecolor=T["dot_bad"], lw=1.2, zorder=5))
    fig.text(0.30, 0.055, "High xG", fontsize=11, color=T["subtext"],
             fontfamily="Montserrat", fontweight="600", va="bottom")

    fig.add_artist(plt.Circle((0.38, 0.062), 0.012, transform=fig.transFigure,
                               facecolor=T["dot_ok"], edgecolor=T["text"], lw=1.2, zorder=5))
    fig.text(0.402, 0.055, "Goal", fontsize=11, color=T["text"],
             fontfamily="Montserrat", fontweight="700", va="bottom")
    fig.add_artist(plt.Circle((0.48, 0.062), 0.012, transform=fig.transFigure,
                               facecolor="none", edgecolor=T["dot_bad"], lw=1.2, zorder=5))
    fig.text(0.502, 0.055, "No Goal", fontsize=11, color=T["text"],
             fontfamily="Montserrat", fontweight="700", va="bottom")

    # Stats panel — right side
    sx = 0.66
    def _stat(yp, label, value, large=False):
        fig.text(sx, yp, label, fontsize=13, fontweight="700",
                 color=T["text"], fontfamily="Montserrat", va="top")
        fig.text(sx, yp - 0.058, str(value), fontsize=28 if large else 22,
                 fontweight="900", color=T["accent"], fontfamily="Montserrat", va="top")

    fig.text(sx, 0.88, "SEASON STATS", fontsize=10, fontweight="900",
             color=T["subtext"], fontfamily="Montserrat", va="top")

    _stat(0.82, "Goals",       goals_val,  large=True)
    _stat(0.70, "xG",          f"{xg_val:.2f}", large=True)
    _stat(0.58, "Total Shots", shots_val)
    xgps = xg_val / max(shots_val, 1) if shots_val else 0
    _stat(0.50, "xG per Shot", f"{xgps:.2f}")
    if right_val is not None: _stat(0.42, "Right Foot", right_val)
    if left_val  is not None: _stat(0.34, "Left Foot",  left_val)
    if head_val  is not None: _stat(0.26, "Head",       head_val)

    _labels(fig, title, subtitle, brand, ty=0.96, sy=0.91)
    return _to_png(fig)

# ─────────────────────────────────────────────────────────────────────────────
# DRAW: PASS NETWORK
# ─────────────────────────────────────────────────────────────────────────────
def draw_pass_network(data, title, subtitle, brand):
    pn    = data.get("pass_network", {})
    nodes = pn.get("nodes", [])
    edges = pn.get("edges", [])
    if not nodes:
        raise ValueError("No pass network nodes found.")

    fig = _canvas()
    ax  = _pitch_ax(fig)
    _draw_pitch(ax)

    node_map  = {n["id"]: n for n in nodes}
    max_count = max((e.get("count", 1) for e in edges), default=1)

    # Draw edges first (behind nodes)
    for e in edges:
        f, t = e["from"], e["to"]
        if f not in node_map or t not in node_map: continue
        nf, nt = node_map[f], node_map[t]
        cnt   = e.get("count", 1)
        alpha = 0.25 + 0.70 * (cnt / max_count)
        lw    = 1.5  + 8.5  * (cnt / max_count)
        ax.plot([_cx(nf["x"]), _cx(nt["x"])],
                [_cy(nf["y"]), _cy(nt["y"])],
                color=T["accent"], alpha=alpha, lw=lw, zorder=2,
                solid_capstyle="round")

    # Node size by involvement
    inv = {n["id"]: sum(e.get("count",1) for e in edges
                        if e["from"]==n["id"] or e["to"]==n["id"])
           for n in nodes}
    max_inv = max(inv.values(), default=1)

    # Draw nodes and labels
    txt_col = "#000" if T["accent"] in ("#f59e0b","#fbbf24","#22c55e") else "#fff"
    for n in nodes:
        x, y = _cx(n["x"]), _cy(n["y"])
        sz   = 700 + 1400 * (inv.get(n["id"], 0) / max_inv)
        # Node circle
        ax.scatter(x, y, s=sz, color=T["accent"], zorder=5,
                   edgecolors=T["text"], linewidths=2.0)
        # Shirt number inside
        ax.text(x, y, str(n["id"]), ha="center", va="center",
                fontsize=11, fontweight="900", color=txt_col,
                zorder=6, fontfamily="Montserrat")
        # Name below — show if non-empty
        name = (n.get("name") or "").strip()
        if name:
            # Use last word (surname) only, capitalised
            surname = name.split()[-1].capitalize()
            ax.text(x, y - 4.2, surname,
                    ha="center", va="top", fontsize=9, fontweight="700",
                    color=T["text"], zorder=6, fontfamily="Montserrat",
                    path_effects=[mpe.withStroke(linewidth=3, foreground=T["bg"])])

    _labels(fig, title, subtitle, brand)
    return _to_png(fig)

# ─────────────────────────────────────────────────────────────────────────────
# DRAW: AVERAGE POSITIONS
# ─────────────────────────────────────────────────────────────────────────────
def draw_avg_positions(data, title, subtitle, brand):
    ap      = data.get("avg_positions", {})
    players = ap.get("players", [])
    formation = ap.get("formation", "")
    if not players:
        raise ValueError("No player positions found.")

    fig = _canvas()
    ax  = _pitch_ax(fig)
    _draw_pitch(ax)

    GK_COLOR = "#a78bfa"  # purple for GK only

    for p in players:
        raw_x = float(p.get("x", 50))
        raw_y = float(p.get("y", 50))
        x = float(np.clip(raw_x * 105 / 100, 3, 102))
        y = float(np.clip(raw_y * 68  / 100, 3, 65))
        pos = str(p.get("position", "CM")).upper()
        col = GK_COLOR if pos == "GK" else T["accent"]
        txt_col = "#000" if col in ("#a78bfa", "#fbbf24", "#22c55e", "#4ade80") else "#fff"
        ax.scatter(x, y, s=1100, color=col, zorder=5,
                   edgecolors=T["text"], linewidths=2.2)
        ax.text(x, y, str(p.get("id", "")), ha="center", va="center",
                fontsize=12, fontweight="900", color=txt_col,
                zorder=6, fontfamily="Montserrat")
        name = p.get("name", "")
        if name:
            short = name.split()[-1] if " " in name else name
            ax.text(x, y-4.5, short, ha="center", va="top",
                    fontsize=9, fontweight="600", color=T["text"], zorder=6,
                    fontfamily="Montserrat",
                    path_effects=[mpe.withStroke(linewidth=2.5, foreground=T["bg"])])

    if formation:
        fig.text(0.5, 0.032, f"Formation: {formation}", ha="center",
                 fontsize=13, fontweight="700", color=T["subtext"],
                 fontfamily="Montserrat")

    _labels(fig, title, subtitle, brand)
    return _to_png(fig)

# ─────────────────────────────────────────────────────────────────────────────
# DRAW: HEAT MAP
# ─────────────────────────────────────────────────────────────────────────────
def draw_heat_map(data, title, subtitle, brand):
    tm     = data.get("touch_map", {})
    succ   = tm.get("successful", [])
    unsucc = tm.get("unsuccessful", [])
    all_pts = succ + unsucc
    if not all_pts:
        raise ValueError("No touch data for heat map.")

    fig = _canvas()
    ax  = _pitch_ax(fig)

    grid = np.zeros((68*4, 105*4))
    for p in all_pts:
        gx = int(np.clip(_cx(p["x"])*4, 0, 105*4-1))
        gy = int(np.clip(_cy(p["y"])*4, 0, 68*4-1))
        grid[gy, gx] += 1
    grid = gaussian_filter(grid, sigma=12)

    if theme_name == "Light (Athletic)":
        cmap_colors = ["#f8f5f0","#dbeafe","#93c5fd","#3b82f6","#1d4ed8","#7c3aed","#ef4444"]
    else:
        cmap_colors = ["#0a0f1c","#1e3a5f","#1d4ed8","#7c3aed","#ef4444","#f97316","#fbbf24"]
    cmap = LinearSegmentedColormap.from_list("heat", cmap_colors)

    ax.imshow(grid, extent=[0,105,0,68], origin="lower",
              cmap=cmap, alpha=0.88, aspect="auto", zorder=2)
    _draw_pitch(ax, lw=1.8)

    _labels(fig, title, subtitle, brand)
    return _to_png(fig)

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _empty_box(ph):
    ph.markdown("""<div style="background:#0d1220;border:1px dashed #1e2d4a;
        border-radius:10px;padding:80px;text-align:center;color:#374151;
        font-size:13px;font-weight:700;">Upload a screenshot to get started</div>""",
        unsafe_allow_html=True)

def _prep_image(uploaded):
    img = Image.open(uploaded).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _auto_labels(data):
    t = title_in or data.get("team", "")
    s = subtitle_in or " · ".join(filter(None, [
        data.get("opponent",""), data.get("competition",""), data.get("date","")]))
    return t, s

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 18px;border-bottom:1px solid #1a2540;margin-bottom:24px;">
  <div style="font-size:10px;font-weight:900;letter-spacing:.3em;color:#ef4444;margin-bottom:6px;">FOOTBALL ANALYTICS</div>
  <h1 style="font-size:36px;font-weight:900;color:#fff;margin:0;">VIZ REDRAW</h1>
  <div style="color:#4b5563;font-size:13px;margin-top:6px;">
    Screenshot → AI extracts data → redrawn at 1920×1080 · inspired by The Athletic
  </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Touch Map", "🎯 Shot Map", "🔗 Pass Network",
    "📌 Avg Positions", "🌡️ Heat Map"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOUCH MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([1, 1.8], gap="large")
    with c1:
        up1 = st.file_uploader("Touch map screenshot", type=["png","jpg","jpeg","webp"],
                                key="up1", label_visibility="collapsed")
        if up1:
            st.image(Image.open(up1), use_column_width=True)
            go1 = st.button("EXTRACT & REDRAW", key="go1", use_container_width=True)
        else:
            go1 = False
    with c2:
        ph1 = st.empty(); _empty_box(ph1)
        if go1 and up1:
            if not api_key: st.error("Enter API key in sidebar.")
            else:
                with st.spinner("Reading viz…"):
                    try: data = call_claude(_prep_image(up1), "touch map", api_key)
                    except Exception as e: st.error(f"Error: {e}"); st.stop()
                with st.spinner("Drawing…"):
                    t, s = _auto_labels(data)
                    png = draw_touch_map(data, t, s, brand_in)
                ph1.image(png, use_column_width=True)
                st.download_button("⬇️ Download 1920×1080 PNG", png,
                    f"{(t or 'touchmap').replace(' ','_')}.png", "image/png",
                    use_container_width=True, key="dl1")
                with st.expander("Extracted JSON"): st.json(data)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SHOT MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns([1, 1.8], gap="large")
    with c1:
        up2 = st.file_uploader("Shot map screenshot", type=["png","jpg","jpeg","webp"],
                                key="up2", label_visibility="collapsed")
        if up2:
            st.image(Image.open(up2), use_column_width=True)

        st.markdown("---")
        st.markdown("**Manual Stats Panel** *(tick to override extracted)*")
        use_manual = st.checkbox("Use manual stats", key="use_manual")
        m_goals = st.number_input("Goals",       0, 100, 0, key="m_goals")
        m_xg    = st.number_input("xG",          0.0, 100.0, 0.0, step=0.01, format="%.2f", key="m_xg")
        m_shots = st.number_input("Total Shots", 0, 200, 0, key="m_shots")
        m_right = st.number_input("Right Foot",  0, 200, 0, key="m_right")
        m_left  = st.number_input("Left Foot",   0, 200, 0, key="m_left")
        m_head  = st.number_input("Head",        0, 200, 0, key="m_head")

        if up2:
            go2 = st.button("EXTRACT & REDRAW", key="go2", use_container_width=True)
        else:
            go2 = False
    with c2:
        ph2 = st.empty(); _empty_box(ph2)
        if go2 and up2:
            if not api_key: st.error("Enter API key in sidebar.")
            else:
                with st.spinner("Reading viz…"):
                    try: data = call_claude(_prep_image(up2), "shot map / xG map", api_key)
                    except Exception as e: st.error(f"Error: {e}"); st.stop()
                with st.spinner("Drawing…"):
                    t, s = _auto_labels(data)
                    png = draw_shot_map(data, t, s, brand_in,
                        manual_goals = m_goals if use_manual else None,
                        manual_xg    = m_xg    if use_manual else None,
                        manual_shots = m_shots  if use_manual else None,
                        manual_right = m_right  if use_manual else None,
                        manual_left  = m_left   if use_manual else None,
                        manual_head  = m_head   if use_manual else None)
                ph2.image(png, use_column_width=True)
                st.download_button("⬇️ Download 1920×1080 PNG", png,
                    f"{(t or 'shotmap').replace(' ','_')}.png", "image/png",
                    use_container_width=True, key="dl2")
                with st.expander("Extracted JSON"): st.json(data)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PASS NETWORK
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PASS NETWORK
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    # Phase 1: upload + extract nodes
    c1, c2 = st.columns([1, 1.8], gap="large")
    with c1:
        up3 = st.file_uploader("Pass network screenshot", type=["png","jpg","jpeg","webp"],
                                key="up3", label_visibility="collapsed")
        if up3:
            st.image(Image.open(up3), use_column_width=True)
            if st.button("EXTRACT PLAYERS", key="go3", use_container_width=True):
                if not api_key:
                    st.error("Enter API key in sidebar.")
                else:
                    for k in ["pn_data","pn_png","pn_title"]:
                        st.session_state.pop(k, None)
                    with st.spinner("Reading players and positions..."):
                        try:
                            st.session_state["pn_data"] = call_claude(
                                _prep_image(up3), "pass network", api_key)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    with c2:
        ph3 = st.empty()
        if "pn_png" in st.session_state:
            ph3.image(st.session_state["pn_png"], use_column_width=True)
            st.download_button("⬇️ Download 1920×1080 PNG",
                st.session_state["pn_png"],
                f"{(st.session_state.get('pn_title') or 'passnet').replace(' ','_')}.png",
                "image/png", use_container_width=True, key="dl3")
        else:
            _empty_box(ph3)

    # Phase 2: once nodes extracted, show connection editor
    if "pn_data" in st.session_state:
        data  = st.session_state["pn_data"]
        pn    = data.get("pass_network", {})
        nodes = pn.get("nodes", [])
        claude_edges = pn.get("edges", [])
        nmap  = {n["id"]: (n.get("name") or "").strip() for n in nodes}
        node_ids = sorted([n["id"] for n in nodes])

        # Build set of pairs Claude detected
        claude_pairs = set()
        claude_counts = {}
        for e in claude_edges:
            pair = (min(e["from"], e["to"]), max(e["from"], e["to"]))
            claude_pairs.add(pair)
            claude_counts[pair] = e.get("count", 9)

        st.markdown("---")
        st.markdown("""<div style='font-size:10px;font-weight:900;letter-spacing:.15em;
            color:#ef4444;margin-bottom:6px;'>STEP 2 — CHECK CONNECTIONS</div>
            <div style='color:#6b7280;font-size:12px;margin-bottom:10px;'>
            Claude has pre-ticked what it found. Untick wrong ones, tick missing ones.
            </div>""", unsafe_allow_html=True)

        selected_edges = []
        pairs = [(a, b) for i, a in enumerate(node_ids) for b in node_ids[i+1:]]
        cols = st.columns(3)
        for idx, (a, b) in enumerate(pairs):
            pair = (min(a,b), max(a,b))
            an = nmap.get(a, ""); bn = nmap.get(b, "")
            label = f"#{a}{' '+an if an else ''} ↔ #{b}{' '+bn if bn else ''}"
            col = cols[idx % 3]
            default = pair in claude_pairs
            checked = col.checkbox(label, value=default, key=f"edge_{a}_{b}")
            if checked:
                default_thick = claude_counts.get(pair, 9)
                thickness = col.select_slider("", [4, 9, 15],
                    value=default_thick, key=f"thick_{a}_{b}",
                    label_visibility="collapsed")
                selected_edges.append({"from": a, "to": b, "count": thickness})

        st.markdown("---")
        if st.button("DRAW PASS NETWORK", key="draw_pn", use_container_width=True):
            data["pass_network"]["edges"] = selected_edges
            with st.spinner("Drawing..."):
                t, s = _auto_labels(data)
                try:
                    png = draw_pass_network(data, t, s, brand_in)
                    st.session_state["pn_png"]   = png
                    st.session_state["pn_title"] = t
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AVERAGE POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.caption("Works with lineup screenshots from Sofascore, Livescore, WhoScored, Opta etc.")
    c1, c2 = st.columns([1, 1.8], gap="large")
    with c1:
        up4 = st.file_uploader("Lineup / formation screenshot", type=["png","jpg","jpeg","webp"],
                                key="up4", label_visibility="collapsed")
        if up4:
            st.image(Image.open(up4), use_column_width=True)
            go4 = st.button("EXTRACT & REDRAW", key="go4", use_container_width=True)
        else:
            go4 = False
    with c2:
        ph4 = st.empty(); _empty_box(ph4)
        if go4 and up4:
            if not api_key: st.error("Enter API key in sidebar.")
            else:
                with st.spinner("Reading viz…"):
                    try: data = call_claude(_prep_image(up4),
                                "average positions / lineup / formation", api_key)
                    except Exception as e: st.error(f"Error: {e}"); st.stop()
                with st.spinner("Drawing…"):
                    t, s = _auto_labels(data)
                    try: png = draw_avg_positions(data, t, s, brand_in)
                    except ValueError as e: st.error(str(e)); st.stop()
                ph4.image(png, use_column_width=True)
                st.download_button("⬇️ Download 1920×1080 PNG", png,
                    f"{(t or 'avgpos').replace(' ','_')}.png", "image/png",
                    use_container_width=True, key="dl4")
                with st.expander("Extracted JSON"): st.json(data)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HEAT MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.caption("Upload a touch map screenshot — dots are converted into a gaussian heat map.")
    c1, c2 = st.columns([1, 1.8], gap="large")
    with c1:
        up5 = st.file_uploader("Touch map screenshot (for heat)", type=["png","jpg","jpeg","webp"],
                                key="up5", label_visibility="collapsed")
        if up5:
            st.image(Image.open(up5), use_column_width=True)
            go5 = st.button("EXTRACT & REDRAW", key="go5", use_container_width=True)
        else:
            go5 = False
    with c2:
        ph5 = st.empty(); _empty_box(ph5)
        if go5 and up5:
            if not api_key: st.error("Enter API key in sidebar.")
            else:
                with st.spinner("Reading viz…"):
                    try: data = call_claude(_prep_image(up5), "touch map", api_key)
                    except Exception as e: st.error(f"Error: {e}"); st.stop()
                with st.spinner("Drawing heat map…"):
                    t, s = _auto_labels(data)
                    try: png = draw_heat_map(data, t, s, brand_in)
                    except ValueError as e: st.error(str(e)); st.stop()
                ph5.image(png, use_column_width=True)
                st.download_button("⬇️ Download 1920×1080 PNG", png,
                    f"{(t or 'heatmap').replace(' ','_')}.png", "image/png",
                    use_container_width=True, key="dl5")
                with st.expander("Extracted JSON"): st.json(data)

st.markdown("---")
st.caption("VIZ REDRAW · 1920×1080 output · Inspired by The Athletic")
