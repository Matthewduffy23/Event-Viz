"""
viz_redraw_app.py
─────────────────────────────────────────────────────────────────────────────
Upload a football viz screenshot (pass network, touch map, shot map, etc.)
→ Claude vision extracts the data
→ App redraws it in a clean dark professional style

pip install streamlit matplotlib mplsoccer numpy pillow requests anthropic
streamlit run viz_redraw_app.py
─────────────────────────────────────────────────────────────────────────────
"""

import io
import json
import base64
import re
import textwrap

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

try:
    from mplsoccer import Pitch, VerticalPitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False

import requests
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VIZ REDRAW", layout="wide", page_icon="⚽")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif!important;}
.stApp{background:#07090f!important;}
section[data-testid="stSidebar"]{background:#040608!important;border-right:1px solid #0f1520!important;}
section[data-testid="stSidebar"] *{color:#fff!important;}
.stButton>button{
  background:linear-gradient(135deg,#ef4444,#dc2626)!important;
  color:#fff!important;font-weight:800!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;border-radius:4px!important;
  letter-spacing:.08em!important;text-transform:uppercase!important;
  padding:10px 24px!important;
}
.stButton>button:hover{background:linear-gradient(135deg,#dc2626,#b91c1c)!important;}
.stFileUploader{background:#0d1220!important;border:1px solid #1e2d4a!important;border-radius:8px!important;}
div[data-baseweb="select"]*{background:#0d1424!important;color:#fff!important;}
div[data-baseweb="popover"]*{background:#0d1424!important;color:#fff!important;}
.stSelectbox>div>div{background:#0d1424!important;border:1px solid #1e2d4a!important;}
.stTextInput>div>div>input{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important;}
label{color:#6b7280!important;font-size:9px!important;letter-spacing:.14em!important;text-transform:uppercase!important;}
h1,h2,h3{color:#fff!important;}
footer{display:none!important;}
.upload-hint{color:#4b5563;font-size:12px;text-align:center;margin-top:8px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 24px;border-bottom:1px solid #1a2540;margin-bottom:32px;">
  <div style="font-size:11px;font-weight:900;letter-spacing:.3em;color:#ef4444;text-transform:uppercase;margin-bottom:8px;">
    FOOTBALL ANALYTICS
  </div>
  <h1 style="font-size:42px;font-weight:900;color:#fff;margin:0;letter-spacing:-.01em;line-height:1;">
    VIZ REDRAW
  </h1>
  <div style="color:#4b5563;font-size:13px;font-weight:500;margin-top:8px;letter-spacing:.04em;">
    Upload any football viz screenshot → AI extracts the data → redraws it professionally
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SETTINGS")
    st.markdown("---")

    api_key = st.text_input("Anthropic API Key", type="password",
                             help="Required to call Claude vision for data extraction",
                             key="api_key")

    st.markdown("---")
    st.markdown("**VIZ TYPE**")
    viz_type = st.selectbox("Detect automatically or force type",
                             ["Auto-detect", "Pass Network", "Touch Map",
                              "Shot Map", "Heat Map", "Chance Creation Map"],
                             key="viz_type")

    st.markdown("---")
    st.markdown("**LABELS**")
    title_text  = st.text_input("Title", placeholder="e.g. Coventry City — Pass Network", key="title")
    subtitle_text = st.text_input("Subtitle", placeholder="e.g. vs Preston North End · 18 Mar 2026", key="subtitle")
    brand_text  = st.text_input("Brand / Credit", value="@YourHandle", key="brand")

    st.markdown("---")
    st.markdown("**STYLE**")
    pitch_color  = st.selectbox("Pitch colour", ["Dark Navy (#0a0f1c)", "Pure Black", "Dark Green", "Charcoal"], key="pitch_col")
    accent_color = st.selectbox("Accent colour", ["Red (#ef4444)", "Blue (#3b82f6)", "Gold (#f59e0b)", "Teal (#14b8a6)"], key="accent_col")

PITCH_COLORS = {
    "Dark Navy (#0a0f1c)": "#0a0f1c",
    "Pure Black":           "#000000",
    "Dark Green":           "#0d1f0d",
    "Charcoal":             "#1a1a2e",
}
ACCENT_COLORS = {
    "Red (#ef4444)":   "#ef4444",
    "Blue (#3b82f6)":  "#3b82f6",
    "Gold (#f59e0b)":  "#f59e0b",
    "Teal (#14b8a6)":  "#14b8a6",
}
BG    = PITCH_COLORS.get(pitch_color,  "#0a0f1c")
ACCENT = ACCENT_COLORS.get(accent_color, "#ef4444")
LINE_C = "#2a3a5a"
TEXT_C = "#e2e8f0"

# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE VISION CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude_vision(image_bytes: bytes, viz_hint: str, key: str) -> dict:
    """Send image to Claude, get back structured JSON of viz data."""

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    system_prompt = textwrap.dedent("""
    You are a football analytics data extraction engine.
    Given a screenshot of a football visualisation, extract ALL visible data and return ONLY valid JSON.
    Do not include any explanation, markdown, or code fences — pure JSON only.
    """)

    user_prompt = textwrap.dedent(f"""
    Analyse this football visualisation screenshot carefully.

    Viz type hint: {viz_hint}

    Extract all data and return a JSON object with this structure (use whichever sections apply):

    {{
      "viz_type": "pass_network" | "touch_map" | "shot_map" | "heat_map" | "chance_creation",
      "team_name": "...",
      "opponent": "...",
      "competition": "...",
      "date": "...",

      "pass_network": {{
        "nodes": [
          {{"id": <shirt_number_int>, "x": <0-100_float>, "y": <0-100_float>, "name": "..."}}
        ],
        "edges": [
          {{"from": <id>, "to": <id>, "count": <int>}}
        ],
        "subs_bench": [<shirt_number_int>, ...]
      }},

      "touch_map": {{
        "successful": [{{"x": <0-100>, "y": <0-100>}}],
        "unsuccessful": [{{"x": <0-100>, "y": <0-100>}}]
      }},

      "shot_map": {{
        "shots": [
          {{"x": <0-100>, "y": <0-100>, "xg": <float>, "outcome": "goal"|"saved"|"blocked"|"off_target", "body_part": "foot"|"head"|"other"}}
        ]
      }},

      "chance_creation": {{
        "chances": [
          {{"x_start": <0-100>, "y_start": <0-100>, "x_end": <0-100>, "y_end": <0-100>, "outcome": "goal"|"shot"|"blocked"}}
        ]
      }}
    }}

    IMPORTANT coordinate system:
    - x = 0 is left side of pitch, x = 100 is right side
    - y = 0 is bottom (goal line), y = 100 is top (goal line)
    - For pass networks, estimate positions from visual layout — be as precise as possible
    - For touch/shot maps, read dot positions relative to pitch dimensions
    - Shirt numbers are integers shown inside or next to circles on pass networks
    - Edge thickness/weight = approximate number of passes (thin=1-3, medium=4-8, thick=9+)
    - Nodes shown below the pitch (subs bench) should go in subs_bench array, NOT nodes array

    Return ONLY the JSON object.
    """)

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-opus-4-6",
        "max_tokens": 4000,
        "system": system_prompt,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text",  "text": user_prompt}
            ]
        }]
    }

    resp = requests.post("https://api.anthropic.com/v1/messages",
                         headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING FUNCTIONS  —  all output 1920×1080 px (Canva slide ready)
# ─────────────────────────────────────────────────────────────────────────────

# Canvas constants
W_IN, H_IN = 19.2, 10.8   # inches at 100 dpi = 1920×1080
DPI = 100

# Layout zones (as fractions of figure)
TITLE_TOP    = 0.93   # title baseline
SUBTITLE_TOP = 0.88   # subtitle baseline
PITCH_TOP    = 0.85   # pitch axes top
PITCH_BOT    = 0.10   # pitch axes bottom  (leaves room for legend)
PITCH_LEFT   = 0.06
PITCH_RIGHT  = 0.94
BRAND_X      = 0.97
BRAND_Y      = 0.02


def _make_canvas():
    """Create a 1920×1080 figure."""
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def _make_pitch_ax(fig):
    """
    Add a pitch axes that fills the middle of the canvas,
    preserving a real 105×68 m aspect ratio inside a landscape frame.
    Pitch coordinate space: x=0-105, y=0-68.
    """
    ax = fig.add_axes([PITCH_LEFT, PITCH_BOT,
                       PITCH_RIGHT - PITCH_LEFT,
                       PITCH_TOP   - PITCH_BOT])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _draw_pitch_lines(ax, lw=1.8):
    """Draw standard pitch markings in 105×68 coordinate space."""
    c = LINE_C

    def rect(x, y, w, h):
        ax.add_patch(mpatches.Rectangle((x, y), w, h,
                     fill=False, edgecolor=c, linewidth=lw, zorder=1))

    rect(0, 0, 105, 68)                  # outline
    ax.plot([52.5, 52.5], [0, 68], color=c, lw=lw, zorder=1)   # halfway
    ax.add_patch(plt.Circle((52.5, 34), 9.15,
                 fill=False, edgecolor=c, lw=lw, zorder=1))
    ax.scatter([52.5], [34], s=12, color=c, zorder=1)
    # Penalty areas
    rect(0,  13.84, 16.5, 40.32)
    rect(88.5, 13.84, 16.5, 40.32)
    # 6-yard boxes
    rect(0,  24.84, 5.5, 18.32)
    rect(99.5, 24.84, 5.5, 18.32)
    # Goals
    rect(-2, 29.84, 2, 8.32)
    rect(105, 29.84, 2, 8.32)
    # Centre spot
    ax.scatter([52.5], [34], s=15, color=c, zorder=1)
    # Penalty spots
    ax.scatter([11, 94], [34, 34], s=15, color=c, zorder=1)


def _add_labels(fig, title, subtitle, brand):
    """Add title, subtitle and brand to a 1920×1080 figure."""
    if title:
        fig.text(0.5, TITLE_TOP, title.upper(),
                 ha="center", va="bottom", fontsize=28, fontweight="900",
                 color=TEXT_C, fontfamily="Montserrat",
                 path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    if subtitle:
        fig.text(0.5, SUBTITLE_TOP - 0.01, subtitle,
                 ha="center", va="bottom", fontsize=16, fontweight="500",
                 color="#6b7280", fontfamily="Montserrat")
    if brand:
        fig.text(BRAND_X, BRAND_Y, brand,
                 ha="right", va="bottom", fontsize=13, fontweight="700",
                 color="#374151", fontfamily="Montserrat")


def _add_legend(fig, handles, y=0.065):
    """Add a centred legend below the pitch."""
    leg = fig.legend(handles=handles,
                     loc="lower center",
                     bbox_to_anchor=(0.5, y),
                     ncol=len(handles),
                     frameon=False,
                     fontsize=14,
                     labelcolor=TEXT_C,
                     handlelength=1.2,
                     handleheight=1.0,
                     borderpad=0,
                     columnspacing=2.0)
    plt.setp(leg.get_texts(), fontfamily="Montserrat", fontweight="700")


# ── Pass Network ──────────────────────────────────────────────────────────────

def draw_pass_network(data: dict, title: str, subtitle: str, brand: str) -> plt.Figure:
    pn    = data.get("pass_network", {})
    nodes = pn.get("nodes", [])
    edges = pn.get("edges", [])

    if not nodes:
        raise ValueError("No pass network nodes found in extracted data.")

    fig = _make_canvas()
    ax  = _make_pitch_ax(fig)
    _draw_pitch_lines(ax)

    # Convert incoming 0-100 coords → 105×68
    def cx(v): return float(v) * 105 / 100
    def cy(v): return float(v) * 68  / 100

    node_map  = {n["id"]: n for n in nodes}
    max_count = max((e.get("count", 1) for e in edges), default=1)

    # Edges
    for e in edges:
        f, t = e["from"], e["to"]
        if f not in node_map or t not in node_map: continue
        nf, nt = node_map[f], node_map[t]
        cnt   = e.get("count", 1)
        alpha = 0.25 + 0.70 * (cnt / max_count)
        lw    = 1.5  + 8.0  * (cnt / max_count)
        ax.plot([cx(nf["x"]), cx(nt["x"])],
                [cy(nf["y"]), cy(nt["y"])],
                color=ACCENT, alpha=alpha, linewidth=lw, zorder=2,
                solid_capstyle="round")

    # Node sizes scaled by involvement
    involvement = {n["id"]: sum(e.get("count", 1) for e in edges
                                if e["from"] == n["id"] or e["to"] == n["id"])
                   for n in nodes}
    max_inv = max(involvement.values(), default=1)

    for n in nodes:
        inv = involvement.get(n["id"], 1)
        sz  = 600 + 1400 * (inv / max_inv)
        x, y = cx(n["x"]), cy(n["y"])
        ax.scatter(x, y, s=sz, color=ACCENT, zorder=5,
                   edgecolors="#ffffff", linewidths=2.5)
        ax.text(x, y, str(n["id"]),
                ha="center", va="center", fontsize=11, fontweight="900",
                color="#000000" if ACCENT in ("#f59e0b", "#fbbf24") else "#ffffff",
                zorder=6, fontfamily="Montserrat")
        name = n.get("name", "")
        if name:
            ax.text(x, y - 3.5, name,
                    ha="center", va="top", fontsize=8, fontweight="600",
                    color=TEXT_C, zorder=6, fontfamily="Montserrat",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    _add_labels(fig, title, subtitle, brand)
    return fig


# ── Touch Map ─────────────────────────────────────────────────────────────────

def draw_touch_map(data: dict, title: str, subtitle: str, brand: str) -> plt.Figure:
    tm     = data.get("touch_map", {})
    succ   = tm.get("successful",   [])
    unsucc = tm.get("unsuccessful", [])

    fig = _make_canvas()
    ax  = _make_pitch_ax(fig)
    _draw_pitch_lines(ax)

    def cx(v): return float(v) * 105 / 100
    def cy(v): return float(v) * 68  / 100

    if succ:
        ax.scatter([cx(p["x"]) for p in succ],
                   [cy(p["y"]) for p in succ],
                   s=120, color=ACCENT, alpha=0.85, zorder=4,
                   edgecolors="none")
    if unsucc:
        ax.scatter([cx(p["x"]) for p in unsucc],
                   [cy(p["y"]) for p in unsucc],
                   s=100, color="#9ca3af", alpha=0.60, zorder=3,
                   edgecolors="none")

    handles = []
    if succ:
        handles.append(mpatches.Patch(color=ACCENT,   label=f"{len(succ)} Successful Touches"))
    if unsucc:
        handles.append(mpatches.Patch(color="#9ca3af", label=f"{len(unsucc)} Unsuccessful Touches"))
    if handles:
        _add_legend(fig, handles)

    _add_labels(fig, title, subtitle, brand)
    return fig


# ── Shot Map ──────────────────────────────────────────────────────────────────

def draw_shot_map(data: dict, title: str, subtitle: str, brand: str) -> plt.Figure:
    sm    = data.get("shot_map", {})
    shots = sm.get("shots", [])

    fig = _make_canvas()
    # Use only attacking half — x: 52.5 to 107
    ax = fig.add_axes([0.15, PITCH_BOT, 0.70, PITCH_TOP - PITCH_BOT])
    ax.set_facecolor(BG)
    ax.set_xlim(52.5, 107)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")

    c = LINE_C
    ax.add_patch(mpatches.Rectangle((52.5, 0), 52.5, 68,
                 fill=False, edgecolor=c, lw=1.8, zorder=1))
    ax.plot([52.5, 52.5], [0, 68], color=c, lw=1.8, zorder=1)
    ax.add_patch(plt.Circle((52.5, 34), 9.15,
                 fill=False, edgecolor=c, lw=1.5, zorder=1))
    ax.add_patch(mpatches.Rectangle((88.5, 13.84), 16.5, 40.32,
                 fill=False, edgecolor=c, lw=1.5, zorder=1))
    ax.add_patch(mpatches.Rectangle((99.5, 24.84), 5.5, 18.32,
                 fill=False, edgecolor=c, lw=1.5, zorder=1))
    ax.add_patch(mpatches.Rectangle((105, 29.84), 2, 8.32,
                 fill=False, edgecolor=c, lw=1.5, zorder=1))

    OUTCOME_STYLE = {
        "goal":       {"color": ACCENT,    "marker": "*", "base": 600, "alpha": 1.0},
        "saved":      {"color": "#3b82f6", "marker": "o", "base": 200, "alpha": 0.9},
        "blocked":    {"color": "#9ca3af", "marker": "s", "base": 160, "alpha": 0.8},
        "off_target": {"color": "#4b5563", "marker": "X", "base": 160, "alpha": 0.7},
    }

    for shot in shots:
        outcome = shot.get("outcome", "saved").lower()
        style   = OUTCOME_STYLE.get(outcome, OUTCOME_STYLE["saved"])
        xg      = shot.get("xg", 0.05)
        size    = style["base"] * (0.5 + 2.5 * xg)
        x = float(shot["x"]) * 105 / 100
        y = float(shot["y"]) * 68  / 100
        ax.scatter(x, y, s=size, color=style["color"],
                   marker=style["marker"], alpha=style["alpha"], zorder=4,
                   edgecolors="#ffffff" if outcome == "goal" else "none",
                   linewidths=2.0)

    total_xg = sum(s.get("xg", 0) for s in shots)
    goals     = sum(1 for s in shots if s.get("outcome") == "goal")

    handles = [
        mpatches.Patch(color=ACCENT,    label="Goal"),
        mpatches.Patch(color="#3b82f6", label="Saved"),
        mpatches.Patch(color="#9ca3af", label="Blocked"),
        mpatches.Patch(color="#4b5563", label="Off Target"),
    ]
    _add_legend(fig, handles)

    fig.text(0.5, 0.055,
             f"{goals} Goals  ·  {total_xg:.2f} xG",
             ha="center", va="bottom", fontsize=15, fontweight="700",
             color=TEXT_C, fontfamily="Montserrat")

    _add_labels(fig, title, subtitle, brand)
    return fig


# ── Chance Creation ───────────────────────────────────────────────────────────

def draw_chance_creation(data: dict, title: str, subtitle: str, brand: str) -> plt.Figure:
    cc      = data.get("chance_creation", {})
    chances = cc.get("chances", [])

    fig = _make_canvas()
    ax  = _make_pitch_ax(fig)
    _draw_pitch_lines(ax)

    def cx(v): return float(v) * 105 / 100
    def cy(v): return float(v) * 68  / 100

    OUTCOME_C = {
        "goal":    ACCENT,
        "shot":    "#3b82f6",
        "blocked": "#9ca3af",
    }

    for c in chances:
        col = OUTCOME_C.get(c.get("outcome", "shot"), "#3b82f6")
        ax.annotate("",
                    xy=(cx(c["x_end"]), cy(c["y_end"])),
                    xytext=(cx(c["x_start"]), cy(c["y_start"])),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=2.2, mutation_scale=18),
                    zorder=4)
        ax.scatter(cx(c["x_start"]), cy(c["y_start"]),
                   s=60, color=col, alpha=0.7, zorder=5, edgecolors="none")

    handles = [
        mpatches.Patch(color=ACCENT,    label="Led to Goal"),
        mpatches.Patch(color="#3b82f6", label="Led to Shot"),
        mpatches.Patch(color="#9ca3af", label="Blocked / Lost"),
    ]
    _add_legend(fig, handles)
    _add_labels(fig, title, subtitle, brand)
    return fig


def dispatch_draw(data: dict, title: str, subtitle: str, brand: str) -> plt.Figure:
    vt = data.get("viz_type", "").lower()
    if "pass" in vt:
        return draw_pass_network(data, title, subtitle, brand)
    elif "touch" in vt or "heat" in vt:
        return draw_touch_map(data, title, subtitle, brand)
    elif "shot" in vt:
        return draw_shot_map(data, title, subtitle, brand)
    elif "chance" in vt or "creation" in vt:
        return draw_chance_creation(data, title, subtitle, brand)
    else:
        # Fallback: try to guess from keys present
        if "pass_network" in data:
            return draw_pass_network(data, title, subtitle, brand)
        elif "touch_map" in data:
            return draw_touch_map(data, title, subtitle, brand)
        elif "shot_map" in data:
            return draw_shot_map(data, title, subtitle, brand)
        elif "chance_creation" in data:
            return draw_chance_creation(data, title, subtitle, brand)
        raise ValueError(f"Could not determine viz type from: {vt}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

col_upload, col_output = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown('<div style="font-size:10px;font-weight:900;letter-spacing:.2em;color:#6b7280;text-transform:uppercase;margin-bottom:12px;">INPUT</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        key="upload",
        label_visibility="collapsed",
    )
    st.markdown('<div class="upload-hint">PNG / JPG / WEBP · Pass networks, touch maps, shot maps, chance creation</div>', unsafe_allow_html=True)

    if uploaded:
        img_bytes = uploaded.read()
        # Convert to PNG for consistency
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes_png = buf.getvalue()

        st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
        st.image(img, caption="Original screenshot", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        run_btn = st.button("⚡ EXTRACT & REDRAW", use_container_width=True)
    else:
        run_btn = False
        img_bytes_png = None

with col_output:
    st.markdown('<div style="font-size:10px;font-weight:900;letter-spacing:.2em;color:#6b7280;text-transform:uppercase;margin-bottom:12px;">OUTPUT</div>', unsafe_allow_html=True)

    output_placeholder = st.empty()
    json_placeholder   = st.empty()

    if not uploaded:
        output_placeholder.markdown("""
        <div style="background:#0d1220;border:1px dashed #1e2d4a;border-radius:12px;
                    padding:60px 20px;text-align:center;color:#374151;">
          <div style="font-size:32px;margin-bottom:12px;">🎨</div>
          <div style="font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;">
            Redraw will appear here
          </div>
        </div>
        """, unsafe_allow_html=True)

if run_btn and img_bytes_png:
    if not api_key:
        st.error("⚠️ Enter your Anthropic API key in the sidebar to continue.")
    else:
        with col_output:
            with st.spinner("🔍 Claude is reading your viz..."):
                try:
                    hint = viz_type if viz_type != "Auto-detect" else "unknown — please detect from the image"
                    extracted = call_claude_vision(img_bytes_png, hint, api_key)
                except json.JSONDecodeError as e:
                    st.error(f"JSON parse error: {e}\n\nClaude may have returned non-JSON — check your API key and try again.")
                    st.stop()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            # Auto-fill title/subtitle from extracted metadata if not set
            auto_title    = title_text or extracted.get("team_name", "")
            auto_subtitle = subtitle_text or " · ".join(filter(None, [
                extracted.get("opponent", ""),
                extracted.get("competition", ""),
                extracted.get("date", ""),
            ]))

            with st.spinner("🎨 Drawing your professional viz..."):
                try:
                    fig = dispatch_draw(extracted, auto_title, auto_subtitle, brand_text)
                except ValueError as e:
                    st.error(f"Drawing error: {e}")
                    st.stop()

            # Render figure — exact 1920×1080 (19.2in × 10.8in @ 100dpi)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100,
                        facecolor=BG, bbox_inches=None)
            plt.close(fig)
            img_out = buf.getvalue()

            output_placeholder.image(img_out, use_column_width=True)

            st.download_button(
                "⬇️  Download PNG",
                data=img_out,
                file_name=f"{(auto_title or 'viz').replace(' ', '_')}_redrawn.png",
                mime="image/png",
                use_container_width=True,
            )

            # Show extracted JSON in expander
            with json_placeholder.expander("📋 Extracted data (JSON)", expanded=False):
                st.json(extracted)

# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px;">
  <div style="background:#0d1220;border:1px solid #1e2d4a;border-radius:10px;padding:20px;">
    <div style="font-size:20px;margin-bottom:8px;">📸</div>
    <div style="font-size:10px;font-weight:900;letter-spacing:.15em;color:#ef4444;text-transform:uppercase;margin-bottom:6px;">01 · Upload</div>
    <div style="color:#6b7280;font-size:12px;line-height:1.6;">Screenshot any football viz — from Opta Analyst, WhoScored, Wyscout, FBref or anywhere else.</div>
  </div>
  <div style="background:#0d1220;border:1px solid #1e2d4a;border-radius:10px;padding:20px;">
    <div style="font-size:20px;margin-bottom:8px;">🤖</div>
    <div style="font-size:10px;font-weight:900;letter-spacing:.15em;color:#ef4444;text-transform:uppercase;margin-bottom:6px;">02 · Extract</div>
    <div style="color:#6b7280;font-size:12px;line-height:1.6;">Claude vision reads every node, dot and line — extracting coordinates, values and labels into structured JSON.</div>
  </div>
  <div style="background:#0d1220;border:1px solid #1e2d4a;border-radius:10px;padding:20px;">
    <div style="font-size:20px;margin-bottom:8px;">✨</div>
    <div style="font-size:10px;font-weight:900;letter-spacing:.15em;color:#ef4444;text-transform:uppercase;margin-bottom:6px;">03 · Redraw</div>
    <div style="color:#6b7280;font-size:12px;line-height:1.6;">The data is redrawn from scratch in your brand style — dark theme, Montserrat typography, custom accent colour.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:24px;padding:16px;background:#0d1220;border:1px solid #1e2d4a;border-radius:10px;">
  <div style="font-size:10px;font-weight:900;letter-spacing:.15em;color:#6b7280;text-transform:uppercase;margin-bottom:8px;">Supported viz types</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="background:#1e2d4a;color:#94a3b8;padding:4px 10px;border-radius:4px;font-size:11px;font-weight:700;">Pass Network</span>
    <span style="background:#1e2d4a;color:#94a3b8;padding:4px 10px;border-radius:4px;font-size:11px;font-weight:700;">Touch Map</span>
    <span style="background:#1e2d4a;color:#94a3b8;padding:4px 10px;border-radius:4px;font-size:11px;font-weight:700;">Shot Map</span>
    <span style="background:#1e2d4a;color:#94a3b8;padding:4px 10px;border-radius:4px;font-size:11px;font-weight:700;">Heat Map</span>
    <span style="background:#1e2d4a;color:#94a3b8;padding:4px 10px;border-radius:4px;font-size:11px;font-weight:700;">Chance Creation</span>
  </div>
</div>
""", unsafe_allow_html=True)
