import time
import math
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
page_title="Whiplash v0.1 — Extreme-Velocity Governance Console",
layout="wide",
)

# -----------------------------
# Constants & Presets
# -----------------------------
DT_SECONDS = 1.0
MAX_HISTORY = 600

ENVELOPE_PRESETS: Dict[str, Dict[str, float | str]] = {
"Conservative Range Test": {
"max_mach": 4.0,
"max_q_kpa": 800.0,
"max_g": 5.0,
"max_thermal_index": 0.80,
"max_latency_ms": 250.0,
"description": "Tight margins for early envelope expansion flights.",
},
"Nominal Demo Flight": {
"max_mach": 5.5,
"max_q_kpa": 1100.0,
"max_g": 7.0,
"max_thermal_index": 0.92,
"max_latency_ms": 300.0,
"description": "Balanced demonstration profile with headroom.",
},
"Aggressive Envelope Probe": {
"max_mach": 7.0,
"max_q_kpa": 1350.0,
"max_g": 8.5,
"max_thermal_index": 0.98,
"max_latency_ms": 350.0,
"description": "Pushes up against envelope limits for test-only runs.",
},
}

ENVIRONMENTS: Dict[str, Dict[str, float]] = {
"Clear Skies / Clean Link": {
"latency_base": 120.0,
"latency_jitter": 40.0,
"thermal_bias": 0.0,
"imu_drift_bias": 0.0,
},
"High Latency Link": {
"latency_base": 260.0,
"latency_jitter": 80.0,
"thermal_bias": 0.05,
"imu_drift_bias": 0.02,
},
"Thermal Stress Test": {
"latency_base": 180.0,
"latency_jitter": 60.0,
"thermal_bias": 0.18,
"imu_drift_bias": 0.01,
},
"Sensor Degradation": {
"latency_base": 220.0,
"latency_jitter": 90.0,
"thermal_bias": 0.10,
"imu_drift_bias": 0.04,
},

}

# -----------------------------
# Utility
# -----------------------------

def clamp(v: float, lo: float, hi: float) -> float:
return max(lo, min(hi, v))

def sha256_hash(data: Dict[str, Any]) -> str:
s = json.dumps(data, sort_keys=True)
return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -----------------------------
# Session State Init
# -----------------------------
def init_state():
ss = st.session_state
ss.tick = 0
ss.mission_time = 0.0
ss.running = False
ss.last_update = time.time()
ss.history: List[Dict[str, Any]] = []

ss.events: List[Dict[str, Any]] = []
ss.proposals: List[Dict[str, Any]] = []
ss.audit_log: List[Dict[str, Any]] = []
ss.prev_hash = "0" * 64
ss.gate_open = False
ss.preset = "Nominal Demo Flight"
ss.environment = "Clear Skies / Clean Link"

if "tick" not in st.session_state:
init_state()

# -----------------------------
# Audit & Events
# -----------------------------

def record_audit(event_type: str, payload: Dict[str, Any]):
ss = st.session_state
entry = {
"tick": ss.tick,
"timestamp": datetime.utcnow().isoformat() + "Z",
"event_type": event_type,
"payload": payload,
"prev_hash": ss.prev_hash,
}

entry["hash"] = sha256_hash(entry)
ss.prev_hash = entry["hash"]
ss.audit_log.append(entry)
ss.audit_log = ss.audit_log[-200:]

def log_event(level: str, msg: str, extras: Dict[str, Any] | None = None):
ss = st.session_state
payload: Dict[str, Any] = {
"timestamp": datetime.utcnow().strftime("%H:%M:%S"),
"level": level,
"msg": msg,
}
if extras:
payload.update(extras)
ss.events.append(payload)
ss.events = ss.events[-200:]

# -----------------------------
# Flight Physics / Risk
# -----------------------------

def classify_phase(mach: float, altitude_m: float, t: float) -> str:
if t < 5.0:

return "PRELAUNCH"
if mach < 0.8 and altitude_m < 2000.0:
return "TAKEOFF"
if mach < 3.5:
return "ASCENT"
if mach < 6.5 and altitude_m > 18000.0:
return "HYPERCRUISE"
if altitude_m < 5000.0 and mach < 1.5:
return "RETURN"
return "TRANSITION"

def compute_clarity_and_risk(
row: Dict[str, Any], preset_name: str
) -> Dict[str, Any]:
preset = ENVELOPE_PRESETS[preset_name]
mach = row["mach"]
q_kpa = row["q_kpa"]
g_load = row["g_load"]
thermal = row["thermal_index"]
latency = row["link_latency_ms"]
imu_drift = row["imu_drift_deg_s"]

mach_util = clamp(mach / float(preset["max_mach"]), 0.0, 1.5)
q_util = clamp(q_kpa / float(preset["max_q_kpa"]), 0.0, 1.6)
g_util = clamp(abs(g_load) / float(preset["max_g"]), 0.0, 1.8)

thermal_util = clamp(
thermal / float(preset["max_thermal_index"]), 0.0, 1.8
)
latency_util = clamp(
latency / float(preset["max_latency_ms"]), 0.0, 2.0
)
imu_util = clamp(imu_drift / 0.12, 0.0, 2.0) # 0.12 deg/s is "high"

envelope_pressure = (
0.25 * q_util
+ 0.20 * thermal_util
+ 0.18 * mach_util
+ 0.17 * g_util
+ 0.10 * latency_util
+ 0.10 * imu_util
)

clarity = clamp(100.0 - envelope_pressure * 28.0, 0.0, 100.0)

risk = clamp(
(envelope_pressure * 65.0) + (100.0 - clarity) * 0.35,
0.0,
100.0,
)

if risk < 25:

state = "STABLE"
elif risk < 45:
state = "STRESSED"
elif risk < 70:
state = "HIGH_RISK"
else:
state = "CRITICAL"

predicted_risk = clamp(
risk
+ 8.0 * (q_util - 0.8)
+ 8.0 * (thermal_util - 0.85)
+ 5.0 * (latency_util - 1.0),
0.0,
100.0,
)

return {
"clarity": round(clarity, 1),
"risk": round(risk, 1),
"predicted_risk": round(predicted_risk, 1),
"state": state,
"envelope_pressure": round(envelope_pressure, 3),
}

def generate_tick(prev_row: Dict[str, Any] | None) -> Dict[str, Any]:
ss = st.session_state
t = ss.mission_time + DT_SECONDS
env = ENVIRONMENTS[ss.environment]

rng = np.random.default_rng(int((time.time() * 1000) % 2**32))

if prev_row is None:
mach = 0.0
altitude_m = 0.0
g_load = 1.0
else:
mach = prev_row["mach"]
altitude_m = prev_row["altitude_m"]
g_load = prev_row["g_load"]

# Synthetic profile
if t <= 5.0:
mach = 0.0 + rng.normal(0.0, 0.01)
altitude_m = max(0.0, altitude_m + rng.normal(0.0, 1.0))
elif t <= 35.0:
mach += rng.uniform(0.05, 0.12)
altitude_m += rng.uniform(250.0, 550.0)
elif t <= 65.0:
mach += rng.uniform(0.03, 0.09)
altitude_m += rng.uniform(200.0, 420.0)

elif t <= 90.0:
mach += rng.uniform(-0.02, 0.04)
altitude_m += rng.uniform(-180.0, 220.0)
else:
mach += rng.uniform(-0.08, -0.02)
altitude_m += rng.uniform(-450.0, -180.0)

mach = clamp(mach, 0.0, 7.5)
altitude_m = clamp(altitude_m, 0.0, 42000.0)

a_local = 295.0 # m/s, fake local speed of sound
velocity_mps = mach * a_local

rho0 = 1.225
rho = rho0 * math.exp(-altitude_m / 8500.0)
q_pa = 0.5 * rho * velocity_mps**2
q_kpa = clamp(q_pa / 1000.0, 0.0, 1600.0)

thermal_index = clamp(
0.2
+ 0.65 * (mach / 7.0)
+ 0.2 * (q_kpa / 1400.0)
+ env["thermal_bias"]
+ rng.normal(0.0, 0.025),
0.0,
1.2,

)

base_g = 1.0
if t <= 15.0:
g_load = base_g + rng.normal(0.5, 0.3)
elif t <= 55.0:
g_load = base_g + rng.normal(1.2, 0.5)
elif t <= 80.0:
g_load = base_g + rng.normal(2.5, 0.8)
else:
g_load = base_g + rng.normal(0.8, 0.4)

g_load = clamp(g_load, -1.0, 9.0)

latency = max(
40.0,
env["latency_base"] + rng.normal(0.0, env["latency_jitter"]),
)

imu_drift = clamp(
abs(rng.normal(0.02 + env["imu_drift_bias"], 0.02)),
0.0,
0.15,
)

phase = classify_phase(mach, altitude_m, t)

row: Dict[str, Any] = {
"tick": ss.tick + 1,
"mission_time_s": round(t, 1),
"phase": phase,
"mach": round(mach, 3),
"velocity_mps": round(velocity_mps, 1),
"altitude_m": round(altitude_m, 1),
"q_kpa": round(q_kpa, 1),
"thermal_index": round(thermal_index, 3),
"g_load": round(g_load, 3),
"link_latency_ms": round(latency, 1),
"imu_drift_deg_s": round(imu_drift, 4),
}

clarity_pack = compute_clarity_and_risk(row, ss.preset)
row.update(
{
"clarity": clarity_pack["clarity"],
"risk": clarity_pack["risk"],
"predicted_risk": clarity_pack["predicted_risk"],
"state": clarity_pack["state"],
"envelope_pressure": clarity_pack["envelope_pressure"],
}
)

return row

# -----------------------------
# Proposal Engine
# -----------------------------

def maybe_generate_proposal(latest: Dict[str, Any]):
ss = st.session_state
open_pending = [p for p in ss.proposals if p["status"] == "PENDING"]
if len(open_pending) >= 5:
return

preset = ENVELOPE_PRESETS[ss.preset]
triggers: List[str] = []

if latest["risk"] > 55.0:
triggers.append("overall_high")
if latest["q_kpa"] > 0.9 * float(preset["max_q_kpa"]):
triggers.append("q_near_limit")
if latest["thermal_index"] > 0.9 * float(preset["max_thermal_index"]):
triggers.append("thermal_near_limit")
if latest["link_latency_ms"] > 1.1 * float(preset["max_latency_ms"]):
triggers.append("latency_high")
if latest["clarity"] < 72.0:

triggers.append("clarity_low")
if latest["predicted_risk"] > 70.0:
triggers.append("rising_risk")

if not triggers:
return

pid = len(ss.proposals) + 1

if "latency_high" in triggers:
title = "Shift to latency-safe mode (reduce authority / tighten bounds)"
action = "enable_latency_safe_mode"
elif "thermal_near_limit" in triggers:
title = "Tighten thermal envelope & cap Mach under stress band"
action = "tighten_thermal_envelope"
elif "q_near_limit" in triggers:
title = "Reduce peak dynamic pressure by trimming ascent profile"
action = "reduce_dynamic_pressure"
elif "clarity_low" in triggers:
title = "Hold envelope; prioritize sensor stability and link integrity"
action = "stabilize_clarity"
else:
title = "Hold current profile, observation-only stance"
action = "hold_profile"

rationale_bits: List[str] = []

if "overall_high" in triggers:
rationale_bits.append(f"risk {latest['risk']:.1f}% above 55% band")
if "q_near_limit" in triggers:
rationale_bits.append("dynamic pressure near envelope limit")
if "thermal_near_limit" in triggers:
rationale_bits.append("thermal index near max envelope")
if "latency_high" in triggers:
rationale_bits.append("link latency above preset band")
if "clarity_low" in triggers:
rationale_bits.append(
"clarity below 72% target for this profile"
)
if "rising_risk" in triggers:
rationale_bits.append("predicted risk indicates upward slope")

proposal = {
"id": pid,
"created": datetime.utcnow().strftime("%H:%M:%S"),
"title": title,
"action": action,
"rationale": "; ".join(rationale_bits),
"status": "PENDING",
"snapshot": {
"tick": latest["tick"],
"mission_time_s": latest["mission_time_s"],

"phase": latest["phase"],
"mach": latest["mach"],
"q_kpa": latest["q_kpa"],
"thermal_index": latest["thermal_index"],
"g_load": latest["g_load"],
"clarity": latest["clarity"],
"risk": latest["risk"],
"predicted_risk": latest["predicted_risk"],
"link_latency_ms": latest["link_latency_ms"],
},
}

ss.proposals.append(proposal)
log_event(
"PROPOSAL", f"Proposal #{pid} queued: {title}", {"risk": latest["risk"]}
)
record_audit(
"proposal_created", {"id": pid, "title": title, "triggers": triggers}
)

def apply_proposal(proposal: Dict[str, Any]):
# Governance-only / simulated actions.
action = proposal["action"]
if action == "enable_latency_safe_mode":
log_event("INFO", "Latency-safe mode enabled (simulated).")

elif action == "tighten_thermal_envelope":
log_event("INFO", "Thermal envelope tightened by 10% (simulated).")
elif action == "reduce_dynamic_pressure":
log_event("INFO", "Dynamic pressure profile trimmed (simulated).")
elif action == "stabilize_clarity":
log_event("INFO", "Stability-focused adjustments applied (simulated).")
elif action == "hold_profile":
log_event("INFO", "Profile hold applied (simulated).")

record_audit(
"proposal_applied", {"id": proposal["id"], "action": proposal["action"]}
)

# -----------------------------
# UI Components
# -----------------------------
def render_header():
col1, col2 = st.columns([3, 1])
with col1:
st.markdown(
"""
<h1 style='margin-bottom:0;'>
Whiplash v0.1 — Extreme-Velocity Governance Console
</h1>
""",

unsafe_allow_html=True,
)
st.caption(
"Synthetic hypersonic-style test profile. "
"Non-weapon, governance-focused demo. Human-gated autonomy only."
)
with col2:
st.write("")
st.write("")
st.toggle(
"Human Gate Open",
key="gate_open",
help="Gate must be open before approving any proposal.",
)

def render_sidebar():
ss = st.session_state

st.sidebar.markdown("### Envelope Preset")
preset_names = list(ENVELOPE_PRESETS.keys())
current_index = (
preset_names.index(ss.preset) if ss.preset in preset_names else 1
)
preset_choice = st.sidebar.radio(
"Select preset",

options=preset_names,
index=current_index,
)
ss.preset = preset_choice
st.sidebar.info(
f"**{ss.preset}** — {ENVELOPE_PRESETS[ss.preset]['description']}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Environment Profile")
env_names = list(ENVIRONMENTS.keys())
env_index = (
env_names.index(ss.environment) if ss.environment in env_names else 0
)
env_choice = st.sidebar.radio(
"Select environment",
options=env_names,
index=env_index,
)
ss.environment = env_choice

st.sidebar.markdown("---")
st.sidebar.caption(
"All values are synthetic and non-physical. "
"This is a concept UI for governance, not flight software."
)

def render_controls():
ss = st.session_state
ctrl1, ctrl2, ctrl3 = st.columns(3)

with ctrl1:
if st.button("▶ Start Simulation", type="primary", use_container_width=True):
ss.running = True
log_event("INFO", "Simulation started.")
record_audit("sim_started", {})

with ctrl2:
if st.button("⏸ Pause", use_container_width=True):
ss.running = False
log_event("INFO", "Simulation paused.")
record_audit("sim_paused", {})

with ctrl3:
if st.button("⟲ Reset", use_container_width=True):
init_state()
log_event("INFO", "Simulation reset.")
record_audit("sim_reset", {})

def render_top_metrics(latest: pd.Series):

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
st.metric("Tick", int(latest["tick"]))
st.metric("Phase", latest["phase"])
with m2:
st.metric("Mach", f"{latest['mach']:.2f}")
st.metric("Velocity (m/s)", f"{latest['velocity_mps']:.0f}")
with m3:
st.metric("Altitude (m)", f"{latest['altitude_m']:.0f}")
st.metric("q (kPa)", f"{latest['q_kpa']:.0f}")
with m4:
st.metric("Thermal Index", f"{latest['thermal_index']:.2f}")
st.metric("G-Load", f"{latest['g_load']:.2f}")
with m5:
st.metric("Clarity (%)", f"{latest['clarity']:.1f}")
st.metric("Risk (%)", f"{latest['risk']:.1f}")

# -----------------------------
# Synthetic PFD (Flight Deck)
# -----------------------------
def render_pfd(latest: pd.Series):
st.subheader("Synthetic Flight Deck")

# Rough attitude from Mach + g-load (just for visual flavor)
pitch_deg = float(

np.interp(
latest["mach"],
[0.0, 7.5],
[-5.0, 25.0],
)
)
roll_deg = float(
np.interp(
latest["g_load"],
[0.0, 9.0],
[-35.0, 35.0],
)
)

pitch_rad = math.radians(pitch_deg)
roll_rad = math.radians(roll_deg)

def rotate_point(x: float, y: float) -> tuple[float, float]:
xr = x * math.cos(roll_rad) - y * math.sin(roll_rad)
yr = x * math.sin(roll_rad) + y * math.cos(roll_rad)
return xr, yr

# Horizon line endpoints
x1, y1 = rotate_point(-1.2, 0.0)
x2, y2 = rotate_point(1.2, 0.0)

# Pitch shift (small vertical offset)
pitch_shift = pitch_deg / 45.0 # -1 .. +1 range-ish
y1 += pitch_shift
y2 += pitch_shift

fig = go.Figure()

# Sky and ground
fig.add_shape(
type="rect",
x0=-2,
y0=0 + pitch_shift,
x1=2,
y1=2,
fillcolor="#1f2e5b",
line=dict(width=0),
layer="below",
)
fig.add_shape(
type="rect",
x0=-2,
y0=-2,
x1=2,
y1=0 + pitch_shift,
fillcolor="#4a3116",
line=dict(width=0),

layer="below",
)

# Horizon
fig.add_trace(
go.Scatter(
x=[x1, x2],
y=[y1, y2],
mode="lines",
line=dict(width=3, color="#ffffff"),
showlegend=False,
)
)

# Center aircraft symbol
fig.add_trace(
go.Scatter(
x=[0, -0.12, 0.12, 0],
y=[0.0, -0.08, -0.08, 0.0],
mode="lines",
line=dict(width=3, color="#ffff00"),
showlegend=False,
)
)

# Velocity / altitude / Mach tapes as text blocks

fig.add_annotation(
x=-1.6,
y=0.8,
text=f"VEL<br>{latest['velocity_mps']:.0f} m/s",
showarrow=False,
font=dict(color="white", size=12),
align="left",
bgcolor="rgba(0,0,0,0.5)",
)
fig.add_annotation(
x=1.6,
y=0.8,
text=f"ALT<br>{latest['altitude_m']:.0f} m",
showarrow=False,
font=dict(color="white", size=12),
align="right",
bgcolor="rgba(0,0,0,0.5)",
)
fig.add_annotation(
x=0.0,
y=1.4,
text=f"Mach {latest['mach']:.2f}",
showarrow=False,
font=dict(color="white", size=14),
align="center",
bgcolor="rgba(0,0,0,0.5)",

)

# Risk / clarity strip
fig.add_annotation(
x=0.0,
y=-1.4,
text=(
f"Clarity {latest['clarity']:.1f}% | "
f"Risk {latest['risk']:.1f}% | "
f"State: {latest['state']}"
),
showarrow=False,
font=dict(color="white", size=11),
align="center",
bgcolor="rgba(0,0,0,0.6)",
)

fig.update_xaxes(
visible=False,
range=[-2, 2],
fixedrange=True,
)
fig.update_yaxes(
visible=False,
range=[-2, 2],
fixedrange=True,

)
fig.update_layout(
margin=dict(l=0, r=0, t=0, b=0),
paper_bgcolor="#000000",
plot_bgcolor="#000000",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Charts & Tables
# -----------------------------
def render_middle_row(df: pd.DataFrame, latest: pd.Series):
top_left, top_mid, top_right = st.columns([1.4, 1.2, 0.9])

with top_left:
st.subheader("Speed / Altitude Profile")
tail = df.tail(220).set_index("mission_time_s")
if not tail.empty:
chart_df = tail[["mach", "altitude_m"]]
st.line_chart(chart_df)
st.caption("Mach and altitude over mission time.")
else:
st.info("No data yet. Start the simulation.")

with top_mid:
st.subheader("Clarity & Risk Timeline")
tail = df.tail(220).set_index("mission_time_s")
if not tail.empty:
cr_df = tail[["clarity", "risk", "predicted_risk"]]
st.line_chart(cr_df)
st.caption("Clarity, current risk, and predicted risk over time.")
else:
st.info("No data yet. Start the simulation.")

with top_right:
st.subheader("Envelope Snapshot")
preset = ENVELOPE_PRESETS[st.session_state.preset]
st.metric("Envelope Pressure", f"{latest['envelope_pressure']:.2f}")
st.metric("Latency (ms)", f"{latest['link_latency_ms']:.0f}")
st.metric("IMU Drift (°/s)", f"{latest['imu_drift_deg_s']:.3f}")
st.markdown("**Preset Limits:**")
st.write(
f"- Max Mach: {preset['max_mach']:.1f}\n"
f"- Max q: {preset['max_q_kpa']:.0f} kPa\n"
f"- Max G: {preset['max_g']:.1f} g\n"
f"- Max Thermal Index: {preset['max_thermal_index']:.2f}\n"
f"- Max Latency: {preset['max_latency_ms']:.0f} ms"
)

def render_proposals_and_events(df: pd.DataFrame):
bottom_left, bottom_mid, bottom_right = st.columns([1.3, 1.1, 1.0])
ss = st.session_state

# Proposals
with bottom_left:
st.subheader("Human-Gated Proposals")
if not ss.proposals:
st.info("No proposals yet. Simulation will surface them under stress.")
else:
labels = [
f"#{p['id']} [{p['status']}] {p['title']}"
for p in ss.proposals
]
selected_label = st.selectbox("Select proposal", options=labels)
selected_id = int(selected_label.split(" ")[0].replace("#", ""))
proposal = next(p for p in ss.proposals if p["id"] == selected_id)

st.markdown(f"**Title:** {proposal['title']}")
st.markdown(f"**Status:** `{proposal['status']}`")
st.markdown(f"**Created:** {proposal['created']}")
st.markdown("**Rationale:**")
st.write(proposal["rationale"])
st.markdown("**Snapshot at Creation:**")
st.json(proposal["snapshot"])

col_a, col_b, col_c = st.columns(3)
with col_a:
approve = st.button(
"Approve",
disabled=proposal["status"] != "PENDING",
key=f"approve_{proposal['id']}",
)
with col_b:
reject = st.button(
"Reject",
disabled=proposal["status"] != "PENDING",
key=f"reject_{proposal['id']}",
)
with col_c:
defer = st.button(
"Defer",
disabled=proposal["status"] != "PENDING",
key=f"defer_{proposal['id']}",
)

if approve and proposal["status"] == "PENDING":
if not ss.gate_open:
st.warning("Human gate is CLOSED. Open it before approving.")
log_event(
"WARN",
f"Attempted approval of proposal #{proposal['id']} "

"with gate closed.",
)
record_audit(
"approve_blocked_gate_closed",
{"id": proposal["id"]},
)
else:
proposal["status"] = "APPROVED"
apply_proposal(proposal)
log_event(
"APPROVED",
f"Proposal #{proposal['id']} approved by operator.",
)
record_audit(
"proposal_approved",
{
"id": proposal["id"],
"title": proposal["title"],
},
)

if reject and proposal["status"] == "PENDING":
proposal["status"] = "REJECTED"
log_event(
"REJECTED",
f"Proposal #{proposal['id']} rejected by operator.",

)
record_audit(
"proposal_rejected",
{
"id": proposal["id"],
"title": proposal["title"],
},
)

if defer and proposal["status"] == "PENDING":
proposal["status"] = "DEFERRED"
log_event(
"INFO",
f"Proposal #{proposal['id']} deferred by operator.",
)
record_audit(
"proposal_deferred",
{
"id": proposal["id"],
"title": proposal["title"],
},
)

# Events + audit
with bottom_mid:
st.subheader("Event Feed")

level_filter = st.selectbox(
"Filter level",
["ALL", "INFO", "WARN", "ALERT", "PROPOSAL", "APPROVED", "REJECTED"],
)
search = st.text_input("Search messages")

events = ss.events
if level_filter != "ALL":
events = [e for e in events if e["level"] == level_filter]
if search:
events = [
e for e in events if search.lower() in e["msg"].lower()
]

if not events:
st.info("No events match current filters.")
else:
df_events = pd.DataFrame(events[-100:])[::-1]
st.dataframe(df_events, use_container_width=True, height=260)

if ss.audit_log:
st.subheader("Audit Chain (Short View)")
audit_df = pd.DataFrame(ss.audit_log[-30:])
audit_df["hash_short"] = audit_df["hash"].apply(
lambda h: h[:10] + "..."

)
audit_df["prev_short"] = audit_df["prev_hash"].apply(
lambda h: h[:10] + "..."
)
st.dataframe(
audit_df[["tick", "event_type", "prev_short", "hash_short"]],
use_container_width=True,
height=220,
)
else:
st.caption(
"Tamper-evident chain from previous hash → current hash."
)
st.info("No audit entries yet.")

# Raw telemetry
with bottom_right:
st.subheader("Raw Telemetry (Last 40 Ticks)")
cols = [
"tick",
"mission_time_s",
"phase",
"mach",
"velocity_mps",
"altitude_m",
"q_kpa",

"thermal_index",
"g_load",
"link_latency_ms",
"imu_drift_deg_s",
"clarity",
"risk",
"predicted_risk",
"state",
]
view = df[cols].tail(40).set_index("tick")
st.dataframe(
view,
use_container_width=True,
height=260,
)

# -----------------------------
# Main Loop
# -----------------------------
def main():
render_header()
render_sidebar()
render_controls()

ss = st.session_state

now = time.time()

if ss.running and now - ss.last_update >= DT_SECONDS:
prev = ss.history[-1] if ss.history else None
new_row = generate_tick(prev)
ss.tick = new_row["tick"]
ss.mission_time = new_row["mission_time_s"]
ss.history.append(new_row)
ss.history = ss.history[-MAX_HISTORY:]
ss.last_update = now

if new_row["state"] in ("HIGH_RISK", "CRITICAL"):
log_event(
"ALERT",
f"{new_row['state']} at tick {new_row['tick']} — "
f"risk {new_row['risk']:.1f}%, "
f"Mach {new_row['mach']:.2f}, "
f"q {new_row['q_kpa']:.0f} kPa",
)
record_audit("high_risk_state", new_row)

maybe_generate_proposal(new_row)

if not ss.history:
st.info("Press **Start Simulation** to bring Whiplash online.")
return

df = pd.DataFrame(ss.history)
latest = df.iloc[-1]

st.markdown("---")
render_top_metrics(latest)
st.markdown("---")

# New synthetic PFD section
pfd_col, _ = st.columns([2, 1])
with pfd_col:
render_pfd(latest)

st.markdown("---")
render_middle_row(df, latest)
st.markdown("---")
render_proposals_and_events(df)

if ss.running:
time.sleep(0.25)
st.experimental_rerun()

if __name__ == "__main__":
main()
