Whiplash v0.1 — Extreme-Velocity Governance Console

Whiplash is a synthetic hypersonic-style governance environment built in Streamlit.
It simulates high-velocity flight conditions, risk envelopes, link degradation, and human-in-the-loop autonomy workflows — without any weapons logic, guidance stacks, or real aerodynamics.
This is a governance UI, not flight software.

The core concept:
At hypersonic speeds, the problem isn’t control — it’s trust, clarity, and gating.
Whiplash gives you a place to study that.

What This Repo Does
✔ Real-time synthetic telemetry

Mach, altitude, velocity, q-pressure

Thermal index, g-load

IMU drift + comm latency (environment-driven)

Phase classifier (PRELAUNCH → RETURN)

✔ Envelope-based risk + clarity engine

Each tick feeds into an envelope-pressure model to compute:

Clarity (%)

Risk (%)

Predicted risk

Operational state: STABLE → CRITICAL

✔ Human-gated autonomy proposals

When stressors stack, the system generates proposals:

reduce dynamic pressure

tighten thermal envelope

enable latency-safe mode

stabilize clarity

or just “hold profile”

Operator must explicitly approve.
Gate must be open.
Everything is auditable.

✔ Tamper-evident audit chain

Every event hashes the previous event → creates a lightweight blockchain-style chain:

proposals created

approvals/rejections

simulation starts/pauses

alerts

✔ UI Panels

Live synthetic PFD (pitch/roll visualization)

Speed/altitude timeline

Clarity/risk timeline

Proposal browser

Event feed

Audit chain

Raw telemetry table

Running Whiplash
1. Install dependencies
pip install streamlit numpy pandas plotly

2. Launch
streamlit run app.py

3. Use the UI

Pick envelope presets

Pick environment conditions

Hit Start Simulation

Approve/reject proposals as they appear

Observe clarity/risk dynamics

Architecture Overview
/simulate/

Synthetic physics model

Envelope-pressure + clarity/risk computation

Environmental modifiers

/governance/

Proposal generator

Human-gated approval flow

Audit-log chain

/ui/

Streamlit layout

Synthetic PFD

Charts, tables, metrics

Why This Exists

It’s a sandbox for building and testing governance UX patterns for systems operating at speeds where:

humans can’t see everything

autonomy can’t be trusted blindly

risk escalates exponentially

decisions require gating, prediction, and clarity metrics

Great for:

safety-first autonomy research

envelope exploration UI design

latency-sensitive decision workflows

human-machine trust modeling

Limitations (Intentional)

Not a physics simulator

No navigation/INS model

No real thermal model

No vehicle geometry

No weaponization logic

Data is synthetic by design

This is about governance patterns, not performance modeling.

Extending the System

Suggested future modules:

risk-aware mission planning

authority-level escalations

anomaly clustering

explainable autonomy rationales

telemetry playback / offline mode

multi-vehicle swarm governance

Security & Ethics

Whiplash is intentionally:

non-weaponized

unphysical

focused on human authority and transparency

built for safety research and governance UX

If you extend it, keep it in the same lane.

License

MIT — open to modify, fork, or embed in internal governance tooling.
