# ============================================================
#  dashboard.py — Streamlit Dashboard (Bonus)
#  Run: streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title = "Helmet Safety Dashboard",
    page_icon  = "⛑️",
    layout     = "wide"
)

st.title("⛑️ Helmet Safety Compliance Dashboard")
st.markdown("---")

CSV_PATH  = "output/compliance_logs.csv"
HEATMAP   = "output/heatmap.jpg"
SNAPS_DIR = "output/violations"

if not os.path.exists(CSV_PATH):
    st.warning("⚠️ Pehle `python main.py` chalao!")
    st.stop()

df = pd.read_csv(CSV_PATH)

# ── KPI Cards ────────────────────────────────────────────────
total    = df["track_id"].nunique()
compliant= df[df["status"]=="Compliant"]["track_id"].nunique()
violated = df[df["status"]=="Non-Compliant"]["track_id"].nunique()
rate     = round(compliant / total * 100 if total else 0, 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("👷 Total Workers",  total)
c2.metric("✅ Compliant",      compliant)
c3.metric("❌ Violations",     violated)
c4.metric("📈 Compliance Rate", f"{rate}%")

st.markdown("---")

# ── Charts ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Compliance Distribution")
    counts = df.groupby("status")["track_id"].nunique().reset_index()
    fig = px.pie(counts, names="status", values="track_id",
                 color="status",
                 color_discrete_map={
                     "Compliant"    : "#00cc66",
                     "Non-Compliant": "#ff4444"
                 }, hole=0.35)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Violations Over Time")
    viol_df = df[df["status"]=="Non-Compliant"]
    if not viol_df.empty:
        fig2 = px.histogram(
            viol_df, x="frame", nbins=30,
            color_discrete_sequence=["#ff4444"])
        st.plotly_chart(fig2, use_container_width=True)

# ── Violation Snapshots ───────────────────────────────────────
if os.path.exists(SNAPS_DIR):
    snaps = [f for f in os.listdir(SNAPS_DIR)
             if f.endswith(".jpg")]
    if snaps:
        st.markdown("---")
        st.subheader(f"📸 Violation Snapshots ({len(snaps)})")
        cols = st.columns(min(4, len(snaps)))
        for i, snap in enumerate(snaps[:8]):
            cols[i % 4].image(
                os.path.join(SNAPS_DIR, snap),
                caption=snap, use_column_width=True)

# ── Heatmap ──────────────────────────────────────────────────
if os.path.exists(HEATMAP):
    st.markdown("---")
    st.subheader("🗺️ Worker Movement Heatmap")
    st.image(HEATMAP, use_column_width=True)

# ── Raw Data ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Compliance Log")
st.dataframe(df, use_container_width=True)

csv_dl = df.to_csv(index=False).encode()
st.download_button("⬇️ Download CSV", csv_dl,
                   "compliance_logs.csv", "text/csv")
