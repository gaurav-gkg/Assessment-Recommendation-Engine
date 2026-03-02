"""
SHL Assessment Recommendation Engine – Streamlit Frontend
===========================================================
Run with:  streamlit run frontend/app.py
"""

from __future__ import annotations

import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import streamlit as st
import pandas as pd

from config import settings

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .metric-card {
            background: #f0f4ff;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
        }
        .badge {
            display: inline-block;
            background: #e0e7ff;
            color: #3730a3;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.8em;
            margin-right: 4px;
        }
        .yes-badge { background: #d1fae5; color: #065f46; }
        .no-badge  { background: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://www.shl.com/wp-content/themes/shl/assets/images/logo.svg",
        width=120,
    )
    st.title("⚙️ Settings")

    api_url = st.text_input(
        "API Base URL",
        value=f"http://localhost:{settings.API_PORT}",
        help="URL of the running FastAPI backend.",
    )

    num_results = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=10,
        value=10,
        step=1,
    )

    st.markdown("---")
    st.caption("SHL AI Research Engineer – Assessment Recommendation Engine")

    # Health check
    if st.button("Check API Health", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                st.success("✅ API is healthy")
            else:
                st.error(f"⚠️ Unexpected response: {r.text}")
        except Exception as e:
            st.error(f"❌ Cannot reach API: {e}")

# ---------------------------------------------------------------------------
# Main Interface
# ---------------------------------------------------------------------------

st.title("🎯 SHL Assessment Recommendation Engine")
st.markdown(
    "Enter a **natural language hiring query**, paste a **job description**, "
    "or provide a **URL** to a job posting — and get the most relevant "
    "SHL Individual Test Solutions recommended."
)

# Input tabs
tab_query, tab_jd, tab_url = st.tabs(["💬 Query", "📄 Job Description", "🌐 URL"])

with tab_query:
    query_text = st.text_area(
        "Hiring query",
        placeholder="e.g. We need a mid-level Java developer who collaborates well under pressure.",
        height=120,
    )
    submit_query = st.button("Get Recommendations", key="btn_query", type="primary")

with tab_jd:
    jd_text = st.text_area(
        "Paste full job description here",
        placeholder="Copy-paste the complete job description …",
        height=250,
    )
    submit_jd = st.button("Get Recommendations", key="btn_jd", type="primary")

with tab_url:
    url_input = st.text_input(
        "Job description URL",
        placeholder="https://example.com/careers/software-engineer",
    )
    submit_url = st.button("Get Recommendations", key="btn_url", type="primary")


# ---------------------------------------------------------------------------
# Determine active input
# ---------------------------------------------------------------------------

active_query: str = ""
submit = False

if submit_query and query_text.strip():
    active_query = query_text.strip()
    submit = True
elif submit_jd and jd_text.strip():
    active_query = jd_text.strip()
    submit = True
elif submit_url and url_input.strip():
    active_query = url_input.strip()
    submit = True


# ---------------------------------------------------------------------------
# Call API & display results
# ---------------------------------------------------------------------------

if submit:
    if not active_query:
        st.warning("Please enter a query, job description, or URL.")
    else:
        with st.spinner("🔍 Analysing query and retrieving assessments …"):
            try:
                resp = requests.post(
                    f"{api_url}/recommend",
                    json={"query": active_query, "num_results": num_results},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API. "
                    "Make sure the FastAPI server is running "
                    f"(`uvicorn api.main:app --port {settings.API_PORT}`)."
                )
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API error {resp.status_code}: {resp.text}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.stop()

        recommendations = data.get("recommended_assessments", [])

        if not recommendations:
            st.warning("No recommendations returned. Try a different query.")
        else:
            st.success(
                f"✅ Found **{len(recommendations)}** relevant assessment(s) "
                f"for your query."
            )

            # ── Summary table (all results) ──────────────────────────────── #
            st.subheader("📊 Results Overview")

            table_rows = []
            for i, rec in enumerate(recommendations, 1):
                dur = rec.get("duration")
                table_rows.append(
                    {
                        "#": i,
                        "Assessment": rec["name"],
                        "Link": rec["url"],
                        "Test Types": ", ".join(rec.get("test_type", [])) or "—",
                        "Duration": f"{dur} min" if dur else "—",
                        "Remote": "✅" if rec.get("remote_support") == "Yes" else "❌",
                        "Adaptive": "✅" if rec.get("adaptive_support") == "Yes" else "❌",
                    }
                )

            df = pd.DataFrame(table_rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Assessment": st.column_config.TextColumn("Assessment"),
                    "Link": st.column_config.LinkColumn("Link", display_text="🔗 Open"),
                },
            )

            # ── Detailed cards ───────────────────────────────────────────── #
            st.subheader("📋 Detailed View")

            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['name']}", expanded=(i == 1)):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Description:** {rec.get('description') or '_No description available._'}")
                        if rec.get("test_type"):
                            badges = " ".join(
                                f'<span class="badge">{t}</span>'
                                for t in rec["test_type"]
                            )
                            st.markdown(f"**Test Types:** {badges}", unsafe_allow_html=True)

                    with col2:
                        dur = rec.get("duration")
                        st.metric("Duration", f"{dur} min" if dur else "—")
                        remote_label = rec.get("remote_support", "No")
                        adaptive_label = rec.get("adaptive_support", "No")
                        is_remote = remote_label == "Yes"
                        is_adaptive = adaptive_label == "Yes"
                        st.markdown(
                            f'Remote: <span class="{"yes-badge" if is_remote else "no-badge"}">'
                            f"{remote_label}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'Adaptive: <span class="{"yes-badge" if is_adaptive else "no-badge"}">'
                            f"{adaptive_label}</span>",
                            unsafe_allow_html=True,
                        )

                    st.link_button("🔗 View on SHL Catalogue", rec["url"])

            # ── Export ───────────────────────────────────────────────────── #
            st.subheader("💾 Export")
            export_df = pd.DataFrame(
                [
                    {
                        "Assessment Name": r["name"],
                        "URL": r["url"],
                        "Test Types": ", ".join(r.get("test_type", [])),
                        "Duration (min)": r.get("duration", ""),
                        "Remote Support": r.get("remote_support", "No"),
                        "Adaptive Support": r.get("adaptive_support", "No"),
                    }
                    for r in recommendations
                ]
            )
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_bytes,
                file_name="shl_recommendations.csv",
                mime="text/csv",
            )
