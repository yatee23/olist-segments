"""Olist customer segmentation UI.

Run with:  streamlit run app.py

Expects two files in ./data or next to app.py
  final_.json               LLM interpretation + recommendation per cluster
  customers_clustered.csv   one row per customer with a cluster column
The CSV is optional. Without it the app still shows the LLM outputs and
the model selection results reported in Section 4.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Olist customer segments", page_icon="🛒", layout="wide")

# ---------------------------------------------------------------- config
APP_DIR = Path(__file__).parent


def find(name):
    """Look in ./data first, then next to app.py."""
    for p in (APP_DIR / "data" / name, APP_DIR / name):
        if p.exists():
            return p
    return APP_DIR / "data" / name


def find_llm_json():
    """Use final_.json if present, otherwise any JSON in the folder shaped like the LLM output."""
    exact = find("final_.json")
    if exact.exists():
        return exact
    candidates = sorted(list(APP_DIR.glob("*.json")) + list((APP_DIR / "data").glob("*.json")),
                        key=lambda p: (not p.name.lower().startswith("final"),
                                       "merged" not in p.name.lower(), p.name))
    for p in candidates:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            first = next(iter(d.values()))
            if "interpretation" in first and "recommendation" in first:
                return p
        except Exception:
            continue
    return exact


JSON_PATH = find_llm_json()
CSV_PATH = find("customers_clustered.csv")
ID_COL = "customer_unique_id"
CLUSTER_CANDIDATES = ["cluster", "segment", "kmeans_cluster", "kmeans_label", "cluster_label",
                      "cluster_id", "segment_id", "label", "labels"]

# One colour per segment, used in every chart so segments are easy to track
SEG_COLOURS = ["#2340C8", "#E8A317", "#1B9E77", "#D1495B", "#7B4FA0", "#4A8FB8"]

# Clustering inputs shown in original units (raw versions rebuilt from _log if needed)
FEATURES = {
    "recency_days": "Recency (days)",
    "n_orders": "Orders",
    "n_items": "Items bought",
    "total_order_value": "Total spend (R$)",
    "freight_ratio": "Freight ratio",
    "max_installments": "Max instalments",
    "payment_sequential_count": "Payments per order",
    "seller_distance_km": "Seller distance (km)",
    "n_categories": "Categories per order",
    "purchase_quarter": "Purchase quarter",
    "is_weekend_purchase": "Weekend share",
    "is_holiday_season": "Holiday season share",
    "product_weight_g": "Item weight (g)",
    "product_length_cm": "Item length (cm)",
}
# Held out of clustering (leakage), used for profiling only
HELD_OUT = {
    "review_score": "Review score",
    "delivery_delay_days": "Delivery delay (days)",
    "order_late": "Late order rate",
    "approval_delay_days": "Approval delay (days)",
    "carrier_delay_days": "Carrier delay (days)",
    "review_creation_delay_days": "Review creation delay (days)",
    "review_answer_delay_days": "Review answer delay (days)",
    "shipping_limit_gap_days": "Shipping limit gap (days)",
}
SHARE_PREFIXES = {
    "broad_category_": "Product category",
    "uses_": "Payment method",
    "customer_state_": "Customer state",
    "city_tier_": "City tier",
}

# ---------------------------------------------------------------- report results (Section 4.2)
K_TABLE = pd.DataFrame({
    "k": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Silhouette": [0.147, 0.161, 0.155, 0.154, 0.162, 0.157, 0.170, 0.176, 0.171, 0.163],
    "Davies-Bouldin": [2.181, 1.797, 1.679, 1.765, 1.637, 1.503, 1.419, 1.350, 1.405, 1.446],
    "Calinski-Harabasz": [15511, 14779, 13713, 13032, 12600, 12671, 12569, 12282.5, 11837.4, 11346.2],
    "Composite": [0, 0.81, 0.72, 0.62, 1.0, None, None, None, None, None],
    "Smallest cluster %": [46.4, 3.9, 3.9, 3.5, 3.5, 0.27, 0.27, 0.17, 0.17, 0.17],
})
K_TABLE["Passes 3% size filter"] = K_TABLE["Smallest cluster %"] >= 3

BASELINES = pd.DataFrame({
    "Model": ["RFM baseline", "Full feature set, no reduction", "Full feature set + PCA (final)"],
    "Best k": [2, 3, 6],
    "Silhouette": [0.66, 0.121, 0.162],
    "What the segments capture": [
        "One-time vs repeat buyers only (96.9% / 3.1%)",
        "Repeat buyers, large/expensive goods, small/cheap goods",
        "Repeat buyers plus spend, freight, distance, instalments, recency, timing",
    ],
    "Held-out practical distinctness": ["None (all negligible)", "Not tested", "5 small effects, 5 negligible"],
})

LOADINGS = pd.DataFrame({
    "Component": [f"PC{i}" for i in range(1, 10)],
    "Main loadings (|value| > 0.3)": [
        "Max instalments 0.40, total order value 0.61, size/mass 0.42, freight ratio -0.46",
        "Items 0.64, orders 0.62, freight ratio 0.34",
        "Seller distance 0.77",
        "Purchase quarter 0.98",
        "Recency 0.40, payments per order 0.87",
        "Recency 0.86, payments per order -0.43",
        "Size/mass 0.74, max instalments -0.50",
        "Max instalments 0.70, freight ratio 0.44",
        "Orders 0.72, items -0.65",
    ],
    "Reading": ["Spend and item size", "Purchase volume", "Seller distance", "Purchase timing",
                "Split payments", "Recency", "Item size vs instalments", "Instalments and freight",
                "Orders vs items per order"],
})


# ---------------------------------------------------------------- loading
@st.cache_data
def load_llm(raw: bytes | None):
    data = json.loads(raw) if raw else json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def _rebuild_label(df, prefix, out, reference):
    """Turn one-hot columns back into a single label column (reference level = all zeros)."""
    if out in df.columns:
        return
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return
    sub = df[cols]
    lab = sub.idxmax(axis=1).str.replace(prefix, "", regex=False)
    lab[sub.max(axis=1) == 0] = reference
    df[out] = lab


def _cluster_col(cols):
    lower = {c.lower(): c for c in cols}
    return next((lower[c] for c in CLUSTER_CANDIDATES if c in lower), None)


def _discover_csv():
    """Find customer data in the app folder. The labels file is the CSV with a cluster column;
    the features file is the other CSV with the most columns. They are joined on
    customer_unique_id, or by row order if that is not possible."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH), CSV_PATH.name
    files = sorted({f for d in (APP_DIR, APP_DIR / "data") for pat in ("*.csv", "*.csv.gz") for f in d.glob(pat)})
    heads = {}
    for f in files:
        try:
            heads[f] = list(pd.read_csv(f, nrows=0).columns)
        except Exception as e:
            print(f"Could not read {f.name}: {e}")
    labelled = [f for f in heads if _cluster_col(heads[f])]
    if not labelled:
        return None, None
    lab_f = min(labelled, key=lambda f: len(heads[f]))
    others = [f for f in heads if f != lab_f]
    if not others:
        return pd.read_csv(lab_f), lab_f.name
    feat_f = max(others, key=lambda f: len(heads[f]))
    if len(heads[lab_f]) >= len(heads[feat_f]):
        return pd.read_csv(lab_f), lab_f.name
    feats, labs = pd.read_csv(feat_f), pd.read_csv(lab_f)
    ccol = _cluster_col(labs.columns)
    feats = feats.drop(columns=[c for c in feats.columns if c.lower() == ccol.lower()])
    if ID_COL in labs.columns and ID_COL in feats.columns:
        merged = feats.merge(labs[[ID_COL, ccol]].drop_duplicates(ID_COL), on=ID_COL, how="inner")
        if len(merged) > 0.9 * len(labs):
            return merged, f"{feat_f.name} + {lab_f.name}"
    if len(labs) == len(feats):
        feats[ccol] = labs[ccol].values
        if ID_COL not in feats.columns and ID_COL in labs.columns:
            feats[ID_COL] = labs[ID_COL].values
        return feats, f"{feat_f.name} + {lab_f.name} (joined by row order)"
    return labs, f"{lab_f.name} (could not join {feat_f.name})"


@st.cache_data
def load_customers(raw: bytes | None):
    if raw:
        from io import BytesIO
        df, source = pd.read_csv(BytesIO(raw)), "uploaded file"
    else:
        df, source = _discover_csv()
        if df is None:
            return None, None, None
    cluster_col = _cluster_col(df.columns)
    if cluster_col is None:
        return None, f"No cluster column found in {source}. Expected one of {CLUSTER_CANDIDATES}.", None
    df = df.rename(columns={cluster_col: "cluster"})
    df["cluster"] = df["cluster"].astype(int)
    for c in ["total_order_value", "seller_distance_km", "freight_ratio"]:
        if c not in df.columns and f"{c}_log" in df.columns:
            df[c] = np.expm1(df[f"{c}_log"])
    _rebuild_label(df, "customer_state_", "customer_state", "Other")
    _rebuild_label(df, "city_tier_", "city_tier", "rest")
    return df, None, source


def present(df, cols):
    return {c: lab for c, lab in cols.items() if df is not None and c in df.columns}


def share_columns(df):
    out = {}
    for prefix, group in SHARE_PREFIXES.items():
        for c in df.columns:
            if c.startswith(prefix) and pd.api.types.is_numeric_dtype(df[c]):
                out[c] = (group, c.replace(prefix, "").replace("_", " "))
    return out


def seg_name(llm, c, short=False):
    label = llm.get(c, {}).get("interpretation", {}).get("segment_label", f"Segment {c}")
    return f"{c}" if short else f"{c} · {label}"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Olist segments")
page = st.sidebar.radio(
    "View",
    ["Overview", "Segment profile", "Compare segments", "Model and validation", "Customer lookup"],
)
st.sidebar.divider()
json_up = st.sidebar.file_uploader("LLM outputs (JSON)", type="json") if not JSON_PATH.exists() else None
csv_up = st.sidebar.file_uploader("Clustered customers (CSV)", type="csv")

if not JSON_PATH.exists() and json_up is None:
    st.info("Put final_.json in the same folder as app.py (or a data folder inside it), or upload it in the sidebar.")
    st.stop()

llm = load_llm(json_up.getvalue() if json_up else None)
df, csv_error, csv_source = load_customers(csv_up.getvalue() if csv_up else None)
if csv_error:
    st.sidebar.error(csv_error)
clusters = sorted(llm.keys())
colour = {c: SEG_COLOURS[i % len(SEG_COLOURS)] for i, c in enumerate(clusters)}
colour_by_name = {seg_name(llm, c): colour[c] for c in clusters}

st.sidebar.caption(f"LLM outputs from {json_up.name if json_up else JSON_PATH.name}")
if df is None:
    st.sidebar.caption("No customer CSV with cluster labels found in the app folder. Upload one above.")
else:
    st.sidebar.caption(f"{len(df):,} customers loaded from {csv_source}")


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("Who buys on Olist")
    st.write("Six behavioural segments of 95,402 customers, built with k-means on nine PCA "
             "components and described by an LLM from aggregate cluster statistics only.")

    m = st.columns(5)
    m[0].metric("Customers", "95,402")
    m[1].metric("Segments", "6")
    m[2].metric("Silhouette", "0.162")
    m[3].metric("Davies-Bouldin", "1.637")
    m[4].metric("Seed stability (mean ARI)", "0.793")

    if df is not None:
        sizes = df["cluster"].value_counts().reindex(clusters).fillna(0).astype(int)
        sz = pd.DataFrame({"Segment": [seg_name(llm, c) for c in clusters],
                           "Customers": sizes.values,
                           "Share": (sizes / sizes.sum() * 100).round(1).values})
        fig = px.bar(sz.sort_values("Customers"), x="Customers", y="Segment", orientation="h",
                     color="Segment", color_discrete_map=colour_by_name, text="Share")
        fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, height=320, margin=dict(l=0, r=40, t=10, b=0),
                          yaxis_title=None, xaxis_title="Customers")
        st.subheader("Segment sizes")
        st.plotly_chart(fig, width="stretch")
    else:
        sizes = None

    st.subheader("The segments")
    for row in range(0, len(clusters), 3):
        cols = st.columns(3)
        for col, c in zip(cols, clusters[row:row + 3]):
            interp = llm[c]["interpretation"]
            with col.container(border=True):
                st.markdown(f"<div style='height:4px;background:{colour[c]};border-radius:2px;margin-bottom:8px'></div>",
                            unsafe_allow_html=True)
                st.markdown(f"**{interp['segment_label']}**")
                if sizes is not None:
                    st.caption(f"Segment {c}, {sizes[c]:,} customers ({sizes[c] / sizes.sum():.1%})")
                else:
                    st.caption(f"Segment {c}")
                st.write(interp["critical_insight"])


def page_profile():
    c = st.selectbox("Segment", clusters, format_func=lambda x: seg_name(llm, x))
    interp, rec = llm[c]["interpretation"], llm[c]["recommendation"]
    st.markdown(f"<div style='height:6px;background:{colour[c]};border-radius:3px'></div>", unsafe_allow_html=True)
    st.title(interp["segment_label"])
    if df is not None:
        n = int((df["cluster"] == c).sum())
        st.caption(f"Segment {c}, {n:,} customers ({n / len(df):.1%} of all customers)")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Behavioural profile")
        st.write(interp["behavioral_profile"])
        st.subheader("Key insight")
        st.info(interp["critical_insight"])
        st.subheader("Held-out outcomes")
        st.write(interp["held_out_outcomes"])
    with right:
        st.subheader("Tone and positioning")
        st.write(rec["tone_positioning"])
        st.subheader("Operational note")
        st.write(rec["operational_note"])

    st.subheader("Recommended campaigns")
    strategies = rec["campaign_strategies"]
    cols = st.columns(len(strategies))
    for col, s in zip(cols, strategies):
        with col.container(border=True):
            st.markdown(f"**{s['lever'].capitalize()}**")
            st.caption(f"Channel: {s['channel']}")
            st.write(s["offer"].capitalize())
            st.write(s["message_angle"])
            with st.expander("Evidence"):
                st.code(s["evidence"], language=None)
    st.caption("Generated by an LLM from aggregate cluster statistics. Every figure was checked against the input JSON by a team member.")

    if df is None:
        return
    seg = df[df["cluster"] == c]

    feats = {**present(df, FEATURES), **present(df, HELD_OUT)}
    if feats:
        st.subheader("Segment vs all customers")
        rows = []
        for col_, lab in feats.items():
            rows.append({"Feature": lab,
                         "Used in clustering": col_ in FEATURES,
                         "Segment mean": seg[col_].mean(), "Overall mean": df[col_].mean(),
                         "Segment median": seg[col_].median(), "Overall median": df[col_].median()})
        tab = pd.DataFrame(rows)
        st.dataframe(tab.style.format({k: "{:,.2f}" for k in tab.columns if "mean" in k or "median" in k}),
                     hide_index=True, width="stretch")

    shares = share_columns(df)
    if shares:
        st.subheader("Over and under represented groups")
        st.caption("Lift is the segment share divided by the overall share. Above 1.2 or below 0.8 counts as reportable. "
                   "Location is descriptive only and does not explain behaviour.")
        rows = []
        for col_, (group, lab) in shares.items():
            overall = df[col_].mean()
            if overall > 0:
                rows.append({"Group": group, "Level": lab, "Segment share": seg[col_].mean(),
                             "Overall share": overall, "Lift": seg[col_].mean() / overall})
        lift = pd.DataFrame(rows)
        group = st.radio("Group", lift["Group"].unique(), horizontal=True)
        g = lift[lift["Group"] == group].sort_values("Lift")
        fig = px.bar(g, x="Lift", y="Level", orientation="h",
                     hover_data={"Segment share": ":.1%", "Overall share": ":.1%", "Lift": ":.2f"})
        fig.update_traces(marker_color=[colour[c] if abs(v - 1) >= 0.2 else "#C5CBDA" for v in g["Lift"]])
        fig.add_vline(x=1, line_dash="dot", line_color="#18213A")
        fig.update_layout(height=60 + 34 * len(g), margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")


def page_compare():
    st.title("How the segments differ")
    if df is None:
        st.info("Put customers_clustered.csv next to app.py to compare segments on the underlying features.")
        return
    feats = present(df, FEATURES)
    held = present(df, HELD_OUT)
    include_held = st.toggle("Include held-out variables (not used in clustering)", value=False)
    use = {**feats, **(held if include_held else {})}
    if not use:
        st.warning("The loaded customer file has cluster labels but no feature columns, so there is nothing "
                   "to compare. Check the sidebar to see which files were loaded.")
        return
    top_n = st.slider("Features shown", 5, len(use), min(12, len(use)))

    means = df.groupby("cluster")[list(use)].mean().reindex(clusters)
    z = (means - df[list(use)].mean()) / df[list(use)].std(ddof=0).replace(0, np.nan)
    order = z.abs().max().sort_values(ascending=False).index[:top_n]
    z, means = z[order], means[order]

    fig = go.Figure(go.Heatmap(
        z=z.values, x=[use[c] for c in order], y=[seg_name(llm, c) for c in clusters],
        text=means.map(lambda v: f"{v:,.2f}").values, texttemplate="%{text}",
        colorscale="RdBu_r", zmid=0, zmin=-1.5, zmax=1.5,
        colorbar=dict(title="Std. diff<br>from mean"),
        hovertemplate="%{y}<br>%{x}<br>Segment mean %{text}<br>Std. diff %{z:.2f}<extra></extra>"))
    fig.update_layout(height=110 + 60 * len(clusters), margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(side="top", tickangle=-30))
    st.plotly_chart(fig, width="stretch")
    st.caption("Cell text is the segment mean in original units. Colour is the standardised difference from the "
               "overall mean. Red is above average, blue is below, darker is more extreme.")

    st.subheader("Distribution of one feature")
    f = st.selectbox("Feature", list(use), format_func=lambda x: use[x])
    plot_df = df[["cluster", f]].dropna().copy()
    plot_df["Segment"] = plot_df["cluster"].map(lambda x: seg_name(llm, x))
    log = st.checkbox("Log scale", value=plot_df[f].skew() > 1)
    fig = px.box(plot_df, x="Segment", y=f, color="Segment", color_discrete_map=colour_by_name,
                 points=False, log_y=log)
    fig.update_layout(showlegend=False, height=420, xaxis_title=None, yaxis_title=use[f],
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    if {"PC1", "PC2"}.issubset(df.columns):
        st.subheader("Customers in PCA space")
        samp = df.sample(min(8000, len(df)), random_state=42).copy()
        samp["Segment"] = samp["cluster"].map(lambda x: seg_name(llm, x))
        fig = px.scatter(samp, x="PC1", y="PC2", color="Segment", color_discrete_map=colour_by_name,
                         opacity=0.5, render_mode="webgl")
        fig.update_traces(marker_size=4)
        fig.update_layout(height=520, legend_title=None, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Random sample of 8,000 customers on the first two of nine components.")


def page_model():
    st.title("Model and validation")
    st.write("K-means (k-means++, n_init 10, random_state 42) on 9 PCA components covering 80% of variance. "
             "k was chosen by the highest silhouette among solutions whose smallest cluster holds at least 3% "
             "of customers, with the composite of silhouette and inverted Davies-Bouldin as the tie break.")

    long = K_TABLE.melt(id_vars=["k", "Passes 3% size filter"], value_vars=["Silhouette", "Davies-Bouldin"],
                        var_name="Index", value_name="Score")
    a, b = st.columns(2)
    for col, idx, note in [(a, "Silhouette", "higher is better"), (b, "Davies-Bouldin", "lower is better")]:
        d = long[long["Index"] == idx]
        fig = px.line(d, x="k", y="Score", markers=True)
        fig.update_traces(line_color="#2340C8")
        fig.add_scatter(x=d.loc[~d["Passes 3% size filter"], "k"], y=d.loc[~d["Passes 3% size filter"], "Score"],
                        mode="markers", marker=dict(color="#C5CBDA", size=11), name="Smallest cluster under 3%")
        fig.add_vline(x=6, line_dash="dot", line_color="#D1495B")
        fig.update_layout(title=f"{idx} ({note})", height=320, margin=dict(l=0, r=0, t=40, b=0),
                          legend=dict(orientation="h", y=-0.25))
        col.plotly_chart(fig, width="stretch")

    st.dataframe(K_TABLE, hide_index=True, width="stretch")

    st.subheader("Against the baselines")
    st.dataframe(BASELINES, hide_index=True, width="stretch")
    st.caption("Silhouette scores are not directly comparable across feature spaces. The RFM score mostly reflects "
               "one very large and one very small cluster.")

    st.subheader("Stability")
    st.write("Adjusted Rand Index across 6 seeds (15 pairs) ranged from 0.657 to 0.998 with a mean of 0.793, "
             "which indicates moderate stability approaching the 0.80 threshold for good stability (Steinley, 2004).")

    st.subheader("What the components capture")
    st.dataframe(LOADINGS, hide_index=True, width="stretch")


def page_lookup():
    st.title("Customer lookup")
    st.caption("Shows segment and coarse location only. ZIP codes, city names and coordinates are never displayed.")
    if df is None or ID_COL not in df.columns:
        st.info(f"Customer lookup needs customers_clustered.csv with a {ID_COL} column.")
        return
    a, b = st.columns([3, 2])
    pick = b.selectbox("Or pick a random customer from", clusters, format_func=lambda x: seg_name(llm, x))
    if b.button("Show a random customer"):
        st.session_state["cust"] = df.loc[df["cluster"] == pick, ID_COL].sample(1).iloc[0]
    cid = a.text_input("Customer ID", value=st.session_state.get("cust", "")).strip()
    if not cid:
        return
    hit = df[df[ID_COL] == cid]
    if hit.empty:
        st.warning("No customer with that ID. Check for extra spaces or paste the full hashed ID.")
        return
    r = hit.iloc[0]
    c = int(r["cluster"])
    with st.container(border=True):
        st.markdown(f"<div style='height:4px;background:{colour[c]};border-radius:2px;margin-bottom:8px'></div>",
                    unsafe_allow_html=True)
        st.markdown(f"**{llm[c]['interpretation']['segment_label']}** (segment {c})")
        m = st.columns(4)
        m[0].metric("State", r.get("customer_state", "n/a"))
        m[1].metric("City tier", str(r.get("city_tier", "n/a")).replace("_", " "))
        if "n_orders" in r:
            m[2].metric("Orders", int(r["n_orders"]))
        if "total_order_value" in r:
            m[3].metric("Total spend", f"R$ {r['total_order_value']:,.2f}")
        m2 = st.columns(4)
        if "recency_days" in r:
            m2[0].metric("Days since last order", int(r["recency_days"]))
        if "max_installments" in r:
            m2[1].metric("Max instalments", int(r["max_installments"]))
        if "review_score" in r and pd.notna(r["review_score"]):
            m2[2].metric("Review score", f"{r['review_score']:.1f} / 5")
        if "sentiment_score" in r and pd.notna(r["sentiment_score"]):
            m2[3].metric("Review sentiment", f"{r['sentiment_score']:.1f} / 5")
    st.write("Suggested campaigns for this segment")
    for s in llm[c]["recommendation"]["campaign_strategies"]:
        st.markdown(f"- **{s['lever'].capitalize()}** via {s['channel']}. {s['offer'].capitalize()}.")


{"Overview": page_overview, "Segment profile": page_profile, "Compare segments": page_compare,
 "Model and validation": page_model, "Customer lookup": page_lookup}[page]()
