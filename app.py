
import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader

# ─── NLTK ─────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk()

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="文献データ解析アプリ",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; }
    h3 { color: #34495e; }
    div[data-testid="stForm"] { max-width: 400px; margin: auto; }
    .kpi-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
    .kpi-card { background:#f8fafc; border-radius:8px; padding:12px 20px;
                min-width:140px; text-align:center; border-top:4px solid #3498db; flex:1; }
    .kpi-card.green  { border-top-color:#2ecc71; }
    .kpi-card.orange { border-top-color:#f39c12; }
    .kpi-card.purple { border-top-color:#9b59b6; }
    .kpi-label { font-size:0.78rem; color:#7f8c8d; margin-bottom:4px; }
    .kpi-value { font-size:1.8rem; font-weight:700; color:#2c3e50; line-height:1; }
    .kpi-unit  { font-size:0.72rem; color:#aaa; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 認証
# ═══════════════════════════════════════════════════════════════
def load_config():
    """
    認証設定の読み込み（優先順位）
      1. Google Colab Secrets  (PUBMED_APP_CONFIG)
      2. Streamlit Cloud Secrets (st.secrets)
      3. ローカルの config.yaml
    """
    # ── Colab 環境 ──────────────────────────────
    try:
        from google.colab import userdata
        import json as _json
        raw = userdata.get("PUBMED_APP_CONFIG")
        if raw:
            return _json.loads(raw)
    except Exception:
        pass

    # ── Streamlit Cloud 環境 ────────────────────
    if "credentials" in st.secrets:
        return {
            "credentials": dict(st.secrets["credentials"]),
            "cookie": dict(st.secrets["cookie"]),
        }

    # ── ローカル開発環境 ────────────────────────
    with open("config.yaml") as f:
        return yaml.load(f, Loader=SafeLoader)

config = load_config()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# 1. ログインウィジェットの表示
authenticator.login("main")

# 2. 認証状態の確認
if st.session_state["authentication_status"]:
    # ログイン成功時の処理
    name = st.session_state["name"]
    username = st.session_state["username"]
    auth_status = True
    
    # サイドバーにログアウトボタン等を表示
    authenticator.logout("ログアウト", "sidebar")
    
elif st.session_state["authentication_status"] is False:
    st.error("ユーザー名またはパスワードが間違っています")
    auth_status = False
    
elif st.session_state["authentication_status"] is None:
    st.warning("ユーザー名とパスワードを入力してください")
    auth_status = None
    st.stop()
    
if auth_status is False:
    st.error("ユーザー名またはパスワードが正しくありません")
    st.stop()
elif auth_status is None:
    st.info("ユーザー名とパスワードを入力してください")
    st.stop()

authenticator.logout("ログアウト", "sidebar")
st.sidebar.markdown(f"👤 **{name}** さん")
st.sidebar.markdown("---")

# ═══════════════════════════════════════════════════════════════
# MEDLINE パーサー
# ═══════════════════════════════════════════════════════════════
def parse_medline(text: str) -> pd.DataFrame:
    text = re.sub(r"\n      ", " ", text)
    lines = text.splitlines()
    records, current = [], {}
    for line in lines:
        if len(line) < 6:
            continue
        tag = line[:4].strip()
        val = line[6:].strip() if len(line) > 6 else ""
        if not tag:
            continue
        if tag == "PMID":
            if current:
                records.append(current)
            current = {"PMID": val}
        elif tag in current:
            if isinstance(current[tag], list):
                current[tag].append(val)
            else:
                current[tag] = [current[tag], val]
        else:
            current[tag] = val
    if current:
        records.append(current)
    df = pd.DataFrame(records)
    if "DP" in df.columns:
        df["year"] = df["DP"].apply(
            lambda x: int(str(x)[:4]) if pd.notna(x) and str(x)[:4].isdigit() else np.nan
        )
    # list/str 混在を全列で統一（pyarrow ArrowInvalid 対策）
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else [])
            )
    return df

# ═══════════════════════════════════════════════════════════════
# NLP ヘルパー
# ═══════════════════════════════════════════════════════════════
STOP_EN = set(stopwords.words("english"))

def tokenize(text: str):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOP_EN and len(w) > 2]

def make_bigrams(text: str):
    words = tokenize(text)
    # アンダースコア結合で TfidfVectorizer がトークン分割しないようにする
    return [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]

def tfidf_top(group_texts: dict, top_n: int, mode: str = "bigram") -> pd.DataFrame:
    groups = list(group_texts.keys())
    func   = tokenize if mode == "word" else make_bigrams
    processed = [" ".join(func(group_texts[g])) for g in groups]
    if not any(processed):
        return pd.DataFrame(columns=["group", "term", "tfidf"])
    # bigram はアンダースコア結合済みなのでトークナイザをそのまま使う
    vec = TfidfVectorizer(max_features=5000, token_pattern=r"[^\s]+")
    try:
        X = vec.fit_transform(processed)
    except ValueError:
        return pd.DataFrame(columns=["group", "term", "tfidf"])
    terms = vec.get_feature_names_out()
    rows = []
    for i, g in enumerate(groups):
        scores = X[i].toarray()[0]
        for j in scores.argsort()[::-1][:top_n]:
            if scores[j] > 0:
                # bigram: アンダースコアをスペースに戻して表示
                display_term = terms[j].replace("_", " ") if mode == "bigram" else terms[j]
                rows.append({"group": g, "term": display_term, "tfidf": scores[j]})
    return pd.DataFrame(rows)

def tfidf_chart(group_texts: dict, top_n: int, mode: str, title: str):
    result = tfidf_top(group_texts, top_n, mode)
    if result.empty:
        st.info("データが不足しています。")
        return
    # 値の高い順に並べる
    result = result.sort_values("tfidf", ascending=True)
    fig = px.bar(
        result, x="tfidf", y="term", color="group",
        facet_col="group", facet_col_wrap=2, orientation="h",
        labels={"tfidf": "TF-IDF", "term": ""}, title=title,
        height=max(400, len(result) * 22),
    )
    fig.update_yaxes(matches=None, showticklabels=True, categoryorder="total ascending")
    fig.update_layout(showlegend=False)
    # facetタイトル（group=Journal名）の文字を大きく・"group=" プレフィックスを除去
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1],
        font=dict(size=14, color="#2c3e50"),
    ))
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# ネットワーク ヘルパー
# ═══════════════════════════════════════════════════════════════
def shorten_name(name: str) -> str:
    parts = str(name).split(", ")
    return f"{parts[0]}, {parts[1][0]}" if len(parts) == 2 else str(name)[:20]

def build_coauthor_graph(fau_series, top_n: int = 50):
    flat = [shorten_name(a) for authors in fau_series for a in authors]
    top_set = {a for a, _ in Counter(flat).most_common(top_n)}
    G = nx.Graph()
    edge_count = Counter()
    for authors in fau_series:
        short = [shorten_name(a) for a in authors if shorten_name(a) in top_set]
        for i in range(len(short)):
            for j in range(i + 1, len(short)):
                edge_count[tuple(sorted([short[i], short[j]]))] += 1
    for (u, v), w in edge_count.items():
        G.add_edge(u, v, weight=w)
    return G

def build_bigram_graph(texts: list, top_n: int = 80):
    pair_count = Counter()
    for text in texts:
        pair_count.update(make_bigrams(text))
    G = nx.DiGraph()
    for phrase, cnt in pair_count.most_common(top_n):
        parts = phrase.split(" ")
        if len(parts) == 2:
            G.add_edge(parts[0], parts[1], weight=cnt)
    return G

def nx_to_plotly(G, centrality: str = "pagerank", title: str = "",
                 directed: bool = False, show_labels: bool = True):
    if not G.nodes:
        return go.Figure()
    np.random.seed(42)
    pos = nx.spring_layout(G, seed=42, k=1.2 / math.sqrt(max(len(G.nodes), 1)))
    cent = (nx.pagerank(G, weight="weight") if centrality == "pagerank"
            else nx.betweenness_centrality(G, weight="weight"))
    G_u = G.to_undirected() if directed else G
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        node_community = {}
        for idx, com in enumerate(greedy_modularity_communities(G_u, weight="weight")):
            for node in com:
                node_community[node] = idx
    except Exception:
        node_community = {n: 0 for n in G.nodes}
    colors = px.colors.qualitative.Plotly
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    max_cent = max(cent.values()) if cent else 1
    fig = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines",
                   line=dict(width=0.7, color="#aaa"), hoverinfo="none", showlegend=False),
        go.Scatter(
            x=[pos[n][0] for n in G.nodes],
            y=[pos[n][1] for n in G.nodes],
            mode="markers+text" if show_labels else "markers",
            text=list(G.nodes) if show_labels else [],
            textposition="top center", textfont=dict(size=9),
            marker=dict(
                size=[max(8, cent.get(n, 0) / max_cent * 40) for n in G.nodes],
                color=[colors[node_community.get(n, 0) % len(colors)] for n in G.nodes],
                line=dict(width=1, color="white"),
            ),
            hovertext=[f"{n}<br>centrality: {cent.get(n,0):.4f}" for n in G.nodes],
            hoverinfo="text", showlegend=False,
        ),
    ])
    fig.update_layout(
        title=title, title_x=0.5,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20), height=600,
    )
    return fig

def get_community_texts(community_nodes, df_src):
    texts = []
    for _, row in df_src.iterrows():
        fau = row.get("FAU", [])
        if not isinstance(fau, list):
            continue
        if {shorten_name(a) for a in fau} & community_nodes and pd.notna(row.get("AB")):
            texts.append(str(row["AB"]))
    return " ".join(texts)

# ═══════════════════════════════════════════════════════════════
# MeSH ヘルパー
# ═══════════════════════════════════════════════════════════════
MESH_EXCLUDE = {
    "Humans", "Female", "Male", "Adult", "Adolescent", "Young Adult", "Child",
    "Middle Aged", "Aged", "Animals", "Child, Preschool", "Infant",
    "Aged, 80 and over", "Infant, Newborn", "Case Reports",
}

def normalize_mesh(m: str) -> str:
    return re.sub(r"/\*?.*$", "", str(m)).strip()

@st.cache_data
def build_mesh_pivot(df: pd.DataFrame, top_n: int, min_year: int):
    rows = []
    for _, row in df.iterrows():
        yr = row.get("year")
        if pd.isna(yr):
            continue
        for m in (row.get("MH") or []):
            nm = normalize_mesh(m)
            if nm not in MESH_EXCLUDE:
                rows.append({"year": int(yr), "mesh": nm})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    mdf = pd.DataFrame(rows)
    top_mesh = mdf["mesh"].value_counts().head(top_n).index.tolist()
    pivot = (mdf[mdf["mesh"].isin(top_mesh)]
             .groupby(["year", "mesh"]).size().unstack(fill_value=0))
    pivot = pivot[pivot.index >= min_year]
    year_totals = mdf[mdf["year"] >= min_year].groupby("year").size()
    pivot_norm = pivot.div(year_totals, axis=0).fillna(0) * 100
    return pivot_norm, mdf

def detect_bursts(mdf: pd.DataFrame, top_n: int, min_year: int, z_th: float):
    y_max = int(mdf["year"].max())
    burst_mesh = mdf["mesh"].value_counts().head(top_n).index.tolist()
    results = []
    for term in burst_mesh:
        series = (mdf[mdf["mesh"] == term].groupby("year").size()
                  .reindex(range(min_year, y_max + 1), fill_value=0))
        if series.sum() < 5:
            continue
        mean, std = series.mean(), series.std()
        if std == 0:
            continue
        z = (series - mean) / std
        in_b, b_start = False, None
        for yr_val, zv in z.items():
            if zv >= z_th and not in_b:
                in_b, b_start = True, yr_val
            elif zv < z_th and in_b:
                slice_s = series[b_start:yr_val].dropna()
                if slice_s.empty:
                    in_b = False
                    continue
                results.append({
                    "MeSH": term, "バースト開始": b_start, "バースト終了": yr_val - 1,
                    "期間": yr_val - b_start,
                    "ピーク件数": int(slice_s.max()),
                    "ピーク年": int(slice_s.idxmax()),
                    "最大Zスコア": round(float(z[b_start:yr_val].dropna().max()), 2),
                })
                in_b = False
        if in_b:
            slice_s = series[b_start:].dropna()
            if not slice_s.empty:
                results.append({
                    "MeSH": term, "バースト開始": b_start, "バースト終了": y_max,
                    "期間": y_max - b_start + 1,
                    "ピーク件数": int(slice_s.max()),
                    "ピーク年": int(slice_s.idxmax()),
                    "最大Zスコア": round(float(z[b_start:].dropna().max()), 2),
                })
    return (pd.DataFrame(results)
            .sort_values(["バースト開始", "最大Zスコア"], ascending=[True, False])
            .reset_index(drop=True) if results else pd.DataFrame())

# ═══════════════════════════════════════════════════════════════
# セッション状態
# ═══════════════════════════════════════════════════════════════
if "df" not in st.session_state:
    st.session_state.df = None

# ═══════════════════════════════════════════════════════════════
# ナビゲーション
# ═══════════════════════════════════════════════════════════════
PAGES = ["🏠 Home", "📂 Data Upload", "📊 Overview",
         "🔤 自然言語処理", "🕸️ ネットワーク分析", "🧬 MeSH分析"]
page = st.sidebar.radio("ページ選択", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("PubMed文献解析アプリ v2.1")

df = st.session_state.df

# ═══════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 📚 文献データ分析・可視化アプリケーション")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**できること**
- ✅ MEDLINE形式データのパース・概要確認
- ✅ 整形データ（基本情報・Abstract・著者）のCSVダウンロード
- ✅ 文献数推移・Top Journal（+Others）分析
- ✅ Top Author ランキング（Co / 1st / Last）・著者推移バブルチャート
- ✅ TF-IDF 特徴語抽出（Bigram主体）：Journal別・著者別・年度別
- ✅ 共著ネットワーク（コミュニティ別特徴語・期間比較・新規共著ペア）
- ✅ 単語共起ネットワーク（Bigram・3期間比較）
- ✅ MeSHヒートマップ & バースト検知タイムライン
        """)
    with col2:
        st.markdown("""
**使い方**
1. [PubMed](https://pubmed.ncbi.nlm.nih.gov/) で任意のキーワードで文献を検索
2. **MEDLINE形式** でデータをダウンロード（Send to → Citation manager → MEDLINE）
3. 「📂 Data Upload」からファイルをアップロード
4. 各ページで分析結果を確認・ダウンロード

**ページ構成**

| ページ | 内容 |
|--------|------|
| 📂 Data Upload | アップロード・データ概要・CSVダウンロード |
| 📊 Overview | 文献数推移・Journal・Author・バブルチャート |
| 🔤 自然言語処理 | TF-IDF Bigram特徴語（Journal/著者/年度別） |
| 🕸️ ネットワーク分析 | 共著・コミュニティ・期間比較・単語共起 |
| 🧬 MeSH分析 | ヒートマップ・バースト検知タイムライン |
        """)
    st.info("⬅️ 左のサイドバーから「📂 Data Upload」を選んでください。")

# ═══════════════════════════════════════════════════════════════
# DATA UPLOAD
# ═══════════════════════════════════════════════════════════════
elif page == "📂 Data Upload":
    st.markdown("## 解析データのアップロード・基礎解析")

    uploaded = st.file_uploader(
        "MEDLINE形式テキストファイルをアップロード（.txt）",
        type=["txt"],
        help="PubMedからMEDLINE形式でダウンロードしたファイル。目安1,500文献程度まで。",
    )
    if uploaded:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        with st.spinner("解析中..."):
            parsed = parse_medline(raw)
        st.session_state.df = parsed
        df = parsed
        st.success(f"✅ {len(df):,} 件の文献を読み込みました")

    if df is not None:
        # ── KPI カード ──────────────────────────────────────────
        valid_years  = df["year"].dropna() if "year" in df.columns else pd.Series([], dtype=float)
        year_range   = (f"{int(valid_years.min())} 〜 {int(valid_years.max())}"
                        if not valid_years.empty else "N/A")
        pmid_count   = df["PMID"].nunique() if "PMID" in df.columns else len(df)
        ab_count     = int(df["AB"].notna().sum()) if "AB" in df.columns else 0
        if "FAU" in df.columns:
            all_authors  = [a for lst in df["FAU"] for a in (lst if isinstance(lst, list) else [])]
            author_count = len(set(all_authors))
            fau_count    = int(df["FAU"].apply(lambda x: len(x) > 0).sum())
        else:
            author_count, fau_count = 0, 0

        st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">PMID数（ユニーク）</div>
    <div class="kpi-value">{pmid_count:,}</div>
    <div class="kpi-unit">件</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">著者数（ユニーク）</div>
    <div class="kpi-value">{author_count:,}</div>
    <div class="kpi-unit">名</div>
  </div>
  <div class="kpi-card orange">
    <div class="kpi-label">Abstract あり</div>
    <div class="kpi-value">{ab_count:,}</div>
    <div class="kpi-unit">件</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">著者情報あり文献</div>
    <div class="kpi-value">{fau_count:,}</div>
    <div class="kpi-unit">件</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── 詳細メタ情報 ────────────────────────────────────────
        with st.expander("📋 詳細情報", expanded=True):
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"- **ファイル名**: `{uploaded.name if uploaded else '（読込済み）'}`")
                st.markdown(f"- **総文献数**: {len(df):,} 件")
                st.markdown(f"- **年度範囲**: {year_range}")
            with mc2:
                st.markdown(f"- **検出カラム数**: {len(df.columns)} 個")
                st.markdown(f"- **著者情報あり**: {fau_count:,} 件")
                mh_cnt = int(df["MH"].apply(lambda x: len(x) > 0).sum()) if "MH" in df.columns else 0
                st.markdown(f"- **MeSHデータあり**: {mh_cnt:,} 件")

        # ── 生データプレビュー ───────────────────────────────────
        st.markdown("### 生データプレビュー（先頭10件）")
        preview_cols = [c for c in ["PMID", "TI", "TA", "DP", "year", "AB"] if c in df.columns]
        st.dataframe(df[preview_cols].head(10), use_container_width=True)

        # ── 整形データ ダウンロード ──────────────────────────────
        st.markdown("### 整形データのダウンロード")
        c1, c2, c3 = st.columns(3)

        basic_cols = [c for c in ["PMID", "TA", "TI", "DP", "year"] if c in df.columns]
        with c1:
            st.markdown("**① 基本情報**")
            basic_df = df[basic_cols].copy()
            st.dataframe(basic_df.head(8), use_container_width=True)
            st.download_button("⬇️ basic_information.csv",
                               basic_df.to_csv(index=False).encode(),
                               "basic_information.csv", "text/csv")

        with c2:
            st.markdown("**② Abstract**")
            if "AB" in df.columns:
                ab_df = df[["PMID", "AB"]].dropna(subset=["AB"])
                st.dataframe(ab_df.head(8), use_container_width=True)
                st.download_button("⬇️ abstract.csv",
                                   ab_df.to_csv(index=False).encode(),
                                   "abstract.csv", "text/csv")

        with c3:
            st.markdown("**③ 共著者データ**")
            if "FAU" in df.columns:
                au_rows = []
                for _, row in df.iterrows():
                    fau = row.get("FAU", [])
                    entry = {"PMID": row.get("PMID", ""), "DP": row.get("DP", "")}
                    for i, a in enumerate(fau if isinstance(fau, list) else [], start=1):
                        entry[str(i)] = a
                    au_rows.append(entry)
                author_df = pd.DataFrame(au_rows)
                num_cols = sorted(
                    [c for c in author_df.columns if c not in ("PMID", "DP")],
                    key=lambda x: int(x)
                )
                author_df = author_df[["PMID", "DP"] + num_cols]
                st.dataframe(author_df.head(8), use_container_width=True)
                st.download_button("⬇️ author.csv",
                                   author_df.to_csv(index=False).encode(),
                                   "author.csv", "text/csv")
    else:
        st.info("⬆️ ファイルをアップロードしてください。")

# ═══════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Overview":
    st.markdown("## Overview")
    if df is None:
        st.warning("先に「📂 Data Upload」でデータをアップロードしてください。")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 文献数推移", "📰 Top Journals", "📊 Journal推移", "👥 Top Authors", "🫧 著者推移"]
    )

    # ── Tab1: 文献数推移 ─────────────────────────────────────────
    with tab1:
        if "year" in df.columns:
            yd = df.dropna(subset=["year"]).copy()
            yd["year"] = yd["year"].astype(int)
            y_min, y_max = int(yd["year"].min()), int(yd["year"].max())
            yr = st.slider("期間変更", y_min, y_max, (max(y_min, y_max - 30), y_max))
            bd = (yd[yd["year"].between(yr[0], yr[1])]
                  .groupby("year").size().reset_index(name="count"))
            fig = px.bar(bd, x="year", y="count", color_discrete_sequence=["#3498db"],
                         labels={"year": "Year", "count": "文献数"},
                         title=f"文献数推移（{yr[0]}〜{yr[1]}）")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ literature_count.csv",
                               bd.to_csv(index=False).encode(),
                               "literature_count.csv", "text/csv")

    # ── Tab2: Top Journals（全期間 Top20 グラフ）─────────────────
    with tab2:
        if "TA" in df.columns:
            TOP_J = st.slider("表示件数", 10, 30, 20, key="topj_n")
            jdf = df.dropna(subset=["TA"]).copy()
            top_j = jdf["TA"].value_counts().head(TOP_J).reset_index()
            top_j.columns = ["Journal", "総文献数"]


            fig_ja = px.bar(
                top_j, x="総文献数", y="Journal", orientation="h",
                color_discrete_sequence=["#2ecc71"],
                title=f"Top {TOP_J} Journal（全期間・総文献数順）",
            )
            fig_ja.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=max(400, TOP_J * 26),
            )
            st.plotly_chart(fig_ja, use_container_width=True)
            st.dataframe(top_j.reset_index(drop=True), use_container_width=True)
            st.download_button("⬇️ top_journals.csv",
                               top_j.to_csv(index=False).encode(),
                               "top_journals.csv", "text/csv")

    # ── Tab3: Journal推移（Top10+Others 積み上げ）────────────────
    with tab3:
        if "TA" in df.columns and "year" in df.columns:
            TOP_STACK = st.slider("個別表示Journal数", 5, 20, 10, key="stack_n")
            jdf2 = df.dropna(subset=["TA", "year"]).copy()
            jdf2["year"] = jdf2["year"].astype(int)
            top_list = jdf2["TA"].value_counts().head(TOP_STACK).index.tolist()
            jdf2["TA_grp"] = jdf2["TA"].where(jdf2["TA"].isin(top_list), other="Others")
            trend = (jdf2.groupby(["year", "TA_grp"]).size().reset_index(name="count"))
            trend["year_str"] = trend["year"].astype(str)
            # year_str の並び順を明示（文字列軸はアルファベット順になるため積み上げが崩れる対策）
            year_order = [str(y) for y in sorted(trend["year"].unique())]
            order = top_list + ["Others"]
            trend["TA_grp"] = pd.Categorical(trend["TA_grp"], categories=order, ordered=True)
            trend = trend.sort_values(["year", "TA_grp"])
            palette = px.colors.qualitative.Plotly
            color_map = {j: palette[i % len(palette)] for i, j in enumerate(top_list)}
            color_map["Others"] = "#cccccc"
            fig_stack = px.bar(
                trend, x="year_str", y="count", color="TA_grp",
                barmode="stack",
                color_discrete_map=color_map,
                category_orders={"year_str": year_order, "TA_grp": order},
                labels={"count": "文献数", "year_str": "Year", "TA_grp": "Journal"},
                title=f"Top {TOP_STACK} Journal + Others 年別推移",
                height=500,
            )
            fig_stack.update_layout(
                bargap=0.15,
                xaxis=dict(tickangle=-45),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    # ── Tab4: Top Authors ────────────────────────────────────────
    with tab4:
        if "FAU" in df.columns:
            TOP_AU = st.slider("表示著者数", 10, 30, 20, key="top_au")
            all_au, first_au, last_au = [], [], []
            for authors in df["FAU"]:
                if isinstance(authors, list) and authors:
                    s = [shorten_name(a) for a in authors]
                    all_au.extend(s); first_au.append(s[0]); last_au.append(s[-1])
            co  = pd.DataFrame(Counter(all_au).most_common(TOP_AU),  columns=["Author", "Co-author数"])
            fst = pd.DataFrame(Counter(first_au).most_common(TOP_AU), columns=["Author", "1st Author数"])
            lst = pd.DataFrame(Counter(last_au).most_common(TOP_AU),  columns=["Author", "Last Author数"])

            # グラフ（上）
            c1, c2, c3 = st.columns(3)
            for col_st, data, xcol, clr, ttl in [
                (c1, co,  "Co-author数",  "#e74c3c", f"Co-author Top{TOP_AU}"),
                (c2, fst, "1st Author数", "#3498db", f"1st Author Top{TOP_AU}"),
                (c3, lst, "Last Author数","#9b59b6", f"Last Author Top{TOP_AU}"),
            ]:
                with col_st:
                    fig = px.bar(data, x=xcol, y="Author", orientation="h",
                                 color_discrete_sequence=[clr], title=ttl, height=500)
                    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                    st.plotly_chart(fig, use_container_width=True)

            # 統合テーブル（下）
            st.markdown("#### 著者ランキング統合テーブル")
            merged = (co.merge(fst, on="Author", how="outer")
                        .merge(lst, on="Author", how="outer")
                        .fillna(0)
                        .sort_values("Co-author数", ascending=False)
                        .head(TOP_AU).reset_index(drop=True))
            st.dataframe(merged, use_container_width=True)

    # ── Tab5: 著者バブル ────────────────────────────────────────
    with tab5:
        if "FAU" in df.columns and "year" in df.columns:
            TOP_BB = st.slider("表示著者数", 10, 30, 20, key="bb_n")
            rows = [{"year": r["year"], "author": shorten_name(a)}
                    for _, r in df.iterrows() for a in (r.get("FAU") or [])]
            auyr = pd.DataFrame(rows).dropna()
            auyr["year"] = auyr["year"].astype(int)
            top_authors = auyr["author"].value_counts().head(TOP_BB).index.tolist()
            bbl = (auyr[auyr["author"].isin(top_authors)]
                   .groupby(["year", "author"]).size().reset_index(name="当年文献数"))
            bbl = bbl.sort_values(["author", "year"])
            bbl["累積文献数"] = bbl.groupby("author")["当年文献数"].cumsum()
            bbl["hover"] = (bbl["author"] + "<br>当年: " + bbl["当年文献数"].astype(str)
                            + " 件<br>累積: " + bbl["累積文献数"].astype(str) + " 件")
            fig = px.scatter(
                bbl, x="year", y="累積文献数", size="当年文献数", color="author",
                hover_name="author", custom_data=["hover"], size_max=35, height=620,
                labels={"year": "Year", "累積文献数": "累積文献数", "当年文献数": "当年文献数"},
                title=f"著者別年次推移（Top {TOP_BB}）",
            )
            fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
            fig.update_layout(legend=dict(font=dict(size=9)),
                              yaxis_title="累積文献数（件）", xaxis_title="Year")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 自然言語処理（Bigram主体）
# ═══════════════════════════════════════════════════════════════
elif page == "🔤 自然言語処理":
    st.markdown("## 自然言語処理（TF-IDF 特徴語抽出）")
    st.caption("Bigramを主体としています。各グラフは値の高い順に上から並びます。")
    if df is None:
        st.warning("先にデータをアップロードしてください。")
        st.stop()
    if "AB" not in df.columns:
        st.error("AB（Abstract）列が見つかりません。")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📰 Journal別特徴語", "👤 著者別特徴語", "📅 年度別特徴語"])

    # ── Tab1: Journal別 ─────────────────────────────────────────
    with tab1:
        if "TA" in df.columns:
            top_j_n = st.slider("対象Journal数", 3, 10, 6, key="nlp_j_n")
            top_w   = st.slider("表示フレーズ数", 5, 20, 10, key="nlp_j_w")
            jnlp = df.dropna(subset=["TA", "AB"]).copy()
            top6j = jnlp["TA"].value_counts().head(top_j_n).index.tolist()
            j_texts = {j: " ".join(jnlp[jnlp["TA"] == j]["AB"].tolist()) for j in top6j}
            st.markdown("#### Bigram（主）")
            tfidf_chart(j_texts, top_w, "bigram", "Journal別 特徴語 (bigram / TF-IDF)")
            with st.expander("単語（参考）"):
                tfidf_chart(j_texts, top_w, "word", "Journal別 特徴語 (word / TF-IDF)")

    # ── Tab2: 著者別 ─────────────────────────────────────────────
    with tab2:
        if "FAU" in df.columns:
            top_a_n = st.slider("対象著者数", 3, 10, 6, key="nlp_a_n")
            top_w2  = st.slider("表示フレーズ数", 5, 20, 10, key="nlp_a_w")
            au_rows = [{"author": shorten_name(r["FAU"][0]), "AB": r["AB"]}
                       for _, r in df.iterrows()
                       if isinstance(r.get("FAU"), list) and r["FAU"] and pd.notna(r.get("AB"))]
            au_df2 = pd.DataFrame(au_rows)
            if not au_df2.empty:
                top6a = au_df2["author"].value_counts().head(top_a_n).index.tolist()
                a_texts = {a: " ".join(au_df2[au_df2["author"] == a]["AB"].tolist()) for a in top6a}
                st.markdown("#### Bigram（主）")
                tfidf_chart(a_texts, top_w2, "bigram", "著者別 特徴語 (bigram / TF-IDF)")
                with st.expander("単語（参考）"):
                    tfidf_chart(a_texts, top_w2, "word", "著者別 特徴語 (word / TF-IDF)")

    # ── Tab3: 年度別 ─────────────────────────────────────────────
    with tab3:
        if "year" in df.columns:
            SPAN   = st.slider("集約年数（1期間）", 1, 5, 3, key="nlp_span")
            top_w3 = st.slider("表示フレーズ数", 5, 20, 10, key="nlp_y_w")
            ynlp = df.dropna(subset=["year", "AB"]).copy()
            ynlp["year"] = ynlp["year"].astype(int)
            ym = ynlp["year"].max()
            periods = {}
            for i in range(4):
                y_end = ym - i * SPAN; y_start = y_end - SPAN + 1
                label = f"{y_end}〜{y_start}" if SPAN > 1 else str(y_end)
                subset = ynlp[ynlp["year"].isin(range(y_start, y_end + 1))]["AB"].tolist()
                if subset:
                    periods[label] = " ".join(subset)
            st.markdown("#### Bigram（主）")
            tfidf_chart(periods, top_w3, "bigram", "年度別 特徴語 (bigram / TF-IDF)")
            with st.expander("単語（参考）"):
                tfidf_chart(periods, top_w3, "word", "年度別 特徴語 (word / TF-IDF)")

# ═══════════════════════════════════════════════════════════════
# ネットワーク分析（共著 + 単語共起）
# ═══════════════════════════════════════════════════════════════
elif page == "🕸️ ネットワーク分析":
    st.markdown("## ネットワーク分析")
    if df is None:
        st.warning("先にデータをアップロードしてください。")
        st.stop()

    tab_co, tab_word = st.tabs(["👥 共著関係", "💬 単語共起 (bigram)"])

    # ════════════════════════════════════════════
    # 共著関係（3部構成）
    # ════════════════════════════════════════════
    with tab_co:
        if "FAU" not in df.columns:
            st.error("FAU列が見つかりません。")
        else:
            cl, cr = st.columns([1, 3])
            with cl:
                top_n_co   = st.slider("対象著者数", 20, 100, 50, key="co_topn")
                cent_co    = st.selectbox("中心性指標", ["pagerank", "betweenness"], key="co_cent")
                show_lbl   = st.checkbox("著者名を表示", value=True, key="co_lbl")
                top_comm_n = st.slider("コミュニティ表示数", 3, 8, 5, key="co_comm_n")
                top_feat_n = st.slider("特徴語表示数", 5, 15, 8, key="co_feat_n")
                span_net   = st.slider("期間比較（年数）", 2, 5, 3, key="co_span")
                run_co     = st.button("🔍 分析実行", key="co_run")

            if run_co:
                with st.spinner("共著ネットワークを構築中..."):
                    G_co = build_coauthor_graph(df["FAU"], top_n=top_n_co)

                # ── ① 全期間ネットワーク ──────────────────────────
                with cr:
                    fig_co = nx_to_plotly(
                        G_co, centrality=cent_co,
                        title=f"共著ネットワーク（Top{top_n_co}・{cent_co}）",
                        directed=False, show_labels=show_lbl,
                    )
                    st.plotly_chart(fig_co, use_container_width=True)

                # ── 中心性スコア上位 ───────────────────────────────
                cent_scores = (nx.pagerank(G_co, weight="weight") if cent_co == "pagerank"
                               else nx.betweenness_centrality(G_co, weight="weight"))
                top_cent_df = pd.DataFrame(
                    sorted(cent_scores.items(), key=lambda x: x[1], reverse=True)[:15],
                    columns=["著者", f"{cent_co}スコア"]
                )
                st.markdown("#### 中心性スコア上位15名")
                st.dataframe(top_cent_df, use_container_width=True)

                # ── ② コミュニティ別特徴語 ──────────────────────────
                st.markdown("---")
                st.markdown("#### コミュニティ別 特徴語（bigram）")
                if "AB" in df.columns:
                    from networkx.algorithms.community import greedy_modularity_communities
                    comms = list(greedy_modularity_communities(G_co, weight="weight"))
                    comms = sorted([c for c in comms if len(c) >= 3], key=len, reverse=True)
                    st.caption(f"検出コミュニティ数（3名以上）: {len(comms)}")

                    comm_texts = {}
                    comm_info_rows = []
                    for i, comm in enumerate(comms[:top_comm_n]):
                        label = f"Comm {i+1}（{len(comm)}名）"
                        txt   = get_community_texts(comm, df)
                        if txt.strip():
                            comm_texts[label] = txt
                        top3 = [a for a, _ in sorted(
                            cent_scores.items(), key=lambda x: x[1], reverse=True
                        ) if a in comm][:3]
                        comm_info_rows.append({
                            "コミュニティ": label,
                            "人数": len(comm),
                            "主要著者（中心性上位3名）": ", ".join(top3),
                        })
                    st.dataframe(pd.DataFrame(comm_info_rows), use_container_width=True)

                    if comm_texts:
                        tfidf_chart(comm_texts, top_feat_n, "bigram",
                                    f"コミュニティ別 特徴語（Top{top_feat_n} bigram）")
                else:
                    st.info("AB列がないためコミュニティ特徴語は表示できません。")

                # ── ③ 期間比較 ──────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 期間比較ネットワーク")
                y_max_net = int(df["year"].dropna().max())
                recent_y  = list(range(y_max_net - span_net + 1, y_max_net + 1))
                prev_y    = list(range(y_max_net - span_net * 2 + 1, y_max_net - span_net + 1))
                st.caption(f"直近{span_net}年: {recent_y[0]}〜{recent_y[-1]}　／　前{span_net}年: {prev_y[0]}〜{prev_y[-1]}")

                G_rec  = build_coauthor_graph(df[df["year"].isin(recent_y)]["FAU"], top_n=top_n_co)
                G_prev = build_coauthor_graph(df[df["year"].isin(prev_y)]["FAU"],   top_n=top_n_co)

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.plotly_chart(
                        nx_to_plotly(G_rec, centrality=cent_co,
                                     title=f"直近{span_net}年（{recent_y[0]}〜{recent_y[-1]}）",
                                     show_labels=show_lbl),
                        use_container_width=True,
                    )
                with cc2:
                    st.plotly_chart(
                        nx_to_plotly(G_prev, centrality=cent_co,
                                     title=f"前{span_net}年（{prev_y[0]}〜{prev_y[-1]}）",
                                     show_labels=show_lbl),
                        use_container_width=True,
                    )

                # ── 新規共著ペア ────────────────────────────────────
                prev_edge_set = set(frozenset(e) for e in G_prev.edges())
                new_edge_rows = [
                    {"著者A": u, "著者B": v, "共著件数": int(data.get("weight", 1))}
                    for u, v, data in G_rec.edges(data=True)
                    if frozenset((u, v)) not in prev_edge_set
                ]
                if new_edge_rows:
                    new_df = (pd.DataFrame(new_edge_rows)
                              .sort_values("共著件数", ascending=False)
                              .reset_index(drop=True))
                    st.markdown(f"#### 新規共著ペア（直近{span_net}年に初登場）: {len(new_df)} ペア")
                    st.dataframe(new_df.head(30), use_container_width=True)
                else:
                    st.info("新規共著ペアなし。")

    # ════════════════════════════════════════════
    # 単語共起（bigram・3期間比較）
    # ════════════════════════════════════════════
    with tab_word:
        if "AB" not in df.columns:
            st.error("AB列が見つかりません。")
        else:
            cl2, cr2 = st.columns([1, 3])
            with cl2:
                period_span = st.slider("1期間の年数", 1, 5, 1, key="word_span")
                top_edges   = st.slider("共起ペア上限", 40, 120, 80, key="word_edges")
                cent_w      = st.selectbox("中心性指標", ["pagerank", "betweenness"], key="w_cent")
                run_w       = st.button("🔍 分析実行", key="w_run")

            if run_w:
                y_max_w = int(df["year"].dropna().max())
                w_periods = []
                for i in range(3):
                    y_end   = y_max_w - i * period_span
                    y_start = y_end - period_span + 1
                    label   = f"{y_end}年" if period_span == 1 else f"{y_start}〜{y_end}年"
                    w_periods.append((label, list(range(y_start, y_end + 1))))

                compare_rows = []
                for label, years in w_periods:
                    texts_w = df[df["year"].isin(years)]["AB"].dropna().tolist()
                    with cr2:
                        st.markdown(f"##### {label}　（Abstract {len(texts_w)} 件）")
                    if not texts_w:
                        with cr2:
                            st.warning("この期間のデータがありません。")
                        continue
                    with st.spinner(f"{label} 単語共起ネットワーク構築中..."):
                        G_w = build_bigram_graph(texts_w, top_n=top_edges)
                        fig_w = nx_to_plotly(
                            G_w, centrality=cent_w,
                            title=f"単語共起ネットワーク [bigram]　{label}",
                            directed=True, show_labels=True,
                        )
                    with cr2:
                        st.plotly_chart(fig_w, use_container_width=True)

                    # 中心語収集
                    if G_w.nodes:
                        scores = (nx.pagerank(G_w, weight="weight") if cent_w == "pagerank"
                                  else nx.betweenness_centrality(G_w, weight="weight"))
                        for rank, (word, sc) in enumerate(
                            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10], 1
                        ):
                            succs = [(v, G_w[word][v].get("weight", 0))
                                     for v in G_w.successors(word)]
                            display_term = (f"{word} {max(succs, key=lambda x: x[1])[0]}"
                                            if succs else word)
                            compare_rows.append({"期間": label, "rank": rank, "bigram": display_term})

                # 比較テーブル
                if compare_rows:
                    comp_df = (pd.DataFrame(compare_rows)
                               .pivot(index="rank", columns="期間", values="bigram")
                               .reindex(columns=[lbl for lbl, _ in w_periods]))
                    comp_df.index.name = "順位"
                    with cr2:
                        st.markdown("#### 3期間 中心bigram 比較")
                        st.dataframe(comp_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# MeSH 分析
# ═══════════════════════════════════════════════════════════════
elif page == "🧬 MeSH分析":
    st.markdown("## 🧬 MeSH ターム分析")
    if df is None:
        st.warning("先に「📂 Data Upload」でデータをアップロードしてください。")
        st.stop()
    if "MH" not in df.columns:
        st.error("MH（MeSH）列が見つかりません。")
        st.stop()

    mh_count = df["MH"].apply(lambda x: len(x) > 0).sum()
    st.caption(f"MeSHデータあり文献: {mh_count:,} 件 / {len(df):,} 件")

    tab1, tab2 = st.tabs(["🌡️ ヒートマップ", "💥 バースト検知"])

    with st.sidebar:
        st.markdown("**MeSH 分析設定**")
        top_mesh_n  = st.slider("ヒートマップ表示数", 20, 100, 60, key="mesh_top")
        min_year_m  = st.slider("表示開始年",
                                int(df["year"].dropna().min()),
                                int(df["year"].dropna().max()),
                                max(int(df["year"].dropna().min()),
                                    int(df["year"].dropna().max()) - 20),
                                key="mesh_miny")
        burst_top_n = st.slider("バースト対象MeSH数", 10, 60, 30, key="burst_top")
        burst_z_th  = st.slider("バースト Zスコア閾値", 0.5, 3.0, 1.5, 0.1, key="burst_z")

    with st.spinner("MeSHデータを集計中..."):
        pivot_norm, mdf_all = build_mesh_pivot(df, top_mesh_n, min_year_m)

    if pivot_norm.empty:
        st.warning("条件に一致するMeSHデータがありません。")
        st.stop()

    with tab1:
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_norm.T.values,
            x=[str(y) for y in pivot_norm.index],
            y=pivot_norm.columns.tolist(),
            colorscale=[
                [0.0, "#eaf4fb"], [0.2, "#aad4f0"],
                [0.5, "#f5c842"], [0.75, "#f07b2a"], [1.0, "#c0392b"],
            ],
            colorbar=dict(title="出現率 (%)", thickness=14),
            hoverongaps=False,
            hovertemplate="年: %{x}<br>MeSH: %{y}<br>出現率: %{z:.2f}%<extra></extra>",
        ))
        fig_heat.update_layout(
            title=f"MeSHターム 年別出現頻度ヒートマップ Top{top_mesh_n}（{min_year_m}年〜）",
            xaxis=dict(title="Year", tickangle=-45),
            yaxis=dict(title="", autorange="reversed", tickfont=dict(size=10)),
            height=max(500, top_mesh_n * 16),
            margin=dict(l=300, r=60, t=60, b=60),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.download_button("⬇️ ヒートマップデータ (CSV)",
                           pivot_norm.T.reset_index().to_csv(index=False).encode(),
                           "mesh_heatmap.csv", "text/csv")

    with tab2:
        with st.spinner("バースト検知中..."):
            burst_df = detect_bursts(mdf_all, burst_top_n, min_year_m, burst_z_th)
        if burst_df.empty:
            st.info("バーストが検出されませんでした。Zスコア閾値を小さくしてみてください。")
        else:
            st.success(f"検出バースト数: {len(burst_df)} 件")
            st.dataframe(burst_df, use_container_width=True)
            fig_burst = go.Figure()
            max_z = burst_df["最大Zスコア"].max()
            cmap  = px.colors.sequential.YlOrRd
            for _, brow in burst_df.iterrows():
                ci = min(int(brow["最大Zスコア"] / max_z * (len(cmap) - 1)), len(cmap) - 1)
                fig_burst.add_trace(go.Bar(
                    x=[brow["期間"]], y=[brow["MeSH"]], orientation="h",
                    base=brow["バースト開始"],
                    marker=dict(color=cmap[ci], line=dict(width=0.5, color="white")),
                    text=f"ピーク: {brow['ピーク年']}年 ({brow['ピーク件数']}件)",
                    textposition="inside",
                    hovertemplate=(
                        f"<b>{brow['MeSH']}</b><br>"
                        f"期間: {brow['バースト開始']}〜{brow['バースト終了']}年<br>"
                        f"ピーク: {brow['ピーク年']}年 / {brow['ピーク件数']}件<br>"
                        f"最大Zスコア: {brow['最大Zスコア']}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            fig_burst.add_trace(go.Scatter(
                x=burst_df["ピーク年"], y=burst_df["MeSH"], mode="markers",
                marker=dict(symbol="diamond", size=10, color="#2c3e50",
                            line=dict(width=1, color="white")),
                hovertemplate="ピーク年: %{x}<extra></extra>", name="ピーク年",
            ))
            fig_burst.update_layout(
                title=f"MeSH バースト検知タイムライン（Zスコア閾値={burst_z_th}）",
                xaxis=dict(title="Year",
                           range=[min_year_m - 1, int(mdf_all["year"].max()) + 1], dtick=2),
                yaxis=dict(title="", autorange="reversed", tickfont=dict(size=11)),
                barmode="overlay",
                height=max(420, len(burst_df) * 28 + 100),
                margin=dict(l=300, r=60, t=60, b=60),
                legend=dict(x=1.01, y=1),
            )
            st.plotly_chart(fig_burst, use_container_width=True)
            st.download_button("⬇️ バースト検知結果 (CSV)",
                               burst_df.to_csv(index=False).encode(),
                               "mesh_burst.csv", "text/csv")
