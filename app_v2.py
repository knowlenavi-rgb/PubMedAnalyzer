import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import copy
from collections import Counter
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import base64

# ─── NLTK ─────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk()

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="文献Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── デザイントークン ────────────────────────────────────────────
# 配色: ネイビー（信頼感）× ティール（先進性のアクセント）
INK      = "#16202c"   # ヘッダー・サイドバーのベース
INK_SOFT = "#27384a"
TEAL     = "#19b3a6"   # アクセント
TEAL_DK  = "#0e8a80"
SLATE    = "#5a6b7d"   # 補助テキスト
BG       = "#f7f9fb"   # ページ背景

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Noto Sans JP', -apple-system, sans-serif;
    }}

    .stApp {{ background: {BG}; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1320px; }}

    /* ── 見出し ──────────────────────────────────────────── */
    h1 {{
        color: {INK}; font-weight: 800; letter-spacing: -0.02em;
        line-height: 1.35; padding-top: 0.15em; margin-top: 0;
    }}
    h2 {{
        color: {INK}; font-weight: 700; letter-spacing: -0.01em;
        border-bottom: 2px solid {TEAL}; padding-bottom: 0.4rem;
        line-height: 1.4; padding-top: 0.1em; margin-top: 0.2rem;
    }}
    h3, h4 {{ color: {INK_SOFT}; font-weight: 600; line-height: 1.4; }}

    /* ── サイドバー ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {INK} 0%, {INK_SOFT} 100%);
    }}
    section[data-testid="stSidebar"] * {{ color: #e7ecf1 !important; }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.95rem;
    }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {{
        background: {TEAL} !important;
    }}
    .sidebar-logo-box {{
        display:flex; align-items:center; gap:10px;
        padding: 6px 4px 18px 4px; margin-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }}
    .sidebar-logo-box img {{
        max-height: 42px; max-width: 100%; border-radius: 6px;
        background: rgba(255,255,255,0.92); padding: 4px 8px;
    }}
    .sidebar-brand {{
        font-weight: 800; font-size: 1.05rem; color: #fff !important;
        letter-spacing: -0.01em;
    }}

    /* ── タブ ─────────────────────────────────────────────── */
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] {{
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        color: {SLATE};
    }}
    div[data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 2px solid #e3e8ee;
    }}
    div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {{
        border-bottom: 3px solid {TEAL} !important;
        color: {TEAL_DK} !important;
    }}

    /* ── ボタン ───────────────────────────────────────────── */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 8px; font-weight: 600;
        border: 1px solid #d8dfe6;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {TEAL}; color: {TEAL_DK};
    }}

    div[data-testid="stForm"] {{ max-width: 400px; margin: auto; }}

    /* ── KPI カード ───────────────────────────────────────── */
    .kpi-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }}
    .kpi-card {{
        background:#ffffff; border-radius:12px; padding:14px 20px;
        min-width:140px; text-align:center; flex:1;
        border-top:4px solid {TEAL};
        box-shadow: 0 1px 3px rgba(22,32,44,0.06), 0 1px 2px rgba(22,32,44,0.04);
    }}
    .kpi-card.green  {{ border-top-color:#2ecc71; }}
    .kpi-card.orange {{ border-top-color:#f39c12; }}
    .kpi-card.purple {{ border-top-color:#9b59b6; }}
    .kpi-label {{ font-size:0.78rem; color:{SLATE}; margin-bottom:4px; font-weight: 500; }}
    .kpi-value {{ font-size:1.9rem; font-weight:800; color:{INK}; line-height:1; }}
    .kpi-unit  {{ font-size:0.72rem; color:#aaa; margin-top:2px; }}

    /* ── データフレーム・エクスパンダー ──────────────────── */
    div[data-testid="stExpander"] {{
        border: 1px solid #e3e8ee; border-radius: 10px;
    }}
    /* 列数の多い表・長文を含む表は横スクロールで全体を確認できるようにする */
    div[data-testid="stDataFrame"] {{
        max-width: 100%;
        overflow-x: auto !important;
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 認証（デモ版のため省略）
# ═══════════════════════════════════════════════════════════════
# 本番版（app.py）では Streamlit Cloud Secrets / config.yaml を使った
# ログイン認証がありますが、このデモ版では誰でもすぐに試せるよう
# 認証をすべて省略し、サンプルデータを自動読み込みする構成にしています。


# ═══════════════════════════════════════════════════════════════
# MEDLINE パーサー
# ═══════════════════════════════════════════════════════════════
def parse_medline(text: str) -> pd.DataFrame:
    # 改行コードを LF に統一してから処理する。
    # （Windows由来の \r\n ファイルだと継続行の正規表現にマッチせず、
    #   AD/TI/AB などの長いフィールドが行の途中で切れてしまうバグの対策）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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

# 医学論文の本文に頻出するが、トピックの特徴を表さない定型語
# （研究記述の決まり文句・統計表現・著者ノイズなど）を追加で除外する
MEDICAL_STOPWORDS = set("""
patient patients case cases study studies report reported reports background
objective objectives purpose aim aims method methods methodology result results
conclusion conclusions discussion analysis analyses finding findings data
significant significance versus compared comparison clinical present presented
investigate investigated investigation describe described article articles
review reviewed literature paper papers author authors group groups year
years old age aged month months week weeks day days follow followed following
based associated associate including include included may also however
therefore thus could would might one two three first second third using
use used new previously previous total mean median number respectively
copyright rights reserved elsevier published publication doi pmid anti vs
""".split())

STOP_EN = STOP_EN | MEDICAL_STOPWORDS

def tokenize(text: str):
    # ハイフンでつながった複合語（anti-nmdar 等）は1単語として保持する。
    # 医学論文では "anti-NMDAR", "N-methyl-D-aspartate" のような
    # ハイフン連結の専門用語が多く、分割すると意味のない断片になるため。
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())
    return [w for w in words if w not in STOP_EN and len(w) > 2 and not w.isdigit()]

def make_bigrams(text: str):
    # 重要: 先にストップワードを除去してから隣接ペアを作ってはいけない。
    # 除去後に偶然隣り合っただけの無関係な単語同士（例: "disease is a common"
    # → "is a" 除去後に disease/common が隣接）がペアになり、ノイズの原因になる。
    # 元の文中で実際に隣接していた単語ペア（コロケーション）だけを残すため、
    # 「生の単語列からまず隣接ペアを作り、両方ともストップワードでないペアのみ残す」
    # という順序にする（R/tidytext版の bind_tf_idf と同じアプローチ）。
    raw_words = re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())
    bigrams = []
    for i in range(len(raw_words) - 1):
        w1, w2 = raw_words[i], raw_words[i + 1]
        if (w1 in STOP_EN or w2 in STOP_EN
                or len(w1) <= 2 or len(w2) <= 2
                or w1.isdigit() or w2.isdigit()
                or w1 == w2):  # 同一単語の繰り返しは除外
            continue
        # アンダースコア結合で TfidfVectorizer がトークン分割しないようにする
        bigrams.append(f"{w1}_{w2}")
    return bigrams

def pretokenize(text: str):
    # MeSHタームのように、すでに「アンダースコア結合された語句」として
    # 意味のある単位に分割済みのテキストを、それ以上分解せずそのまま返す。
    # （tokenize() はハイフンのみ保持しアンダースコアを分割してしまうため、
    #   "Multiple_Sclerosis" のような複合語専用にこちらを使う）
    return [w for w in text.split() if w]

def tfidf_top(group_texts: dict, top_n: int, mode: str = "bigram") -> pd.DataFrame:
    """
    tf-idf の計算式は R の tidytext::bind_tf_idf と同じ定義に合わせている:
      tf  = そのグループ内でのその語の出現回数 ÷ そのグループの総語数（相対頻度）
      idf = ln(全グループ数 ÷ その語が出現するグループ数)（平滑化なし）
      tf_idf = tf × idf

    scikit-learn の TfidfVectorizer のデフォルト（tf=生カウント、
    idf に +1 のスムージングを加える、結果をL2正規化する）を使うと、
    全グループに共通して頻出する語のidfが下がりきらず、結果としてtf（頻度）の
    影響が支配的になり、「そのグループに特有の語」が埋もれてしまう問題があった。
    そのため、グループ間の違いがより際立つ tidytext 互換の式に統一している。
    """
    groups = list(group_texts.keys())
    func   = {"word": tokenize, "bigram": make_bigrams, "pretoken": pretokenize}[mode]

    group_term_counts = {g: Counter(func(group_texts[g])) for g in groups}
    if not any(group_term_counts.values()):
        return pd.DataFrame(columns=["group", "term", "tfidf"])

    all_terms = set()
    for c in group_term_counts.values():
        all_terms.update(c.keys())

    n_groups = len(groups)
    doc_freq = {
        t: sum(1 for g in groups if group_term_counts[g].get(t, 0) > 0)
        for t in all_terms
    }

    rows = []
    for g in groups:
        counts = group_term_counts[g]
        total_words = sum(counts.values())
        if total_words == 0:
            continue
        scores = []
        for t, cnt in counts.items():
            tf = cnt / total_words
            idf = math.log(n_groups / doc_freq[t]) if doc_freq[t] > 0 else 0.0
            scores.append((t, tf * idf))
        scores.sort(key=lambda x: x[1], reverse=True)
        for t, score in scores[:top_n]:
            if score <= 0:
                continue
            display_term = t.replace("_", " ") if mode in ("bigram", "pretoken") else t
            rows.append({"group": g, "term": display_term, "tfidf": score})
    return pd.DataFrame(rows)



def kwic_extract_single(doc_text: str, term_words: list, window: int = 7) -> str:
    """
    1つの文書から指定フレーズの最初の出現コンテキストを
    「...前文脈 [フレーズ] 後文脈...」の文字列で返す。ホバー表示用。
    """
    raw = re.findall(r"[A-Za-z][A-Za-z\-]*|[.,;:]", doc_text)
    lower = [w.lower() for w in raw]
    n = len(term_words)
    for i in range(len(lower) - n + 1):
        if lower[i:i + n] == term_words:
            start = max(0, i - window)
            end   = min(len(raw), i + n + window)
            before  = " ".join(raw[start:i])
            matched = " ".join(raw[i:i + n])
            after   = " ".join(raw[i + n:end])
            return f"…{before} <b>{matched}</b> {after}…"
    return ""


def build_kwic_map(group_docs: dict, top_n: int, max_hits: int = 2) -> dict:
    """
    TF-IDF上位語（tfidf_top()と同じ計算）を対象にKWICを抽出する。
    旧版は「頻度Top(top_n×2)」でKWICを集めていたが、TF-IDF上位語は
    「そのグループに特徴的だが全体頻度は低い語」が多いため
    頻度ランキングでは漏れが71.7%に達していた。
    修正後は tfidf_top() で確定した上位語のみを対象にする。
    """
    # まず全グループのTF-IDF上位語を確定させる
    group_texts_joined = {g: " ".join(docs) for g, docs in group_docs.items()}
    tfidf_result = tfidf_top(group_texts_joined, top_n, mode="bigram")

    kwic_map = {}
    for g, docs in group_docs.items():
        kwic_map[g] = {}
        # このグループのTF-IDF上位語を取得
        group_terms = tfidf_result[tfidf_result["group"] == g]["term"].tolist()
        for term_display in group_terms:
            term_words = term_display.lower().split()
            hits = []
            for doc in docs:
                if not isinstance(doc, str):
                    continue
                ctx = kwic_extract_single(doc, term_words)
                if ctx:
                    hits.append(ctx)
                if len(hits) >= max_hits:
                    break
            kwic_map[g][term_display] = hits
    return kwic_map


def tfidf_chart(group_texts: dict, top_n: int, mode: str, title: str, wrap: int = None,
                 group_order: list = None, kwic_map: dict = None):
    result = tfidf_top(group_texts, top_n, mode)
    if result.empty:
        st.info("データが不足しています。")
        return
    result = result.sort_values("tfidf", ascending=True)
    n_groups = result["group"].nunique()
    facet_wrap = wrap if wrap is not None else min(n_groups, 3)
    n_rows = math.ceil(n_groups / facet_wrap)
    max_rows_per_group = result.groupby("group").size().max()
    row_h = 22
    category_orders = {}
    if group_order:
        present = set(result["group"].unique())
        ordered = [g for g in group_order if g in present]
        chunks = [ordered[i:i + facet_wrap] for i in range(0, len(ordered), facet_wrap)]
        category_orders["group"] = [g for chunk in reversed(chunks) for g in chunk]

    # KWIC が渡されている場合はホバーテキストに文脈を追加する
    if kwic_map and mode == "bigram":
        def _kwic_hover(row):
            hits = (kwic_map.get(row["group"], {}).get(row["term"], []))
            ctx  = "<br>".join(hits[:2]) if hits else "（文脈例なし）"
            return f"<b>{row['term']}</b><br>TF-IDF: {row['tfidf']:.4f}<br><br>📖 文脈:<br>{ctx}"
        result = result.copy()
        result["kwic_hover"] = result.apply(_kwic_hover, axis=1)
        fig = px.bar(
            result, x="tfidf", y="term", color="group",
            facet_col="group", facet_col_wrap=facet_wrap, orientation="h",
            labels={"tfidf": "TF-IDF", "term": ""}, title=title,
            height=max(400, max_rows_per_group * row_h * n_rows + n_rows * 40),
            category_orders=category_orders or None,
            custom_data=["kwic_hover"],
        )
        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    else:
        fig = px.bar(
            result, x="tfidf", y="term", color="group",
            facet_col="group", facet_col_wrap=facet_wrap, orientation="h",
            labels={"tfidf": "TF-IDF", "term": ""}, title=title,
            height=max(400, max_rows_per_group * row_h * n_rows + n_rows * 40),
            category_orders=category_orders or None,
        )

    fig.update_yaxes(
        matches=None, showticklabels=True,
        categoryorder="total ascending",
        tickfont=dict(size=11),
    )
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_layout(
        showlegend=False,
        title_font=dict(size=14),
        margin=dict(t=60, b=20),
    )
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1],
        font=dict(size=13, color="#2c3e50"),
    ))
    st.plotly_chart(fig, use_container_width=True)


def feature_word_panel(group_docs: dict, top_n: int, mode: str, title: str,
                        key_prefix: str, group_order: list = None, wrap: int = None,
                        title_lookup: dict = None):
    """
    特徴語のTF-IDFグラフを表示する。
    bigram モードのときは KWIC 文脈をホバーに埋め込む。
    """
    group_texts_joined = {g: " ".join(docs) for g, docs in group_docs.items()}
    # bigram モードのみ KWIC 事前抽出（他モードは不要）
    kwic_data = build_kwic_map(group_docs, top_n) if mode == "bigram" else None
    tfidf_chart(group_texts_joined, top_n, mode, title,
                wrap=wrap, group_order=group_order, kwic_map=kwic_data)


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
                 directed: bool = False, show_labels: bool = True,
                 fullname_map: dict = None, affil_map: dict = None,
                 country_map: dict = None):
    if not G.nodes:
        return go.Figure()
    fullname_map = fullname_map or {}
    affil_map = affil_map or {}
    country_map = country_map or {}
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

    def _hover(n):
        full = fullname_map.get(n, n)
        affil = affil_map.get(n, "（所属情報なし）")
        country = country_map.get(n, "")
        country_str = f"　🌏 {country}" if country and country != "Unknown" else ""
        return f"<b>{full}</b><br>所属: {affil}{country_str}"

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
            hovertext=[_hover(n) for n in G.nodes],
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

def nx_to_plotly_diff(G, centrality: str = "pagerank", title: str = "",
                       show_labels: bool = True,
                       highlight_nodes: set = None, highlight_edges: set = None,
                       fullname_map: dict = None, affil_map: dict = None):
    """期間比較ネットワーク用。ノードはコミュニティ色で塗り、
    新規ノード・新規エッジはオレンジの太枠／太線で強調表示する。"""
    highlight_nodes = highlight_nodes or set()
    highlight_edges = highlight_edges or set()
    fullname_map = fullname_map or {}
    affil_map = affil_map or {}
    if not G.nodes:
        return go.Figure()
    np.random.seed(42)
    pos = nx.spring_layout(G, seed=42, k=1.2 / math.sqrt(max(len(G.nodes), 1)))
    cent = (nx.pagerank(G, weight="weight") if centrality == "pagerank"
            else nx.betweenness_centrality(G, weight="weight"))
    max_cent = max(cent.values()) if cent else 1

    # コミュニティ検出（全期間ネットワークと同じロジック）
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        node_community = {}
        for idx, com in enumerate(greedy_modularity_communities(G, weight="weight")):
            for node in com:
                node_community[node] = idx
    except Exception:
        node_community = {n: 0 for n in G.nodes}
    palette = px.colors.qualitative.Plotly

    NEW_OUTLINE = "#f39c12"     # 新規ノードの強調枠線（オレンジ）
    NEW_EDGE_COLOR = "#f39c12"  # 新規エッジ（オレンジ太線）
    BASE_EDGE_COLOR = "#ccc"

    # エッジを「新規」「既存」に分けてそれぞれ描画（凡例に出すため）
    base_edge_x, base_edge_y = [], []
    new_edge_x, new_edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        if frozenset((u, v)) in highlight_edges:
            new_edge_x += [x0, x1, None]; new_edge_y += [y0, y1, None]
        else:
            base_edge_x += [x0, x1, None]; base_edge_y += [y0, y1, None]

    node_colors  = [palette[node_community.get(n, 0) % len(palette)] for n in G.nodes]
    node_line_c  = [NEW_OUTLINE if n in highlight_nodes else "white" for n in G.nodes]
    node_line_w  = [3 if n in highlight_nodes else 1 for n in G.nodes]
    node_sizes   = [max(8, cent.get(n, 0) / max_cent * 40) + (4 if n in highlight_nodes else 0)
                     for n in G.nodes]

    def _hover(n):
        full = fullname_map.get(n, n)
        affil = affil_map.get(n, "（所属情報なし）")
        new_badge = "<br>🆕 新規著者" if n in highlight_nodes else ""
        return f"<b>{full}</b><br>所属: {affil}{new_badge}"

    traces = [
        go.Scatter(x=base_edge_x, y=base_edge_y, mode="lines",
                   line=dict(width=0.7, color=BASE_EDGE_COLOR), hoverinfo="none",
                   name="既存の共著関係", showlegend=bool(base_edge_x)),
        go.Scatter(x=new_edge_x, y=new_edge_y, mode="lines",
                   line=dict(width=2.2, color=NEW_EDGE_COLOR), hoverinfo="none",
                   name="新規の共著関係", showlegend=bool(new_edge_x)),
        go.Scatter(
            x=[pos[n][0] for n in G.nodes],
            y=[pos[n][1] for n in G.nodes],
            mode="markers+text" if show_labels else "markers",
            text=list(G.nodes) if show_labels else [],
            textposition="top center", textfont=dict(size=9),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=node_line_w, color=node_line_c),
            ),
            hovertext=[_hover(n) for n in G.nodes],
            hoverinfo="text", name="著者", showlegend=False,
        ),
    ]
    fig = go.Figure(traces)
    fig.update_layout(
        title=title, title_x=0.5,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20), height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", font=dict(size=10)),
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
    return texts

# ── 機関名抽出：スコアリング方式（改良版） ──────────────────────
# 旧版は「最初にキーワードにマッチした断片」を返していたため
# "Brain Research Institute, Niigata University" のような文字列で
# 下位機関（研究所）が大学より先にマッチして誤抽出していた。
# 改良版では各断片に優先度スコアを付け、最も上位の機関種別を選ぶ。
_INST_PRIORITY = [
    (10, ["University", "Universität", "Université", "Universidad",
          "Universitat", "Università", "Universiteit", "Universitet"]),
    (8,  ["Hospital", "Hôpital", "Hôpitaux", "Krankenhaus", "Ospedale"]),
    (6,  ["Research Institute", "Klinikum", "Charité", "CHU",
          "Academy", "Foundation"]),
    (4,  ["Graduate School", "School of Medicine", "Medical School",
          "College of Medicine"]),
    (2,  ["Center", "Centre", "Clinic", "Clinique", "Laboratory", "Department"]),
]

def _inst_score(frag: str) -> int:
    for score, keywords in _INST_PRIORITY:
        for kw in keywords:
            if kw.lower() in frag.lower():
                return score
    return 0


def extract_top_institution(ad_text: str) -> str:
    """
    AD（所属）の生テキストから最上位の組織名（大学・病院など）を抽出する。
    カンマ区切りの各断片にスコアを付け（大学=10 > 病院=8 > 研究所=6 ...）、
    最高スコアの断片を返す。同スコアの場合は文章中の出現順（早い方）を優先。
    """
    ad_clean = re.sub(r"\S+@\S+", "", str(ad_text)).rstrip(".")
    ad_first = ad_clean.split(";")[0]
    fragments = [f.strip().rstrip(".") for f in ad_first.split(",") if f.strip()]
    if not fragments:
        return "―"
    scored = [(_inst_score(f), i, f) for i, f in enumerate(fragments)]
    best = sorted(scored, key=lambda x: (-x[0], x[1]))[0]
    if best[0] == 0:
        return fragments[-2] if len(fragments) >= 2 else fragments[0]
    return best[2]


# AD末尾から国名を抽出するためのキーワードリスト
_COUNTRY_LIST = [
    "United States", "USA", "U.S.A", "England", "UK", "United Kingdom",
    "Germany", "France", "Spain", "Italy", "Japan", "China", "Taiwan",
    "South Korea", "Korea", "Australia", "Canada", "Netherlands", "Switzerland",
    "India", "Brazil", "Sweden", "Norway", "Denmark", "Finland", "Belgium",
    "Austria", "Poland", "Portugal", "Israel", "Turkey", "Argentina", "Mexico",
    "Singapore", "New Zealand", "Ireland", "Greece", "Czech Republic", "Czechia",
    "Hungary", "Romania", "Serbia", "Chile", "Colombia", "Egypt", "Iran",
    "Saudi Arabia", "South Africa", "Thailand", "Malaysia", "Indonesia",
    "Hong Kong", "Scotland", "Wales", "United Arab Emirates", "Russia",
    "Pakistan", "Bangladesh", "Nepal", "Sri Lanka", "Vietnam", "Philippines",
    "Iraq", "Jordan", "Lebanon", "Kuwait", "Qatar", "Bahrain", "Oman",
]
# 長い名称が短い名称にマッチしないよう長い順に並べる
_COUNTRY_LIST_SORTED = sorted(_COUNTRY_LIST, key=len, reverse=True)


def extract_country(ad_text: str) -> str:
    """
    AD（所属）の生テキストから国名を抽出する。
    セミコロン区切りの最初の所属エントリのみを対象とし、
    メールアドレスを除去した後、後ろのカンマ区切り断片から順に国名キーワードをマッチする。
    見つからない場合は "Unknown" を返す。
    """
    # メールアドレスを除去してセミコロン区切りの最初のエントリのみを使う
    ad_clean = re.sub(r"\S+@\S+", "", str(ad_text)).rstrip(".")
    ad_first = ad_clean.split(";")[0]
    # 後ろのカンマ区切り断片から順に探す（国名は通常末尾付近にある）
    frags = [f.strip() for f in ad_first.split(",") if f.strip()]
    for frag in reversed(frags):
        for kw in _COUNTRY_LIST_SORTED:
            if kw.lower() in frag.lower():
                # 表記を正規化（USA→United States 等）
                if kw in ("USA", "U.S.A"):
                    return "United States"
                if kw in ("UK", "England", "Scotland", "Wales"):
                    return "United Kingdom"
                if kw == "South Korea":
                    return "Korea"
                return kw
    return "Unknown"


@st.cache_data
def build_author_country_map(df_src: pd.DataFrame) -> dict:
    """
    著者の短縮名 → 国名の対応辞書を作る。
    AD帰属ルールは build_author_affiliation_map と同じ（1対1 / 第一著者のみ）。
    """
    if "AD" not in df_src.columns:
        return {}
    country_counter: dict = {}
    for _, row in df_src.iterrows():
        fau = row.get("FAU", [])
        ad  = row.get("AD", [])
        if not isinstance(fau, list) or not fau:
            continue
        if not isinstance(ad, list):
            ad = [ad] if (ad and pd.notna(ad)) else []
        n_fau, n_ad = len(fau), len(ad)
        for i, a in enumerate(fau):
            if n_ad == n_fau and i < n_ad:
                ad_text = str(ad[i])
            elif n_ad == 1 and i == 0:
                ad_text = str(ad[0])
            else:
                continue
            country = extract_country(ad_text)
            if country == "Unknown":
                continue
            key = shorten_name(a)
            country_counter.setdefault(key, Counter())[country] += 1
    return {k: v.most_common(1)[0][0] for k, v in country_counter.items()}


@st.cache_data
def build_author_affiliation_map(df_src: pd.DataFrame) -> dict:
    """
    著者の短縮名 → 最上位組織名（大学・病院など）の対応辞書を作る。
    AD帰属ルール:
      FAU数=AD数 → i番目の著者にi番目のADを1対1で付与
      AD=1件のみ → 第一著者(FAU[0])にのみ付与、他著者はスキップ
      AD=0件    → 誰にも付与しない
    """
    if "AD" not in df_src.columns:
        return {}
    affil_counter: dict = {}
    for _, row in df_src.iterrows():
        fau = row.get("FAU", [])
        ad  = row.get("AD", [])
        if not isinstance(fau, list) or not fau:
            continue
        if not isinstance(ad, list):
            ad = [ad] if (ad and pd.notna(ad)) else []
        n_fau, n_ad = len(fau), len(ad)
        for i, a in enumerate(fau):
            if n_ad == n_fau and i < n_ad:
                ad_text = str(ad[i])
            elif n_ad == 1 and i == 0:
                ad_text = str(ad[0])
            else:
                continue
            top_inst = extract_top_institution(ad_text)
            if not top_inst or top_inst == "―":
                continue
            key = shorten_name(a)
            affil_counter.setdefault(key, Counter())[top_inst] += 1
    return {k: v.most_common(1)[0][0] for k, v in affil_counter.items()}


@st.cache_data
def build_author_fullname_map(df_src: pd.DataFrame) -> dict:
    """
    著者の短縮名 → フルネーム（FAU表記）の対応辞書を作る。
    同じ短縮名に複数のフルネームが対応する場合は最頻出のものを採用する。
    ネットワーク図のホバー表示で正式な著者名を見せるために使う。
    """
    if "FAU" not in df_src.columns:
        return {}
    name_counter: dict = {}
    for _, row in df_src.iterrows():
        fau = row.get("FAU", [])
        if not isinstance(fau, list):
            continue
        for a in fau:
            key = shorten_name(a)
            name_counter.setdefault(key, Counter())[str(a)] += 1
    return {k: v.most_common(1)[0][0] for k, v in name_counter.items()}

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
    """
    改良版バースト検知:
    - Zスコアはmdf内の全年範囲（データの最初年〜）で計算する
      （min_year以降で切り出すと、その期間だけで平均・分散が再計算されてしまい
        単調増加中の語が「ずっと平均より多い」状態になってバーストを検出できない）
    - バースト開始がmin_year以前のものは結果から除外する
      （ヒートマップの表示範囲に合わせるため）
    - min_total=5 以上の出現がある語のみ対象
    """
    if mdf.empty:
        return pd.DataFrame()
    y_min_data = int(mdf["year"].min())   # データの最初年（全期間でZを計算）
    y_max      = int(mdf["year"].max())
    burst_mesh = (mdf["mesh"].value_counts()
                  .loc[lambda s: s >= 5]
                  .head(top_n).index.tolist())
    results = []
    for term in burst_mesh:
        # 全年範囲でZスコアを計算
        series = (mdf[mdf["mesh"] == term].groupby("year").size()
                  .reindex(range(y_min_data, y_max + 1), fill_value=0))
        mean, std = series.mean(), series.std()
        if std == 0 or len(series) < 4:
            continue
        z = (series - mean) / std
        in_b, b_start = False, None
        for yr_val, zv in z.items():
            if zv >= z_th and not in_b:
                in_b, b_start = True, yr_val
            elif zv < z_th and in_b:
                slice_s = series[b_start:yr_val].dropna()
                if not slice_s.empty and b_start >= min_year:
                    results.append({
                        "MeSH":       term,
                        "バースト開始": b_start,
                        "バースト終了": yr_val - 1,
                        "期間":        yr_val - b_start,
                        "ピーク件数":  int(slice_s.max()),
                        "ピーク年":    int(slice_s.idxmax()),
                        "最大Zスコア": round(float(z[b_start:yr_val].dropna().max()), 2),
                    })
                in_b = False
        if in_b:
            slice_s = series[b_start:].dropna()
            if not slice_s.empty and b_start >= min_year:
                results.append({
                    "MeSH":       term,
                    "バースト開始": b_start,
                    "バースト終了": y_max,
                    "期間":        y_max - b_start + 1,
                    "ピーク件数":  int(slice_s.max()),
                    "ピーク年":    int(slice_s.idxmax()),
                    "最大Zスコア": round(float(z[b_start:].dropna().max()), 2),
                })
    return (pd.DataFrame(results)
            .sort_values(["バースト開始", "最大Zスコア"], ascending=[True, False])
            .reset_index(drop=True) if results else pd.DataFrame())

# ═══════════════════════════════════════════════════════════════
# セッション状態（デモ版: 起動時にサンプルデータを自動読み込み）
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_medline_text():
    """
    アプリと同じフォルダ内のサンプルデータファイルを探して読み込む。
    ロゴ探索（load_logo_b64）と同じパターンで、__file__基準・カレント
    ディレクトリ基準の両方を探索する（実行環境によるズレに対応するため）。
    見つからなければ (None, 探索したパス一覧) を返す。
    """
    search_bases = []
    try:
        search_bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    search_bases.append(os.getcwd())

    filenames = ["sample_data.txt"]
    sub_dirs = ["assets", ""]

    tried = []
    for base in search_bases:
        for sub in sub_dirs:
            for fname in filenames:
                path = os.path.join(base, sub, fname) if sub else os.path.join(base, fname)
                tried.append(path)
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        raw_bytes = f.read()
                    return raw_bytes.decode("utf-8", errors="ignore"), tried
    return None, tried

if "df" not in st.session_state:
    _sample_text, _sample_tried_paths = load_sample_medline_text()
    if _sample_text:
        st.session_state.df = parse_medline(_sample_text)
        st.session_state.using_sample_data = True
    else:
        st.session_state.df = None
        st.session_state.using_sample_data = False

# ═══════════════════════════════════════════════════════════════
# サイドバー：ロゴ（アプリと同じフォルダ内の画像ファイルを自動表示）
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_logo_b64():
    """
    assets/logo.png（推奨）またはリポジトリ直下の logo.png / logo.jpg を探して
    base64文字列を返す。見つからなければ (None, None, 探索したパス一覧) を返す。
    会社ロゴを差し替えたい場合は、assets/logo.png を置き換えるだけでよい。

    実行環境（Streamlit Cloud / Colab / ローカル）によって __file__ の基準位置と
    カレントディレクトリがずれることがあるため、両方を起点に探索する。
    """
    search_bases = []
    try:
        search_bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    search_bases.append(os.getcwd())

    filenames = ["logo.png", "logo.jpg", "logo.jpeg", "logo.PNG", "logo.JPG"]
    sub_dirs = ["assets", ""]

    tried = []
    for base in search_bases:
        for sub in sub_dirs:
            for fname in filenames:
                path = os.path.join(base, sub, fname) if sub else os.path.join(base, fname)
                tried.append(path)
                if os.path.exists(path):
                    ext = "jpeg" if path.lower().endswith(("jpg", "jpeg")) else "png"
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode(), ext, tried
    return None, None, tried

_logo_b64, _logo_ext, _logo_tried_paths = load_logo_b64()

with st.sidebar:
    if _logo_b64:
        st.markdown(
            f"""<div class="sidebar-logo-box">
                    <img src="data:image/{_logo_ext};base64,{_logo_b64}" />
                    <span class="sidebar-brand">文献Analyzer</span>
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-logo-box"><span class="sidebar-brand">📚 文献Analyzer</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("ロゴが見つかりません（クリックで詳細）", expanded=False):
            st.caption("以下のパスを探しましたが見つかりませんでした。"
                       "assets/logo.png がリポジトリに含まれているか（git push 済みか）を確認してください。")
            for p in _logo_tried_paths:
                st.code(p, language=None)

    st.markdown(
        '<div style="display:inline-block; font-size:0.72rem; font-weight:600; '
        'color:#0e8a80; background:#e1f5ee; padding:3px 10px; border-radius:12px; '
        'margin-bottom:4px;">🎬 デモ版 - サンプルデータ使用中</div>',
        unsafe_allow_html=True,
    )
    st.caption("ログイン不要でお試しいただけるサンプル版です。"
               "実際のデータでの分析もこのまま行えます。")
    st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# ナビゲーション
# ═══════════════════════════════════════════════════════════════
PAGES = [
    "🏠 Home",
    "📊 Overview",
    "📰 Journal分析",
    "🔥 ホットキーワード",
    "👤 著者分析",
]
page = st.sidebar.radio("ページ選択", PAGES)
st.sidebar.markdown("---")

# ── 全ページ共通：国フィルタ ─────────────────────────────────
# ページ選択の直下に配置し、選択した国の著者が関与する文献のみ分析対象にする
_base_df = st.session_state.df

if _base_df is not None and "AD" in _base_df.columns:
    with st.sidebar.expander("🌏 国フィルタ（全ページ共通）", expanded=False):
        @st.cache_data
        def _get_doc_countries(df_src: pd.DataFrame) -> list:
            """各文献の第一著者の国を返す（フィルタ用）"""
            countries = set()
            for _, row in df_src.iterrows():
                ad = row.get("AD", [])
                if not isinstance(ad, list): ad = [ad] if ad else []
                if ad:
                    c = extract_country(str(ad[0]))
                    if c != "Unknown":
                        countries.add(c)
            return sorted(countries)

        available_countries = _get_doc_countries(_base_df)
        selected_country_filter = st.selectbox(
            "第一著者の国で絞り込み",
            ["すべて"] + available_countries,
            key="global_country_filter",
        )
        if selected_country_filter != "すべて":
            st.caption(f"対象: {selected_country_filter} の著者が筆頭の文献のみ")

    # フィルタ適用
    if selected_country_filter != "すべて":
        _target = selected_country_filter   # lambda でキャプチャするためローカル変数に
        def _match_country(row):
            ad = row.get("AD", [])
            if not isinstance(ad, list):
                ad = [ad] if (ad and pd.notna(ad)) else []
            if not ad:
                return False
            return extract_country(str(ad[0])) == _target
        mask = _base_df.apply(_match_country, axis=1)
        df = _base_df[mask].copy()
    else:
        df = _base_df
else:
    selected_country_filter = "すべて"
    df = _base_df

st.sidebar.caption("文献Analyzer v2.3（デモ版）")

# ═══════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 📚 文献Analyzer")
    st.markdown(
        "<p style='font-size:1.05rem; color:#5a6b7d; margin-top:-0.5rem;'>"
        "PubMedのダウンロードデータを使って、該当領域のトレンドや著者の分析ができるアプリケーションです。"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**このデモ版について**

「抗NMDA受容体脳炎」関連のPubMed文献データ（約2,400件）を
あらかじめ読み込んだ状態でご覧いただけます。

左のナビから各ページを選ぶだけで、実際の分析結果を
すぐに確認できます。
        """)
    with col2:
        st.markdown("""
**ページ構成**

| ページ | 主な内容 |
|--------|---------|
| 📊 Overview | 文献数推移・Journalランキング・著者ランキング・バブルチャート |
| 📰 Top Journal分析 | Journal推移（積み上げ）・bigram特徴語・MeSH特徴語 |
| 🔥 Hot Keywords | 年度別特徴語・MeSHヒートマップ・バースト検知タイムライン |
| 👤 著者分析 | 著者別特徴語・共著ネットワーク・コミュニティ・期間比較 |
        """)
    st.success(
        "🎬 このデモ版には「抗NMDA受容体脳炎」関連のPubMed文献データ（約2,400件）が"
        "あらかじめ読み込まれています。左のナビから各ページをすぐにご覧いただけます。"
    )


# ═══════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Overview":
    st.markdown("## Overview")
    if df is None:
        st.error("サンプルデータの読み込みに失敗しました。アプリの再起動をお試しください。")
        st.stop()

    yd_all = df.dropna(subset=["year"]).copy() if "year" in df.columns else pd.DataFrame()
    if not yd_all.empty:
        yd_all["year"] = yd_all["year"].astype(int)
        y_min, y_max = int(yd_all["year"].min()), int(yd_all["year"].max())
    else:
        y_min, y_max = 2000, 2024

    with st.sidebar.expander("⚙️ Overview 設定", expanded=True):
        yr = st.slider("期間変更（文献数推移）", y_min, y_max,
                       (max(y_min, y_max - 30), y_max), key="ov_yr_slider")
        TOP_J  = st.slider("表示Journal数（Top Journals）", 5, 30, 15, key="ov_top_j")
        TOP_AU = st.slider("表示著者数（Top Authors）", 10, 30, 20, key="ov_top_au")
        TOP_BB = st.slider("表示著者数（著者推移）", 10, 30, 20, key="ov_bb_n")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 文献数推移", "📰 Top Journals", "👥 Top Authors", "🫧 著者推移", "🌏 国別推移"]
    )

    # ── Tab1: 文献数推移 ─────────────────────────────────────────
    with tab1:
        if "year" in df.columns:
            yd_f = yd_all[(yd_all["year"] >= yr[0]) & (yd_all["year"] <= yr[1])]
            year_counts = yd_f.groupby("year").size().reset_index(name="文献数")
            fig_yr = px.bar(
                year_counts, x="year", y="文献数",
                color_discrete_sequence=["#3498db"],
                title=f"文献数 年別推移（{yr[0]}〜{yr[1]}）",
                labels={"year": "Year"},
                height=480,
            )
            fig_yr.update_layout(xaxis=dict(dtick=1, tickangle=-45))
            st.plotly_chart(fig_yr, use_container_width=True)

    # ── Tab2: Top Journals ──────────────────────────────────────
    with tab2:
        if "TA" in df.columns:
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
            st.download_button("⬇️ top_journals.csv",
                               top_j.to_csv(index=False).encode(),
                               "top_journals.csv", "text/csv")

    # ── Tab3: Top Authors ────────────────────────────────────────
    with tab3:
        if "FAU" in df.columns:
            all_au, first_au, last_au = [], [], []
            for authors in df["FAU"]:
                if isinstance(authors, list) and authors:
                    s = [shorten_name(a) for a in authors]
                    all_au.extend(s); first_au.append(s[0]); last_au.append(s[-1])
            co  = pd.DataFrame(Counter(all_au).most_common(TOP_AU),  columns=["Author", "Co-author数"])
            fst = pd.DataFrame(Counter(first_au).most_common(TOP_AU), columns=["Author", "1st Author数"])
            lst = pd.DataFrame(Counter(last_au).most_common(TOP_AU),  columns=["Author", "Last Author数"])

            # ネットワーク分析と同じ所属マップを使い、バーにホバーすると所属が出るようにする
            ov_affil_map = build_author_affiliation_map(df) if "AD" in df.columns else {}
            ov_country_map = build_author_country_map(df) if "AD" in df.columns else {}

            c1, c2, c3 = st.columns(3)
            for col_st, data, xcol, clr, ttl in [
                (c1, co,  "Co-author数",  "#e74c3c", f"Co-author Top{TOP_AU}"),
                (c2, fst, "1st Author数", "#3498db", f"1st Author Top{TOP_AU}"),
                (c3, lst, "Last Author数","#9b59b6", f"Last Author Top{TOP_AU}"),
            ]:
                with col_st:
                    data = data.copy()
                    data["所属"] = data["Author"].map(lambda a: ov_affil_map.get(a, "（所属情報なし）"))
                    data["国"] = data["Author"].map(lambda a: ov_country_map.get(a, "―"))
                    fig = px.bar(data, x=xcol, y="Author", orientation="h",
                                 color_discrete_sequence=[clr], title=ttl, height=500,
                                 custom_data=["所属", "国"])
                    fig.update_traces(
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "所属: %{customdata[0]}<br>"
                            "国: %{customdata[1]}<br>"
                            + xcol + ": %{x}<extra></extra>"
                        )
                    )
                    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                    st.plotly_chart(fig, use_container_width=True, key=f"ov_author_{ttl}")
            if not ov_affil_map:
                st.caption("※ AD（所属）列が見つからないため、所属情報は表示されません。")

    # ── Tab4: 著者バブル ────────────────────────────────────────
    with tab4:
        if "FAU" in df.columns and "year" in df.columns:
            st.caption(
                "**横軸**：発表年　｜　**縦軸**：累積文献数（その著者のその年までの合計論文数）"
                "　｜　**バブルの大きさ**：その年の新規文献数（大きいほどその年の発表が多い）"
            )
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

    # ── Tab5: 国別文献推移 ───────────────────────────────────────
    with tab5:
        if "year" not in df.columns or "AD" not in df.columns:
            st.info("year 列または AD（所属）列が見つかりません。")
        else:
            st.caption("第一著者の所属国を基準に年別・国別の文献数を積み上げグラフで表示します。")
            _cy_rows = []
            for _, _row in df.dropna(subset=["year"]).iterrows():
                _ad = _row.get("AD", [])
                if not isinstance(_ad, list): _ad = [_ad] if _ad else []
                _c = extract_country(str(_ad[0])) if _ad else "Unknown"
                _cy_rows.append({"year": int(_row["year"]), "country": _c})
            cy_df = pd.DataFrame(_cy_rows)
            TOP_CTRY = 8
            top_ctry = cy_df[cy_df["country"] != "Unknown"]["country"].value_counts().head(TOP_CTRY).index.tolist()
            cy_df["country_grp"] = cy_df["country"].where(cy_df["country"].isin(top_ctry), other="Others")
            cy_trend = cy_df.groupby(["year", "country_grp"]).size().reset_index(name="count")
            ctry_order = top_ctry + ["Others"]
            cy_trend["country_grp"] = pd.Categorical(cy_trend["country_grp"], categories=ctry_order, ordered=True)
            cy_trend = cy_trend.sort_values(["year", "country_grp"])
            palette_c = px.colors.qualitative.Plotly
            cmap_c = {c: palette_c[i % len(palette_c)] for i, c in enumerate(top_ctry)}
            cmap_c["Others"] = "#cccccc"
            fig_ctry = px.bar(
                cy_trend, x="year", y="count", color="country_grp",
                barmode="stack",
                color_discrete_map=cmap_c,
                category_orders={"country_grp": ctry_order},
                labels={"count": "文献数", "year": "Year", "country_grp": "国"},
                title=f"国別 文献数推移（Top{TOP_CTRY}国 + Others）",
                height=460,
            )
            fig_ctry.update_layout(
                bargap=0.1,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
            )
            st.plotly_chart(fig_ctry, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Journal分析
# ═══════════════════════════════════════════════════════════════
elif page == "📰 Journal分析":
    st.markdown("## 📰 Top Journal分析")
    if df is None:
        st.error("サンプルデータの読み込みに失敗しました。アプリの再起動をお試しください。")
        st.stop()

    with st.sidebar.expander("⚙️ Top Journal分析 設定", expanded=True):
        TOP_STACK = st.slider("個別表示Journal数（残りはOthers）", 5, 20, 10, key="jnl_stack_n")
        top_j_n = st.slider("対象Journal数（bigram特徴語）", 3, 10, 6, key="jnl_j_n")
        top_w   = st.slider("表示フレーズ数（bigram特徴語）", 5, 20, 10, key="jnl_j_w")
        top_j_m = st.slider("対象Journal数（MeSH特徴語）", 3, 10, 6, key="jnl_mesh_n")
        top_m_w = st.slider("表示MeSH数（MeSH特徴語）", 5, 20, 10, key="jnl_mesh_w")
        st.markdown("---")
        jnl_recent_yr = st.slider(
            "比較期間（直近N年）", 1, 15, 5, key="jnl_recent_yr",
            help="全期間のTop Journalと、直近N年のTop Journalを比較します"
        )

    tab1, tab2, tab3 = st.tabs(["📊 Journal推移", "🔤 bigram特徴語", "🧬 MeSH特徴語"])

    # ── Tab1: Journal推移（Top N + Others 積み上げ）──────────────
    with tab1:
        if "TA" not in df.columns or "year" not in df.columns:
            st.error("TA または year 列が見つかりません。")
        else:
            jdf2 = df.dropna(subset=["TA", "year"]).copy()
            jdf2["year"] = jdf2["year"].astype(int)
            top_list = jdf2["TA"].value_counts().head(TOP_STACK).index.tolist()
            jdf2["TA_grp"] = jdf2["TA"].where(jdf2["TA"].isin(top_list), other="Others")
            trend = jdf2.groupby(["year", "TA_grp"]).size().reset_index(name="count")
            trend["year_str"] = trend["year"].astype(str)
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
                height=520,
            )
            fig_stack.update_layout(
                bargap=0.1,
                xaxis=dict(tickangle=-45),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
            )
            st.plotly_chart(fig_stack, use_container_width=True, key="jnl_trend_chart")

            # ── 全期間 vs 直近N年 の比較 ──────────────────────────
            st.markdown("---")
            ym_jnl = int(jdf2["year"].max())
            y_recent_jnl = ym_jnl - jnl_recent_yr + 1
            df_recent_jnl = jdf2[jdf2["year"] >= y_recent_jnl]

            all_top10  = jdf2["TA"].value_counts().head(10).reset_index()
            rec_top10  = df_recent_jnl["TA"].value_counts().head(10).reset_index()
            all_top10.columns = ["Journal", "全期間文献数"]
            rec_top10.columns  = ["Journal", f"直近{jnl_recent_yr}年文献数"]

            st.caption(
                f"**全期間 Top10** vs **直近{jnl_recent_yr}年（{y_recent_jnl}〜{ym_jnl}）Top10** の比較"
                f"（直近文献数: {len(df_recent_jnl):,}件 / 全体: {len(jdf2):,}件）"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                fig_all = px.bar(
                    all_top10[::-1], x="全期間文献数", y="Journal", orientation="h",
                    title="全期間 Top10", color_discrete_sequence=["#19b3a6"],
                    height=360, labels={"全期間文献数": "文献数"},
                )
                fig_all.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_all, use_container_width=True, key="jnl_all_top10")
            with col_b:
                fig_rec = px.bar(
                    rec_top10[::-1], x=f"直近{jnl_recent_yr}年文献数", y="Journal",
                    orientation="h",
                    title=f"直近{jnl_recent_yr}年 Top10", color_discrete_sequence=["#e74c3c"],
                    height=360, labels={f"直近{jnl_recent_yr}年文献数": "文献数"},
                )
                fig_rec.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_rec, use_container_width=True, key="jnl_rec_top10")

            # ── 直近N年の積み上げ棒グラフ ──────────────────────────
            st.markdown(f"**直近{jnl_recent_yr}年（{y_recent_jnl}〜{ym_jnl}）のJournal推移**")
            rec_top_list = df_recent_jnl["TA"].value_counts().head(TOP_STACK).index.tolist()
            df_recent_jnl = df_recent_jnl.copy()
            df_recent_jnl["TA_grp"] = df_recent_jnl["TA"].where(
                df_recent_jnl["TA"].isin(rec_top_list), other="Others"
            )
            rec_trend = df_recent_jnl.groupby(["year", "TA_grp"]).size().reset_index(name="count")
            rec_order = rec_top_list + ["Others"]
            rec_trend["TA_grp"] = pd.Categorical(rec_trend["TA_grp"], categories=rec_order, ordered=True)
            rec_color_map = {j: palette[i % len(palette)] for i, j in enumerate(rec_top_list)}
            rec_color_map["Others"] = "#cccccc"
            fig_rec_stack = px.bar(
                rec_trend, x="year", y="count", color="TA_grp",
                barmode="stack", color_discrete_map=rec_color_map,
                category_orders={"TA_grp": rec_order},
                labels={"count": "文献数", "year": "Year", "TA_grp": "Journal"},
                title=f"直近{jnl_recent_yr}年 Top{TOP_STACK} Journal + Others 年別推移",
                height=400,
            )
            fig_rec_stack.update_layout(
                bargap=0.1,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
            )
            st.plotly_chart(fig_rec_stack, use_container_width=True, key="jnl_recent_stack")

    # ── Tab2: Journal別 bigram特徴語 ────────────────────────────
    with tab2:
        if "TA" not in df.columns or "AB" not in df.columns:
            st.error("TA または AB 列が見つかりません。")
        else:
            jnlp = df.dropna(subset=["TA", "AB"]).copy()
            top_jlist = jnlp["TA"].value_counts().head(top_j_n).index.tolist()
            j_docs = {j: jnlp[jnlp["TA"] == j]["AB"].tolist() for j in top_jlist}
            feature_word_panel(j_docs, top_w, "bigram", "Journal別 特徴語 (bigram)",
                               key_prefix="jnl_bigram", group_order=top_jlist)
            with st.expander("単語（参考）"):
                j_texts = {j: " ".join(docs) for j, docs in j_docs.items()}
                tfidf_chart(j_texts, top_w, "word", "Journal別 特徴語 (word / TF-IDF)")

    # ── Tab3: Journal別 MeSH特徴語 ──────────────────────────────
    with tab3:
        if "TA" not in df.columns or "MH" not in df.columns:
            st.error("TA または MH 列が見つかりません。")
        else:
            jmesh = df.dropna(subset=["TA"]).copy()
            jmesh = jmesh[jmesh["MH"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
            top_jm = jmesh["TA"].value_counts().head(top_j_m).index.tolist()

            # 文献ごとに MeSH タームをスペース区切りの「文書」として扱う
            jm_docs = {j: [] for j in top_jm}
            for _, row in jmesh[jmesh["TA"].isin(top_jm)].iterrows():
                terms = [normalize_mesh(m) for m in row["MH"] if normalize_mesh(m) not in MESH_EXCLUDE]
                if terms:
                    jm_docs[row["TA"]].append(" ".join(t.replace(" ", "_") for t in terms))
            jm_docs = {j: docs for j, docs in jm_docs.items() if docs}
            if not jm_docs:
                st.info("MeSHデータが見つかりません。")
            else:
                # pretokenモードで渡す（MeSH語句はアンダースコア結合済みなので1トークン扱い、
                # tokenize() を経由させずそのままTF-IDFにかける）
                feature_word_panel(jm_docs, top_m_w, "pretoken", f"Journal別 MeSH特徴語 Top{top_m_w}",
                                   key_prefix="jnl_mesh", group_order=top_jm)

# ═══════════════════════════════════════════════════════════════
# ホットキーワード（MeSH分析を統合）
# ═══════════════════════════════════════════════════════════════
elif page == "🔥 ホットキーワード":
    st.markdown("## 🔥 Hot Keywords")
    if df is None:
        st.warning("先にデータをアップロードしてください。")
        st.stop()
    if "AB" not in df.columns:
        st.error("AB（Abstract）列が見つかりません。")
        st.stop()

    with st.sidebar.expander("⚙️ Hot Keywords 設定", expanded=True):
        SPAN   = st.slider("集約年数（1期間）", 1, 5, 2, key="hot_span")
        top_w3 = st.slider("表示フレーズ数", 5, 20, 10, key="hot_y_w")
        n_per  = st.slider("表示期間数", 2, 6, 4, key="hot_n_per")

    with st.sidebar.expander("⚙️ MeSH 分析設定", expanded=False):
        if "MH" in df.columns:
            top_mesh_n  = st.slider("ヒートマップ表示数", 20, 100, 60, key="mesh_top")
            min_year_m  = st.slider("表示開始年",
                                    int(df["year"].dropna().min()),
                                    int(df["year"].dropna().max()),
                                    max(int(df["year"].dropna().min()),
                                        int(df["year"].dropna().max()) - 20),
                                    key="mesh_miny")
            burst_top_n = st.slider("バースト対象MeSH数", 10, 60, 30, key="burst_top")
            burst_z_th  = st.slider("バースト Zスコア閾値", 0.5, 3.0, 1.5, 0.1, key="burst_z")

    tab1, tab2, tab3 = st.tabs(["🌡️ MeSHヒートマップ", "📅 年度別特徴語", "💥 バースト検知"])

    # ── Tab1: MeSHヒートマップ ────────────────────────────────────
    with tab1:
        if "MH" not in df.columns:
            st.error("MH（MeSH）列が見つかりません。")
        else:
            mh_count = df["MH"].apply(lambda x: len(x) > 0).sum()
            st.caption(f"MeSHデータあり文献: {mh_count:,} 件 / {len(df):,} 件")
            with st.spinner("MeSHデータを集計中..."):
                pivot_norm, mdf_all = build_mesh_pivot(df, top_mesh_n, min_year_m)

            if pivot_norm.empty:
                st.warning("条件に一致するMeSHデータがありません。")
            else:
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

                # ── バーストハイライト：バースト検知結果をヒートマップに重ねる ──
                # ヒートマップ描画後にバーストを計算してscatterで★を打つ
                burst_df_hl = detect_bursts(mdf_all, burst_top_n, min_year_m, burst_z_th) \
                              if not pivot_norm.empty else pd.DataFrame()
                y_terms = pivot_norm.columns.tolist()   # ヒートマップのy軸ラベル（MeSH一覧）
                x_years = [str(y) for y in pivot_norm.index]  # x軸ラベル（年文字列）

                if not burst_df_hl.empty:
                    # ヒートマップに載っているMeSHかつ年がx軸範囲内のもののみ
                    hl_x, hl_y, hl_text = [], [], []
                    for _, brow in burst_df_hl.iterrows():
                        mesh = brow["MeSH"]
                        peak_str = str(brow["ピーク年"])
                        if mesh in y_terms and peak_str in x_years:
                            hl_x.append(peak_str)
                            hl_y.append(mesh)
                            hl_text.append(
                                f"<b>🔥 {mesh}</b><br>"
                                f"バースト: {brow['バースト開始']}〜{brow['バースト終了']}年<br>"
                                f"ピーク: {brow['ピーク年']}年 / {brow['ピーク件数']}件<br>"
                                f"最大Zスコア: {brow['最大Zスコア']}"
                            )
                    if hl_x:
                        fig_heat.add_trace(go.Scatter(
                            x=hl_x, y=hl_y,
                            mode="markers",
                            marker=dict(
                                symbol="star",
                                size=12,
                                color="rgba(255,255,255,0.9)",
                                line=dict(color="#c0392b", width=1.5),
                            ),
                            hovertemplate="%{text}<extra></extra>",
                            text=hl_text,
                            name="🔥 バーストピーク",
                            showlegend=True,
                        ))
                        fig_heat.update_layout(
                            legend=dict(x=1.01, y=1, bgcolor="rgba(255,255,255,0.8)",
                                        bordercolor="#e3e8ee", borderwidth=1),
                        )
                        st.caption(
                            f"★ = バースト検知されたMeSHのピーク年（{len(hl_x)}件検出・"
                            f"Zスコア閾値={burst_z_th}）　ヒートマップにカーソルを当てると詳細が出ます"
                        )

                st.plotly_chart(fig_heat, use_container_width=True)
                st.download_button("⬇️ ヒートマップデータ (CSV)",
                                   pivot_norm.T.reset_index().to_csv(index=False).encode(),
                                   "mesh_heatmap.csv", "text/csv")

    # ── Tab2: 年度別 bigram特徴語 ────────────────────────────────
    with tab2:
        if "year" not in df.columns:
            st.error("year 列が見つかりません。")
        else:
            ynlp = df.dropna(subset=["year", "AB"]).copy()
            ynlp["year"] = ynlp["year"].astype(int)
            ym = ynlp["year"].max()
            periods_docs = {}
            period_order = []
            for i in range(n_per):
                y_end = ym - i * SPAN; y_start = y_end - SPAN + 1
                label = str(y_end) if SPAN == 1 else f"{y_end}〜{y_start}"
                subset = ynlp[ynlp["year"].isin(range(y_start, y_end + 1))]["AB"].tolist()
                if subset:
                    periods_docs[label] = subset
                    period_order.append(label)
            # グラフ上で古い→新しい順（時系列順）に並ぶよう並べ替える
            period_order_chrono = list(reversed(period_order))
            feature_word_panel(periods_docs, top_w3, "bigram", "年度別 特徴語 (bigram)",
                               key_prefix="hot_yearly", group_order=period_order_chrono)
            with st.expander("単語（参考）"):
                periods_text = {l: " ".join(docs) for l, docs in periods_docs.items()}
                tfidf_chart(periods_text, top_w3, "word", "年度別 特徴語 (word / TF-IDF)")

    # ── Tab3: バースト検知 ──────────────────────────────────────
    with tab3:
        if "MH" not in df.columns:
            st.error("MH（MeSH）列が見つかりません。")
        else:
            with st.spinner("バースト検知中..."):
                burst_df = detect_bursts(mdf_all, burst_top_n, min_year_m, burst_z_th) \
                           if not pivot_norm.empty else pd.DataFrame()
            if burst_df.empty:
                st.info("バーストが検出されませんでした。Zスコア閾値を小さくしてみてください。")
            else:
                st.success(f"検出バースト数: {len(burst_df)} 件")

                # ── バースト検知 結果テーブル ──────────────────────────
                st.markdown("##### バースト検知 結果一覧")
                st.caption("バースト開始年が早い順・同一開始年はZスコア降順で表示")
                st.dataframe(burst_df, use_container_width=False, hide_index=True)

                # ── バースト検知 タイムライン ───────────────────────────
                st.markdown("##### バースト検知 タイムライン")
                st.caption("バーの長さ＝バースト継続期間　◆マーク＝ピーク年　色が濃いほどZスコアが高い")
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

# ═══════════════════════════════════════════════════════════════
# 著者分析
# ═══════════════════════════════════════════════════════════════
elif page == "👤 著者分析":
    st.markdown("## 👤 著者分析")
    if df is None:
        st.warning("先にデータをアップロードしてください。")
        st.stop()
    if "FAU" not in df.columns:
        st.error("FAU（著者）列が見つかりません。")
        st.stop()

    with st.sidebar.expander("⚙️ 著者別特徴語 設定", expanded=True):
        top_a_n    = st.slider("対象著者数（著者別特徴語）", 3, 10, 6, key="au_a_n")
        top_w2     = st.slider("表示フレーズ数（著者別特徴語）", 5, 20, 10, key="au_a_w")

    with st.sidebar.expander("⚙️ 共著ネットワーク 設定", expanded=False):
        top_n_co   = st.slider("対象著者数（共著ネットワーク）", 20, 100, 50, key="co_topn")
        show_lbl   = st.checkbox("著者名を表示", value=True, key="co_lbl")
        top_comm_n = st.slider("コミュニティ表示数", 3, 8, 5, key="co_comm_n")
        top_feat_n = st.slider("特徴語表示数", 5, 15, 8, key="co_feat_n")
        span_net   = st.slider("期間比較（年数）", 2, 5, 3, key="co_span")
        run_co     = st.button("🔄 再実行（設定変更後）", key="co_run")

    tab1, tab2 = st.tabs(["🔤 著者別特徴語", "🕸️ 共著ネットワーク"])

    # ── Tab1: 著者別特徴語 ────────────────────────────────────
    with tab1:
        if "AB" not in df.columns:
            st.error("AB（Abstract）列が見つかりません。")
        else:
            # 1st Author（筆頭著者）基準: 各文献の最初の著者をその文献の代表とする
            # Overview の「Top Authors」の 1st Author ランキングと同じ基準
            au_rows_1st = [{"author": shorten_name(r["FAU"][0]), "AB": r["AB"]}
                           for _, r in df.iterrows()
                           if isinstance(r.get("FAU"), list) and r["FAU"] and pd.notna(r.get("AB"))]
            au_df_1st = pd.DataFrame(au_rows_1st)

            # Co-author（全著者）基準: 文献に名を連ねる全著者をそれぞれカウント
            au_rows_co = [{"author": shorten_name(a), "AB": r["AB"]}
                          for _, r in df.iterrows() if isinstance(r.get("FAU"), list) and pd.notna(r.get("AB"))
                          for a in r["FAU"]]
            au_df_co = pd.DataFrame(au_rows_co)

            st.markdown("#### 1st Author（筆頭著者）基準")
            if not au_df_1st.empty:
                top6_1st = au_df_1st["author"].value_counts().head(top_a_n).index.tolist()
                a_docs_1st = {a: au_df_1st[au_df_1st["author"] == a]["AB"].tolist() for a in top6_1st}
                feature_word_panel(a_docs_1st, top_w2, "bigram", "著者別 特徴語 (1st Author / bigram)",
                                   key_prefix="au_feat_1st", group_order=top6_1st)
            else:
                st.info("1st Authorのデータがありません。")

            st.markdown("---")
            st.markdown("#### Co-author（全著者）基準")
            if not au_df_co.empty:
                top6_co = au_df_co["author"].value_counts().head(top_a_n).index.tolist()
                a_docs_co = {a: au_df_co[au_df_co["author"] == a]["AB"].tolist() for a in top6_co}
                feature_word_panel(a_docs_co, top_w2, "bigram", "著者別 特徴語 (Co-author / bigram)",
                                   key_prefix="au_feat_co", group_order=top6_co)
            else:
                st.info("Co-authorのデータがありません。")

    # ── Tab2: 共著ネットワーク（3部構成）────────────────────────
    with tab2:
        # ★ デフォルト自動表示：初回ロード時もネットワークを描画する
        if "co_network_done" not in st.session_state:
            st.session_state.co_network_done = False
        do_run = run_co or not st.session_state.co_network_done

        if do_run:
            st.session_state.co_network_done = True
            with st.spinner("共著ネットワークを構築中..."):
                G_co = build_coauthor_graph(df["FAU"], top_n=top_n_co)
                fullname_map = build_author_fullname_map(df)
                affil_map = build_author_affiliation_map(df)
                country_map = build_author_country_map(df)

            # ① 全期間ネットワーク
            fig_co = nx_to_plotly(
                G_co, centrality="pagerank",
                title=f"共著ネットワーク（Top{top_n_co}）",
                directed=False, show_labels=show_lbl,
                fullname_map=fullname_map, affil_map=affil_map,
                country_map=country_map,
            )
            st.plotly_chart(fig_co, use_container_width=True, key="co_network_main")

            # 中心性スコア：pagerank・betweenness 両方をTop10で並列表示
            pr_scores = nx.pagerank(G_co, weight="weight")
            bw_scores = nx.betweenness_centrality(G_co, weight="weight")
            top_pr = pd.DataFrame(
                sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:10],
                columns=["著者", "pagerankスコア"]
            )
            top_bw = pd.DataFrame(
                sorted(bw_scores.items(), key=lambda x: x[1], reverse=True)[:10],
                columns=["著者", "betweennessスコア"]
            )
            st.markdown("#### 中心性スコア Top10（複数軸）")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.caption("pagerank（影響力の大きさ）")
                st.dataframe(top_pr, use_container_width=True, hide_index=True)
            with sc2:
                st.caption("betweenness（橋渡し役の大きさ）")
                st.dataframe(top_bw, use_container_width=True, hide_index=True)

            # コミュニティ抽出は pagerank スコアを使って主要著者を決める
            cent_scores = pr_scores

            # ② コミュニティ分析
            st.markdown("---")
            st.markdown("#### コミュニティ分析")
            if "AB" in df.columns:
                from networkx.algorithms.community import greedy_modularity_communities
                comms = list(greedy_modularity_communities(G_co, weight="weight"))
                comms = sorted([c for c in comms if len(c) >= 3], key=len, reverse=True)
                st.caption(f"検出コミュニティ数（3名以上）: {len(comms)}")

                affil_map = build_author_affiliation_map(df)

                comm_docs = {}
                comm_label_order = []
                comm_info_rows = []
                for i, comm in enumerate(comms[:top_comm_n]):
                    label = f"Comm {i+1}（{len(comm)}名）"
                    comm_label_order.append(label)
                    docs = get_community_texts(comm, df)
                    if docs:
                        comm_docs[label] = docs
                    top5 = [a for a, _ in sorted(
                        cent_scores.items(), key=lambda x: x[1], reverse=True
                    ) if a in comm][:5]
                    top5_affils = [affil_map.get(a, "―") for a in top5]
                    comm_info_rows.append({
                        "コミュニティ": label,
                        "人数": len(comm),
                        "主要著者（中心性上位5名）": ", ".join(top5),
                        "主要著者の所属": " / ".join(top5_affils) if affil_map else "（AD列なし）",
                    })
                st.dataframe(pd.DataFrame(comm_info_rows), use_container_width=False)
                if comm_docs:
                    # Comm1, Comm2... の順番で並ぶように group_order で明示
                    feature_word_panel(comm_docs, top_feat_n, "bigram",
                                       f"コミュニティ別 特徴語（Top{top_feat_n}）",
                                       key_prefix="comm_feat", group_order=comm_label_order)
            else:
                st.info("AB列がないためコミュニティ特徴語は表示できません。")

            # ③ 期間比較（左: 直近span_net年より前の全期間／右: 直近年までの全期間。新規ノード・新規エッジをハイライト）
            st.markdown("---")
            st.markdown("#### 期間比較ネットワーク")
            y_min_net = int(df["year"].dropna().min())
            y_max_net = int(df["year"].dropna().max())
            boundary_y = y_max_net - span_net + 1   # 例: span_net=3, y_max=2026 → 2024
            prev_y    = list(range(y_min_net, boundary_y))        # 例: 〜2023年まで
            recent_y  = list(range(y_min_net, y_max_net + 1))     # 例: 〜2026年まで（全期間）
            st.caption(
                f"左＝{boundary_y - 1}年まで: {prev_y[0] if prev_y else '―'}〜{prev_y[-1] if prev_y else '―'}"
                f"　／　右＝直近まで: {recent_y[0]}〜{recent_y[-1]}（直近{span_net}年分を含む全期間）"
                "　｜　色＝コミュニティ／オレンジ枠＝直近の追加期間で新たに登場した著者・共著関係"
            )
            G_rec  = build_coauthor_graph(df[df["year"].isin(recent_y)]["FAU"], top_n=top_n_co)
            G_prev = build_coauthor_graph(df[df["year"].isin(prev_y)]["FAU"],   top_n=top_n_co) \
                     if prev_y else nx.Graph()

            prev_node_set = set(G_prev.nodes())
            prev_edge_set = set(frozenset(e) for e in G_prev.edges())
            new_nodes_rec = set(G_rec.nodes()) - prev_node_set
            new_edges_rec = [e for e in G_rec.edges() if frozenset(e) not in prev_edge_set]

            # 中心性スコア（pagerank・betweenness 両方をTop10で表示。左右で同じ位置・同じ形式に揃える）
            def _top10_both(G):
                if not G.nodes:
                    return pd.DataFrame(columns=["著者", "pagerankスコア"]), \
                           pd.DataFrame(columns=["著者", "betweennessスコア"])
                pr = nx.pagerank(G, weight="weight")
                bw = nx.betweenness_centrality(G, weight="weight")
                pr_df = pd.DataFrame(sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10],
                                      columns=["著者", "pagerankスコア"])
                bw_df = pd.DataFrame(sorted(bw.items(), key=lambda x: x[1], reverse=True)[:10],
                                      columns=["著者", "betweennessスコア"])
                return pr_df, bw_df

            top_pr_prev, top_bw_prev = _top10_both(G_prev)
            top_pr_rec,  top_bw_rec  = _top10_both(G_rec)
            # 直近側は新規著者がわかるようにマーク
            if not top_pr_rec.empty:
                top_pr_rec["著者"] = top_pr_rec["著者"].apply(
                    lambda a: ("🆕 " if a in new_nodes_rec else "") + a)
            if not top_bw_rec.empty:
                top_bw_rec["著者"] = top_bw_rec["著者"].apply(
                    lambda a: ("🆕 " if a in new_nodes_rec else "") + a)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(
                    nx_to_plotly_diff(G_prev, centrality="pagerank",
                                       title=f"{boundary_y - 1}年まで（{prev_y[0] if prev_y else '―'}〜{prev_y[-1] if prev_y else '―'}）",
                                       show_labels=show_lbl,
                                       highlight_nodes=set(), highlight_edges=set(),
                                       fullname_map=fullname_map, affil_map=affil_map),
                    use_container_width=True, key="co_network_prev",
                )
                st.caption("中心性スコア Top10")
                pr_col1, bw_col1 = st.columns(2)
                with pr_col1:
                    st.caption("pagerank")
                    st.dataframe(top_pr_prev, use_container_width=True, hide_index=True)
                with bw_col1:
                    st.caption("betweenness")
                    st.dataframe(top_bw_prev, use_container_width=True, hide_index=True)
            with cc2:
                st.plotly_chart(
                    nx_to_plotly_diff(G_rec, centrality="pagerank",
                                       title=f"直近まで（{recent_y[0]}〜{recent_y[-1]}）",
                                       show_labels=show_lbl,
                                       highlight_nodes=new_nodes_rec,
                                       highlight_edges=set(frozenset(e) for e in new_edges_rec),
                                       fullname_map=fullname_map, affil_map=affil_map),
                    use_container_width=True, key="co_network_recent",
                )
                st.caption("中心性スコア Top10　🆕＝新規著者")
                pr_col2, bw_col2 = st.columns(2)
                with pr_col2:
                    st.caption("pagerank")
                    st.dataframe(top_pr_rec, use_container_width=True, hide_index=True)
                with bw_col2:
                    st.caption("betweenness")
                    st.dataframe(top_bw_rec, use_container_width=True, hide_index=True)

            # 新規共著ペア（共著件数 ＋ そのペアの共著文献に出現するMeSHターム件数 Top10 を同じ表に）
            new_edge_rows = [
                {"著者A": u, "著者B": v, "共著件数": int(data.get("weight", 1))}
                for u, v, data in G_rec.edges(data=True)
                if frozenset((u, v)) not in prev_edge_set
            ]
            if new_edge_rows:
                new_df = (pd.DataFrame(new_edge_rows)
                          .sort_values("共著件数", ascending=False)
                          .reset_index(drop=True))
                st.markdown(f"#### 新規共著ペア（{boundary_y}年以降に初登場）: {len(new_df)} ペア")

                if "MH" in df.columns:
                    recent_df = df[df["year"].isin(recent_y)]
                    mesh_top10_col = []
                    for u, v in zip(new_df["著者A"], new_df["著者B"]):
                        # そのペア（u と v が両方とも著者リストに含まれる文献）のみを対象にMeSHを集計
                        mesh_counter = Counter()
                        for _, row in recent_df.iterrows():
                            fau = row.get("FAU", [])
                            if not isinstance(fau, list):
                                continue
                            authors_short = {shorten_name(a) for a in fau}
                            if {u, v} <= authors_short:
                                for m in (row.get("MH") or []):
                                    nm = normalize_mesh(m)
                                    if nm not in MESH_EXCLUDE:
                                        mesh_counter[nm] += 1
                        if mesh_counter:
                            top10_str = ", ".join(
                                f"{term}({cnt})" for term, cnt in mesh_counter.most_common(10)
                            )
                        else:
                            top10_str = "―"
                        mesh_top10_col.append(top10_str)
                    new_df["MeSHターム Top10（件数）"] = mesh_top10_col
                else:
                    new_df["MeSHターム Top10（件数）"] = "（MH列なし）"

                st.dataframe(new_df.head(30), use_container_width=False, hide_index=True)
            else:
                st.info("新規共著ペアなし。")

            # ─── PI影響力ネットワーク（Last Author → Other Author 有向グラフ）───
            st.markdown("---")
            st.markdown("#### 🎯 PI影響力ネットワーク（有向グラフ）")
            st.caption(
                "Last Author（最終著者）から Other Author（中間・筆頭著者）への**有向エッジ**を張ったネットワークです。"
                "学術慣行上、Last AuthorはPI（Principal Investigator：研究主宰者）であることが多く、"
                "**矢印の向き**：Last Author → 共著者（その研究を指導・支援した方向）　"
                "**円のサイズ**：PageRankスコア（影響力）　**色**：Last Author文献数（赤いほど多い）"
            )

            if "FAU" in df.columns:
                DG = nx.DiGraph()
                for _, row in df.iterrows():
                    fau = row.get("FAU", [])
                    if not isinstance(fau, list) or len(fau) < 2:
                        continue
                    last = shorten_name(fau[-1])
                    for other in fau[:-1]:
                        other_s = shorten_name(other)
                        if DG.has_edge(last, other_s):
                            DG[last][other_s]["weight"] += 1
                        else:
                            DG.add_edge(last, other_s, weight=1)

                pi_top_n = 60
                flat_all = [shorten_name(a) for authors in df["FAU"] if isinstance(authors, list) for a in authors]
                top_authors_set = {a for a, _ in Counter(flat_all).most_common(pi_top_n)}
                DG_top = DG.subgraph([n for n in DG.nodes() if n in top_authors_set]).copy()

                if DG_top.number_of_nodes() > 0:
                    pr_d = nx.pagerank(DG_top, weight="weight")
                    max_pr_d = max(pr_d.values()) if pr_d else 1

                    np.random.seed(42)
                    pos_d = nx.spring_layout(
                        DG_top, seed=42, k=1.5 / math.sqrt(max(len(DG_top.nodes()), 1))
                    )

                    last_au_counter = Counter()
                    for authors in df["FAU"]:
                        if isinstance(authors, list) and authors:
                            last_au_counter[shorten_name(authors[-1])] += 1

                    # 有向エッジを Plotly の annotations（矢印）で描画
                    # 重みの大きいエッジのみ表示（見やすさのため上位200本に絞る）
                    edges_sorted = sorted(DG_top.edges(data=True), key=lambda x: x[2].get("weight", 1), reverse=True)
                    max_w = max(d.get("weight", 1) for _, _, d in edges_sorted) if edges_sorted else 1

                    annotations = []
                    for u, v, d in edges_sorted[:200]:
                        x0, y0 = pos_d[u]
                        x1, y1 = pos_d[v]
                        w = d.get("weight", 1)
                        alpha = max(0.15, min(0.7, w / max_w))
                        annotations.append(dict(
                            x=x1, y=y1,       # 矢印の先端（共著者側）
                            ax=x0, ay=y0,     # 矢印の始点（Last Author側）
                            xref="x", yref="y",
                            axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1.0,
                            arrowwidth=max(0.5, w / max_w * 2.5),
                            arrowcolor=f"rgba(200,140,60,{alpha:.2f})",
                        ))

                    # ノードのスキャッタープロット
                    node_colors = [last_au_counter.get(n, 0) for n in DG_top.nodes()]
                    node_sizes = [max(8, pr_d.get(n, 0) / max_pr_d * 45) for n in DG_top.nodes()]

                    fig_dir = go.Figure()
                    fig_dir.add_trace(go.Scatter(
                        x=[pos_d[n][0] for n in DG_top.nodes()],
                        y=[pos_d[n][1] for n in DG_top.nodes()],
                        mode="markers+text" if show_lbl else "markers",
                        text=list(DG_top.nodes()) if show_lbl else [],
                        textposition="top center",
                        textfont=dict(size=8),
                        marker=dict(
                            size=node_sizes,
                            color=node_colors,
                            colorscale="YlOrRd",
                            showscale=True,
                            colorbar=dict(title="Last Author<br>文献数", thickness=12),
                            line=dict(width=1, color="white"),
                        ),
                        hovertext=[
                            f"<b>{fullname_map.get(n, n)}</b><br>"
                            f"所属: {affil_map.get(n, '―')}<br>"
                            f"国: {country_map.get(n, '―')}<br>"
                            f"PageRank（影響力）: {pr_d.get(n, 0):.4f}<br>"
                            f"Last Author文献数: {last_au_counter.get(n, 0)}件"
                            for n in DG_top.nodes()
                        ],
                        hoverinfo="text",
                        showlegend=False,
                    ))
                    fig_dir.update_layout(
                        title=(
                            f"PI影響力ネットワーク（Top{pi_top_n}著者・有向グラフ）<br>"
                            "<sup>矢印: Last Author → 共著者　"
                            "円サイズ=PageRank（影響力）　色=Last Author文献数（赤＝多い）</sup>"
                        ),
                        title_x=0.5,
                        height=620,
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        margin=dict(l=20, r=20, t=80, b=20),
                    )
                    st.plotly_chart(fig_dir, use_container_width=True, key="pi_influence_network")

                    # PageRank Top10テーブル
                    top_pi = sorted(pr_d.items(), key=lambda x: x[1], reverse=True)[:10]
                    st.caption("PI影響力 PageRank Top10")
                    pi_df = pd.DataFrame([
                        {
                            "著者": a,
                            "所属": affil_map.get(a, "―"),
                            "国": country_map.get(a, "―"),
                            "PageRank": round(s, 5),
                            "Last Author文献数": last_au_counter.get(a, 0),
                        }
                        for a, s in top_pi
                    ])
                    st.dataframe(pi_df, use_container_width=False, hide_index=True)
                else:
                    st.info("有向グラフを構築できませんでした。")
            else:
                st.info("FAU列が見つかりません。")