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
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import os
import json
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
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 認証
# ═══════════════════════════════════════════════════════════════
def _to_plain_dict(obj):
    """
    st.secrets が返す Secrets/AttrDict 型を、json.dumps に頼らず
    再帰的に素の dict / list / str に変換する。
    （Secrets型は dict 互換だが json.dumps にそのまま渡すと
      TypeError: Object of type Secrets is not JSON serializable
      になることがあるための対策）
    """
    if hasattr(obj, "items"):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]
    return obj


def load_auth_config():
    """
    認証情報の読み込み優先順位:
      1. Streamlit Cloud Secrets の st.secrets["PUBMED_APP_CONFIG"]（本番想定）
      2. ローカルの config.yaml（ローカル開発専用。.gitignore 対象）
    どちらにも見つからない場合は (None, None, エラーメッセージ) を返す。
    """
    # ① Secrets（本番）
    if "PUBMED_APP_CONFIG" in st.secrets:
        try:
            raw = st.secrets["PUBMED_APP_CONFIG"]
            cfg = _to_plain_dict(raw)
            return cfg, "secrets", None
        except Exception as e:
            # ここで握りつぶさず、原因をそのまま持ち帰る
            return None, None, f"Secrets の読み込み中にエラー: {type(e).__name__}: {e}"

    # ② ローカル config.yaml（フォールバック。本番運用では使わない想定）
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml") \
                 if "__file__" in globals() else "config.yaml"
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=SafeLoader)
            return cfg, "local_yaml", None
        except Exception as e:
            return None, None, f"config.yaml の読み込み中にエラー: {type(e).__name__}: {e}"

    return None, None, (
        "st.secrets に 'PUBMED_APP_CONFIG' が見つからず、"
        "config.yaml も存在しませんでした。"
    )


_auth_cfg, _auth_source, _auth_error = load_auth_config()

if _auth_cfg is None:
    st.error(
        "🔒 認証設定が見つかりません。\n\n"
        "管理者の方へ：Streamlit Cloud の **Settings → Secrets** に "
        "`PUBMED_APP_CONFIG`（TOML形式）を登録するか、ローカル開発時は "
        "`config.yaml` を配置してください。詳細は公開手順書を参照してください。"
    )
    with st.expander("🛠️ 管理者向け：詳細なエラー内容", expanded=True):
        st.code(_auth_error or "(詳細不明)", language=None)
        st.caption(
            "上記のメッセージを確認してください。TOMLの構文エラー（クォートの不一致など）や、"
            "キー名の誤り（PUBMED_APP_CONFIG）が典型的な原因です。"
        )
    st.stop()

try:
    _credentials   = _auth_cfg["credentials"]
    _cookie_name   = _auth_cfg["cookie"]["name"]
    _cookie_key    = _auth_cfg["cookie"]["key"]
    _cookie_expiry = float(_auth_cfg["cookie"].get("expiry_days", 7))
except (KeyError, TypeError) as e:
    st.error("🔒 認証設定の形式が不正です。`credentials` / `cookie` の構造を確認してください。")
    with st.expander("🛠️ 管理者向け：詳細なエラー内容", expanded=True):
        st.code(f"{type(e).__name__}: {e}", language=None)
    st.stop()

authenticator = stauth.Authenticate(
    _credentials,
    _cookie_name,
    _cookie_key,
    _cookie_expiry,
    auto_hash=False,  # config側はあらかじめ bcrypt ハッシュ済みのパスワードを保持する運用
)

authenticator.login(location="main")

if st.session_state.get("authentication_status") is False:
    st.error("❌ ユーザー名またはパスワードが正しくありません。")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("🔑 ユーザー名とパスワードを入力してください。")
    st.stop()
# authentication_status が True の場合のみ、以降のアプリ本体が描画される


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



def tfidf_chart(group_texts: dict, top_n: int, mode: str, title: str, wrap: int = None,
                 group_order: list = None):
    result = tfidf_top(group_texts, top_n, mode)
    if result.empty:
        st.info("データが不足しています。")
        return
    result = result.sort_values("tfidf", ascending=True)
    n_groups = result["group"].nunique()
    # wrap未指定時：3列を上限に横並びし、4グループ以上は3列で折り返す（視認性重視）
    facet_wrap = wrap if wrap is not None else min(n_groups, 3)
    n_rows = math.ceil(n_groups / facet_wrap)
    max_rows_per_group = result.groupby("group").size().max()
    row_h = 22  # 1行あたりの高さ（以前のMeSH特徴語と同等のサイズ感）
    category_orders = {}
    if group_order:
        # 実際に存在するグループのみ・指定順を維持
        present = set(result["group"].unique())
        category_orders["group"] = [g for g in group_order if g in present]
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
    # facetタイトルのプレフィックス除去 & フォント拡大
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
    group_docs はグループ名 -> 生テキストのリスト。
    （以前は「推移＋乖離」「文脈(KWIC)」タブも含んでいたが、tf-idf計算式を
     R/tidytext互換に修正したことでグループ間の違いがTF-IDFだけで明確に出る
     ようになったため、シンプルにTF-IDF表示のみとした。
     key_prefix・title_lookup は当時KWICタブ用に使っていた名残で、
     現在は未使用だが呼び出し側の互換性のため引数として残してある）
    """
    group_texts_joined = {g: " ".join(docs) for g, docs in group_docs.items()}
    tfidf_chart(group_texts_joined, top_n, mode, title, wrap=wrap, group_order=group_order)



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
                 fullname_map: dict = None, affil_map: dict = None):
    if not G.nodes:
        return go.Figure()
    fullname_map = fullname_map or {}
    affil_map = affil_map or {}
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
        return f"<b>{full}</b><br>所属: {affil}"

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

# 所属（AD）から最上位の組織名を抽出するためのキーワード
# 優先度1: 大学・病院・研究所本体など、最上位の組織を示すキーワード
_INST_TOP_KEYWORDS = [
    "University", "Univ\\.", "College", "Hospital", "Institute", "Institut[e]?",
    "Academy", "Foundation",
    "Universität", "Klinikum", "Charité",
    "Université", "Hôpital", "Hôpitaux", "CHU",
    "Universidad", "Universitat", "Instituto",
    "Università", "Ospedale",
    "Universitet", "Universiteit",
]
# 優先度2: 学部・センターなど下位区分（優先度1で見つからない場合のフォールバック）
_INST_SUB_KEYWORDS = ["School of Medicine", "Medical School", "Clinic", "Clinique",
                       "Center", "Centre", "Laboratory"]
_INST_TOP_PATTERN = re.compile(r"(" + "|".join(_INST_TOP_KEYWORDS) + r")", re.IGNORECASE)
_INST_SUB_PATTERN = re.compile(r"(" + "|".join(_INST_SUB_KEYWORDS) + r")", re.IGNORECASE)


def extract_top_institution(ad_text: str) -> str:
    """
    AD（所属）の生テキストから最上位の組織名（大学・病院・研究所など）を抽出する。
    "Department of Neurology, Tokyo University, Tokyo, Japan" のような文字列から
    "Tokyo University" のような最上位機関名のみを取り出すことを狙う。

    優先順位:
      1. 大学・病院・研究所などのキーワードを含む最初の断片
      2. 学部・センターなどのキーワードを含む最初の断片
      3. どちらもなければ、末尾から2番目の断片（都市名の前は機関名であることが多い）
    """
    fragments = [f.strip().rstrip(".") for f in re.split(r"[;,]", str(ad_text)) if f.strip()]
    if not fragments:
        return "―"
    for frag in fragments:
        if _INST_TOP_PATTERN.search(frag):
            return frag
    for frag in fragments:
        if _INST_SUB_PATTERN.search(frag):
            return frag
    if len(fragments) >= 2:
        return fragments[-2]
    return fragments[0]


@st.cache_data
def build_author_affiliation_map(df_src: pd.DataFrame) -> dict:
    """
    著者の短縮名 → 最上位組織名（大学・病院など）の対応辞書を作る。
    PubMed/MEDLINE形式では AD は文献単位の情報のため、
    各著者が関わった文献から抽出した組織名のうち最も頻出するものをその著者の所属とみなす。
    AD列が存在しない場合は空の辞書を返す。
    """
    if "AD" not in df_src.columns:
        return {}
    affil_counter: dict = {}
    for _, row in df_src.iterrows():
        fau = row.get("FAU", [])
        ad  = row.get("AD", [])
        if not isinstance(fau, list) or not fau:
            continue
        if isinstance(ad, list):
            ad_text = ad[0] if ad else None
        else:
            ad_text = ad
        if not ad_text or pd.isna(ad_text):
            continue
        top_inst = extract_top_institution(str(ad_text))
        if not top_inst or top_inst == "―":
            continue
        for a in fau:
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

    st.caption(f"👤 {st.session_state.get('name', '')} さんでログイン中")
    authenticator.logout("ログアウト", "sidebar", key="logout_btn")
    st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# ナビゲーション
# ═══════════════════════════════════════════════════════════════
PAGES = [
    "🏠 Home",
    "📂 Data Upload",
    "📊 Overview",
    "📰 Journal分析",
    "🔥 ホットキーワード",
    "👤 著者分析",
]
page = st.sidebar.radio("ページ選択", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("文献Analyzer v2.3")

df = st.session_state.df

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
**使い方**
1. [PubMed](https://pubmed.ncbi.nlm.nih.gov/) で任意のキーワードで文献を検索
2. **PubMed形式** でデータをダウンロード  
   （Send to → Citation manager → Format: PubMed）
3. 「📂 Data Upload」からファイルをアップロード
4. 左のナビから目的のページを選んで分析
        """)
    with col2:
        st.markdown("""
**ページ構成**

| ページ | 主な内容 |
|--------|---------|
| 📂 Data Upload | ファイル読込・基礎統計・CSV出力 |
| 📊 Overview | 文献数推移・Journalランキング・著者ランキング・バブルチャート |
| 📰 Top Journal分析 | Journal推移（積み上げ）・bigram特徴語・MeSH特徴語 |
| 🔥 Hot Keywords | 年度別特徴語・MeSHヒートマップ・バースト検知タイムライン |
| 👤 著者分析 | 著者別特徴語・共著ネットワーク・コミュニティ・期間比較 |
        """)
    st.info("⬅️ 左のサイドバーから「📂 Data Upload」を選んでください。")

# ═══════════════════════════════════════════════════════════════
# DATA UPLOAD
# ═══════════════════════════════════════════════════════════════
elif page == "📂 Data Upload":
    st.markdown("## 解析データのアップロード・基礎解析")

    uploaded = st.file_uploader(
        "PubMed形式テキストファイルをアップロード（.txt）",
        type=["txt"],
        help="PubMedからPubMed形式でダウンロードしたファイル。目安1,500文献程度まで。",
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

        # ── データプレビュー ───────────────────────────────────
        st.markdown("### データプレビュー（先頭10件）")
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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 文献数推移", "📰 Top Journals", "👥 Top Authors", "🫧 著者推移"]
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
                    st.plotly_chart(fig, use_container_width=True, key=f"ov_author_{ttl}")

    # ── Tab4: 著者バブル ────────────────────────────────────────
    with tab4:
        if "FAU" in df.columns and "year" in df.columns:
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
# Journal分析
# ═══════════════════════════════════════════════════════════════
elif page == "📰 Journal分析":
    st.markdown("## 📰 Top Journal分析")
    if df is None:
        st.warning("先に「📂 Data Upload」でデータをアップロードしてください。")
        st.stop()

    with st.sidebar.expander("⚙️ Top Journal分析 設定", expanded=True):
        TOP_STACK = st.slider("個別表示Journal数（残りはOthers）", 5, 20, 10, key="jnl_stack_n")
        top_j_n = st.slider("対象Journal数（bigram特徴語）", 3, 10, 6, key="jnl_j_n")
        top_w   = st.slider("表示フレーズ数（bigram特徴語）", 5, 20, 10, key="jnl_j_w")
        top_j_m = st.slider("対象Journal数（MeSH特徴語）", 3, 10, 6, key="jnl_mesh_n")
        top_m_w = st.slider("表示MeSH数（MeSH特徴語）", 5, 20, 10, key="jnl_mesh_w")

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
        SPAN   = st.slider("集約年数（1期間）", 1, 5, 3, key="hot_span")
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
            au_rows = [{"author": shorten_name(r["FAU"][0]), "AB": r["AB"]}
                       for _, r in df.iterrows()
                       if isinstance(r.get("FAU"), list) and r["FAU"] and pd.notna(r.get("AB"))]
            au_df2 = pd.DataFrame(au_rows)
            if not au_df2.empty:
                top6a = au_df2["author"].value_counts().head(top_a_n).index.tolist()
                a_docs = {a: au_df2[au_df2["author"] == a]["AB"].tolist() for a in top6a}
                feature_word_panel(a_docs, top_w2, "bigram", "著者別 特徴語 (bigram)",
                                   key_prefix="au_feat", group_order=top6a)

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

            # ① 全期間ネットワーク
            fig_co = nx_to_plotly(
                G_co, centrality="pagerank",
                title=f"共著ネットワーク（Top{top_n_co}）",
                directed=False, show_labels=show_lbl,
                fullname_map=fullname_map, affil_map=affil_map,
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
                st.dataframe(pd.DataFrame(comm_info_rows), use_container_width=True)
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

                st.dataframe(new_df.head(30), use_container_width=True, hide_index=True)
            else:
                st.info("新規共著ペアなし。")