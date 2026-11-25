# TD_WIRA.py
import hashlib
import html
import io
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Import dengan fallback untuk menghindari ModuleNotFoundError
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        EMBEDDINGS_AVAILABLE = True
    except ImportError:
        HuggingFaceEmbeddings = None
        EMBEDDINGS_AVAILABLE = False

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI API & HALAMAN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TEDIWIRA",
    page_icon="📓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "faiss_index"

# Model & parameter bisa dioverride lewat environment variable
MODEL_CHAT = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
MODEL_PROPOSAL = os.getenv("GROQ_PROPOSAL_MODEL", "openai/gpt-oss-20b")

MAX_TOKENS_CHAT = int(os.getenv("GROQ_MAX_TOKENS_CHAT", "3000"))
MAX_TOKENS_PROPOSAL = int(os.getenv("GROQ_MAX_TOKENS_PROPOSAL", "4096"))

TEMP_CHAT = float(os.getenv("GROQ_TEMP_CHAT", "0.45"))
TEMP_PROPOSAL = float(os.getenv("GROQ_TEMP_PROPOSAL", "0.3"))

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))

# Batas agar prompt tidak melebihi context window model / tagihan API
MAX_PROPOSAL_CHARS = 30_000
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 12_000
MAX_PROPOSAL_CONTEXT_CHARS = int(os.getenv("MAX_PROPOSAL_CONTEXT_CHARS", "4000"))

MAX_QUESTIONS = 7  # batas pertanyaan per sesi browser (pengaman kuota API)

PROMPT_VERSION = "v2"  # ubah bila prompt analisis diubah (cache buster)

try:
    os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    st.error(
        "GROQ_API_KEY belum diset. Tambahkan `GROQ_API_KEY = \"gsk_...\"` "
        "di `.streamlit/secrets.toml` atau environment variable, lalu jalankan ulang aplikasi."
    )
    st.stop()

# CSS Styling
st.markdown(
    """
    <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #f0f2f6; }
        .bot-message { background-color: #e8f0fe; }
        .analysis-box { background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 1rem 0; border-radius: 0.5rem; }
        .startup-info { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
        .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; color: #721c24; }
        .attach-box { background-color: #eef7ee; border: 1px solid #cfe8cf; padding: 0.75rem 1rem; border-radius: 0.5rem; margin: 0.5rem 0; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul Aplikasi
st.title("📓TEMAN DISKUSI KEWIRAUSAHAAN")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan Tentang Kewirausahaan

    **Chat Bot ini akan membantu Anda memahami lebih dalam tentang dunia KEWIRAUSAHAAN** dan berbagai hal-hal yang perlu di perhatikan baik pada masa persiapan, pelaksanaan, pengembangan, dan bahkan exit strategy.

    Aplikasi ini juga dapat membantu Anda **MERIVIEW PROPOSAL TAWARAN BISNIS atau KERJASAMA** yang datang pada Anda.

    **Pastikanlah Anda memiliki koneksi internet yang baik dan stabil.**
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE DAN INISIALISASI
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_STATE = {
    "chain": None,
    "chain_error": None,
    "chat_history": [],           # [{role, content, sources?}]
    "proposal_analysis": None,
    "proposal_error": None,
    "proposal_file": None,        # {name, bytes}
    "proposal_hash": None,
    "proposal_pages": None,
    "proposal_chars": 0,
    "proposal_truncated": False,
    "analysis_at": None,
    "analysis_nonce": 0,
    "proposal_version": 0,
    "include_proposal_context": True,
    "last_failed_prompt": None,
    "questions_asked": 0,
}
for _key, _value in DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

# Cek ketersediaan embeddings
if not EMBEDDINGS_AVAILABLE:
    st.markdown(
        '<div class="error-box">⚠️ <strong>Error Kritis:</strong> Package <code>langchain-huggingface</code> tidak ditemukan. Aplikasi tidak dapat berjalan. Silakan pastikan semua dependencies terinstal dengan benar.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_INDONESIA = """\
Anda adalah seorang Ahli ENTREPRENEURSHIP yang KREATIF dan berpengalaman lebih dari 25 tahun. Gunakan informasi konteks berikut untuk menjawab berbagai pertanyaan pengguna dalam bahasa Indonesia yang baik dan terstruktur.
Selalu berikan jawaban terbaik yang dapat kamu berikan dengan tone memotivasi dengan gaya santai dan informal tapi tetap santun.
Jika pengguna melampirkan proposal bisnis, gunakan ringkasan analisis proposal di bawah sebagai acuan tambahan saat menjawab pertanyaan lanjutan tentang proposal tersebut.

ATURAN SUMBER:
- Jawablah HANYA berdasarkan konteks dokumen referensi dan konteks analisis proposal yang diberikan.
- Jika informasi yang ditanyakan tidak ada di konteks, katakan terus terang bahwa informasinya tidak ditemukan pada dokumen.
- Saran umum di luar dokumen boleh diberikan, tetapi beri label jelas bahwa itu di luar dokumen.
- Jangan mengarang fakta, angka, kutipan, atau nama sumber.

Konteks dokumen referensi:
{context}

Konteks analisis proposal pengguna:
{proposal_context}

Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "proposal_context", "chat_history", "question"],
    template=PROMPT_INDONESIA,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROMPT UNTUK ANALISIS PROPOSAL BISNIS
# ─────────────────────────────────────────────────────────────────────────────
PROPOSAL_ANALYSIS_PROMPT = """\
Anda adalah seorang ahli bisnis dan investor profesional. Analisis proposal bisnis berikut dan berikan:

1. **Ringkasan Proposal (5-9 kalimat)**: Berikan gambaran komprehensif tentang ide bisnis, model bisnis, target pasar, dan rencana implementasi.
2. **Potensi Keuntungan**: Sebutkan hal-hal yang membuat proposal ini menarik atau potensial menguntungkan.
3. **Risiko atau Hal yang Perlu Diwaspadai**: Sebutkan risiko utama atau kelemahan dalam proposal ini.
4. **Pertanyaan Investor (3-5 pertanyaan)**: Buat daftar pertanyaan penting yang harus diajukan oleh investor kepada pengusul bisnis.

Proposal Bisnis:
{proposal_text}

Jawaban:
"""

# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNGSI BANTU (EKSTRAKSI PDF, ANALISIS, TRANSCRIPT)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    """Ekstrak teks PDF langsung dari memori (tanpa file sementara)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip(), len(reader.pages)


@st.cache_resource(show_spinner=False)
def get_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 60.0,
) -> ChatGroq:
    """Cache instance LLM per konfigurasi agar tidak dibuat ulang tiap rerun."""
    return ChatGroq(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=2,
    )


@st.cache_data(show_spinner=False)
def analyze_proposal_text(proposal_text: str, nonce: int) -> str:
    """
    Analisis proposal. `nonce` membuat cache bisa dipaksa lewat tombol Refresh.
    Hasil di-cache per teks proposal agar tidak diulang saat user chat.
    """
    llm = get_llm(MODEL_PROPOSAL, TEMP_PROPOSAL, MAX_TOKENS_PROPOSAL)
    prompt = PROPOSAL_ANALYSIS_PROMPT.format(proposal_text=proposal_text)
    return llm.invoke(prompt).content


def run_proposal_analysis() -> None:
    """Jalankan analisis untuk lampiran yang tersimpan di session_state."""
    proposal = st.session_state.proposal_file
    if not proposal:
        return

    st.session_state.proposal_error = None
    try:
        text, pages = extract_pdf_text(proposal["bytes"])
        st.session_state.proposal_pages = pages
        st.session_state.proposal_chars = len(text)

        if not text.strip():
            raise ValueError(
                "Tidak ada teks yang bisa dibaca dari PDF ini (kemungkinan hasil scan/gambar). "
                "Gunakan PDF berbasis teks atau jalankan OCR terlebih dahulu."
            )

        truncated = text[:MAX_PROPOSAL_CHARS]
        st.session_state.proposal_truncated = len(text) > MAX_PROPOSAL_CHARS
        with st.spinner("🧠 Mentor sedang menganalisis proposal Anda dengan seksama..."):
            st.session_state.proposal_analysis = analyze_proposal_text(
                truncated, st.session_state.analysis_nonce
            )
        st.session_state.analysis_at = datetime.now().isoformat(timespec="seconds")
    except Exception as exc:
        st.session_state.proposal_analysis = None
        st.session_state.proposal_error = str(exc)


def build_proposal_context() -> str:
    """
    Ringkas hasil analisis proposal untuk dijadikan konteks chat.
    Dibatasi agar prompt tidak membengkak dan biaya token tetap terkendali.
    """
    analysis = st.session_state.proposal_analysis
    if not st.session_state.include_proposal_context or not analysis:
        return "(tidak ada proposal bisnis yang dilampirkan)"

    proposal = st.session_state.proposal_file
    header = (
        f"File: {proposal['name']} ({st.session_state.proposal_pages or '?'} halaman)"
        if proposal
        else ""
    )

    analysis = analysis.strip()
    if len(analysis) > MAX_PROPOSAL_CONTEXT_CHARS:
        analysis = (
            analysis[:MAX_PROPOSAL_CONTEXT_CHARS].rstrip()
            + "\n[... ringkasan analisis dipotong karena batas konteks ...]"
        )

    return "\n".join(part for part in (header, analysis) if part)


def build_chat_history_pairs(history: list[dict]) -> list[tuple[str, str]]:
    """
    Ubah riwayat chat menjadi pasangan (pertanyaan, jawaban) yang benar untuk
    ConversationalRetrievalChain, sekaligus membatasi panjang konteksnya.
    """
    messages = [m for m in history if m.get("role") in ("user", "assistant")]
    messages = messages[-(MAX_HISTORY_MESSAGES * 2):]

    pairs: list[tuple[str, str]] = []
    total_chars = 0
    i = len(messages) - 1
    # Bangun dari belakang agar pesan terbaru selalu ikut
    while i >= 0:
        answer = messages[i] if messages[i]["role"] == "assistant" else None
        question = messages[i - 1] if i >= 1 and messages[i - 1]["role"] == "user" else None
        if question is None and answer is None:
            i -= 1
            continue
        # Jika pesan terakhir adalah pertanyaan tanpa jawaban, lewati
        if answer is None:
            i -= 1
            continue
        q_text = question["content"] if question else "(tanpa pertanyaan)"
        pair_chars = len(q_text) + len(answer["content"])
        if total_chars + pair_chars > MAX_HISTORY_CHARS:
            break
        pairs.insert(0, (q_text, answer["content"]))
        total_chars += pair_chars
        i -= 2 if question else 1

    return pairs


def format_sources(documents) -> list[dict]:
    """Ringkas dokumen sumber agar bisa ditampilkan dan diekspor."""
    sources: list[dict] = []
    for doc in documents or []:
        metadata = getattr(doc, "metadata", {}) or {}
        filename = Path(str(metadata.get("source", "?"))).name
        page = metadata.get("page")
        snippet = " ".join(getattr(doc, "page_content", "").split())[:240]
        sources.append(
            {
                "file": filename,
                "page": (int(page) + 1) if isinstance(page, int) else None,
                "snippet": snippet,
            }
        )
    return sources


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def transcript_available() -> bool:
    return bool(st.session_state.chat_history or st.session_state.proposal_analysis)


TRANSCRIPT_CSS = """
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --brand:#2563eb; --brand-2:#7c3aed; --bg:#eef2f7;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  line-height:1.7;font-size:16px}
.page{max-width:880px;margin:0 auto;padding:32px 20px 56px}
.hero{background:linear-gradient(135deg,var(--brand),var(--brand-2));color:#fff;
  padding:30px 32px;border-radius:20px;box-shadow:0 16px 40px rgba(37,99,235,.28)}
.hero .top{display:flex;justify-content:space-between;gap:12px;align-items:center;
  font-size:13px;letter-spacing:.02em;opacity:.9;flex-wrap:wrap}
.hero h1{margin:14px 0 6px;font-size:28px;line-height:1.25;font-weight:800}
.hero p{margin:0;opacity:.92;font-size:15px}
.chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0 26px}
.chip{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.chip span{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.chip strong{display:block;margin-top:2px;font-size:14px;word-break:break-word}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:24px 26px;margin:0 0 26px}
.card>h2{margin:0 0 14px;font-size:18px}
.chat>h2{font-size:18px;margin:0 0 16px}
.msg{margin:0 0 18px;display:flex;flex-direction:column;max-width:92%}
.msg.user{margin-left:auto;align-items:flex-end}
.msg.mentor{margin-right:auto;align-items:flex-start}
.who{font-size:12px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;
  color:var(--muted);margin:0 6px 6px}
.bubble{padding:16px 20px;border-radius:16px;width:100%}
.msg.user .bubble{background:#dbeafe;border:1px solid #bfdbfe;border-bottom-right-radius:4px}
.msg.mentor .bubble{background:#fff;border:1px solid var(--line);
  border-bottom-left-radius:4px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.md>*:first-child{margin-top:0}
.md>*:last-child{margin-bottom:0}
.md h1,.md h2,.md h3,.md h4{line-height:1.35;margin:18px 0 8px}
.md h1{font-size:20px}
.md h2{font-size:18px}
.md h3{font-size:16px}
.md p{margin:9px 0}
.md ul,.md ol{margin:9px 0;padding-left:22px}
.md li{margin:4px 0}
.md strong{font-weight:700}
.md a{color:var(--brand);word-break:break-word}
.md code{background:#f1f5f9;border:1px solid var(--line);border-radius:6px;
  padding:1px 5px;font-size:.9em}
.md pre{background:#0f172a;color:#e2e8f0;border-radius:12px;padding:14px 16px;overflow:auto}
.md pre code{background:none;border:0;color:inherit;padding:0}
.md blockquote{margin:10px 0;padding:8px 14px;border-left:3px solid var(--brand);
  background:#f8fafc;color:#334155}
.md table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
.md th,.md td{border:1px solid var(--line);padding:8px 10px;text-align:left}
.md th{background:#f8fafc}
.sources{margin:8px 6px 0;font-size:13px;color:#334155}
.sources summary{cursor:pointer;font-weight:600;color:var(--brand)}
.sources ul{margin:8px 0 0;padding-left:20px}
.sources li{margin:4px 0}
.empty{color:var(--muted);font-style:italic}
footer{margin-top:8px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px}
footer strong{color:#334155}
@media print{
  body{background:#fff}
  .page{max-width:none;padding:0}
  .hero{border-radius:0;box-shadow:none;padding:22px 24px}
  .card,.bubble{box-shadow:none}
  .msg{max-width:100%}
  footer{page-break-inside:avoid}
}
@page{margin:14mm}
"""


def _md_to_html(text: str) -> str:
    """Konversi markdown ke HTML; '<' di-escape agar HTML mentah tidak ikut tersisip."""
    escaped = (text or "").strip().replace("<", "&lt;")
    if not escaped:
        return ""
    try:
        import markdown as _markdown

        return _markdown.markdown(
            escaped, extensions=["extra", "sane_lists", "nl2br"]
        )
    except Exception:
        return "<p>" + escaped.replace("\n", "<br>") + "</p>"


def build_transcript_html() -> str:
    """Bangun transkrip HTML mandiri (tanpa aset eksternal) untuk diunduh."""
    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    proposal = st.session_state.proposal_file
    history = st.session_state.chat_history

    chips = [
        ("Lampiran", html.escape(proposal["name"]) if proposal else "Tidak ada"),
        ("Ukuran", human_size(len(proposal["bytes"])) if proposal else "-"),
        ("Halaman", str(st.session_state.proposal_pages or "-")),
        ("Jumlah pesan", str(len(history))),
    ]
    chips_html = "".join(
        f'<div class="chip"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in chips
    )

    analysis_html = ""
    if st.session_state.proposal_analysis:
        analysis_html = (
            '<section class="card"><h2>🔍 Analisis Proposal Bisnis</h2>'
            f'<div class="md">{_md_to_html(st.session_state.proposal_analysis)}</div>'
            "</section>"
        )

    messages_html = []
    for message in history:
        is_user = message.get("role") == "user"
        label = "Anda" if is_user else "Mentor (TEDIWIRA)"
        cls = "user" if is_user else "mentor"
        sources_html = ""
        sources = message.get("sources") or []
        if sources:
            items = "".join(
                "<li><strong>{}</strong>{}<br><span>{}</span></li>".format(
                    html.escape(str(source.get("file", "?"))),
                    f" · hal. {source['page']}" if source.get("page") else "",
                    html.escape(str(source.get("snippet", ""))),
                )
                for source in sources
            )
            sources_html = (
                f'<details open class="sources"><summary>📚 Sumber ({len(sources)})</summary>'
                f"<ul>{items}</ul></details>"
            )
        messages_html.append(
            f'<article class="msg {cls}"><div class="who">{label}</div>'
            f'<div class="bubble md">{_md_to_html(message.get("content", ""))}</div>'
            f"{sources_html}</article>"
        )
    if not messages_html:
        messages_html.append('<p class="empty">Belum ada percakapan.</p>')

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transkrip Diskusi Kewirausahaan — TEDIWIRA</title>
<style>
{TRANSCRIPT_CSS}
</style>
</head>
<body>
<main class="page">
  <header class="hero">
    <div class="top"><span>📓 TEDIWIRA</span><span>{now}</span></div>
    <h1>Transkrip Diskusi Kewirausahaan</h1>
    <p>Teman Diskusi Kewirausahaan · diekspor otomatis dari aplikasi</p>
  </header>
  <section class="chips">{chips_html}</section>
  {analysis_html}
  <section class="chat">
    <h2>💬 Percakapan</h2>
    {"".join(messages_html)}
  </section>
  <footer>
    <strong>Disklaimer:</strong> Jawaban dihasilkan oleh AI dan dapat tidak akurat.
    Verifikasi informasi penting dengan sumber terpercaya sebelum mengambil keputusan bisnis.
  </footer>
</main>
</body>
</html>"""


def reset_proposal_state() -> None:
    """Callback: hapus lampiran + analisis dan reset widget uploader."""
    st.session_state.proposal_version += 1
    st.session_state.proposal_file = None
    st.session_state.proposal_hash = None
    st.session_state.proposal_analysis = None
    st.session_state.proposal_error = None
    st.session_state.proposal_pages = None
    st.session_state.proposal_chars = 0
    st.session_state.proposal_truncated = False
    st.session_state.analysis_at = None
    st.session_state.analysis_nonce += 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. FUNGSI INISIALISASI RAG
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_rag():
    """
    Memuat dokumen PDF dari folder 'documents', memecah menjadi chunk,
    membuat/memuat FAISS vector store, dan membentuk ConversationalRetrievalChain.
    Index disimpan di disk agar startup berikutnya lebih cepat.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    if VECTORSTORE_DIR.exists():
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
        )
    else:
        loader = DirectoryLoader(
            str(DOCUMENTS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=150
        )
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(str(VECTORSTORE_DIR))

    llm = get_llm(MODEL_CHAT, TEMP_CHAT, MAX_TOKENS_CHAT)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": INDO_PROMPT_TEMPLATE,
            "output_key": "answer",
        },
    )


def ensure_chain() -> None:
    """Pastikan chain RAG siap; tampilkan error yang bisa dicoba ulang."""
    if st.session_state.chain is not None:
        return

    try:
        with st.spinner(
            "🧠 Memuat basis pengetahuan... (startup pertama bisa beberapa menit)"
        ):
            st.session_state.chain = initialize_rag()
        st.session_state.chain_error = None
    except Exception as exc:
        st.session_state.chain = None
        st.session_state.chain_error = str(exc)


def ask_mentor(question: str) -> tuple[str, list[dict]]:
    """Kirim pertanyaan + riwayat + konteks proposal ke chain RAG."""
    payload = {
        "question": question,
        "chat_history": build_chat_history_pairs(st.session_state.chat_history[:-1]),
        "proposal_context": build_proposal_context(),
    }
    chain = st.session_state.chain
    result = (
        chain.invoke(payload)
        if hasattr(chain, "invoke")
        else chain(payload)  # kompatibilitas chain lama
    )
    answer = str(result.get("answer", "")).strip()
    if not answer:
        answer = "Maaf, saya belum menemukan jawaban yang relevan. Coba perjelas pertanyaannya."
    return answer, format_sources(result.get("source_documents"))


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📚 Sumber ({len(message['sources'])})"):
                for source in message["sources"]:
                    page = f" · hal. {source['page']}" if source.get("page") else ""
                    st.markdown(f"**{source['file']}**{page}")
                    if source.get("snippet"):
                        st.caption(source["snippet"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR: UPLOAD, STATUS LAMPIRAN, REFRESH & RESET
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Review Proposal Bisnis")
    st.markdown("Unggah file PDF proposal bisnis untuk dianalisis oleh mentor AI.")

    uploaded_file = st.file_uploader(
        "📁 Pilih file PDF",
        type="pdf",
        key=f"proposal_uploader_v{st.session_state.proposal_version}",
    )

    # Analisis otomatis HANYA saat file baru diunggah (bukan tiap rerun/chat).
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash != st.session_state.proposal_hash:
            st.session_state.proposal_hash = file_hash
            st.session_state.proposal_file = {
                "name": uploaded_file.name,
                "bytes": file_bytes,
            }
            st.session_state.proposal_analysis = None
            st.session_state.proposal_error = None
            st.session_state.analysis_nonce += 1
            run_proposal_analysis()

    # Status lampiran
    if st.session_state.proposal_file:
        proposal = st.session_state.proposal_file
        pages = st.session_state.proposal_pages or "-"
        st.markdown(
            f"""
            <div class="attach-box">
                📎 <strong>{html.escape(proposal['name'])}</strong><br>
                Ukuran: {human_size(len(proposal['bytes']))} · Halaman: {pages}<br>
                Status: {"✅ sudah dianalisis" if st.session_state.proposal_analysis else "⏳ belum dianalisis"}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.checkbox(
        "🧠 Sertakan analisis proposal di konteks chat",
        key="include_proposal_context",
        disabled=not st.session_state.proposal_analysis,
        help=(
            "Jika aktif, mentor mengingat hasil analisis proposal saat menjawab "
            "pertanyaan lanjutan. Matikan untuk menghemat token."
        ),
    )

    col_refresh, col_reset = st.columns(2)
    with col_refresh:
        refresh_all = st.button(
            "🔄 Refresh",
            help="Analisis ulang lampiran & muat ulang tampilan obrolan.",
            use_container_width=True,
            type="primary",
        )
    with col_reset:
        st.button(
            "🗑️ Hapus",
            help="Hapus lampiran dan hasil analisisnya.",
            on_click=reset_proposal_state,
            use_container_width=True,
        )

    if refresh_all:
        st.cache_data.clear()  # buang cache teks & analisis
        st.session_state.analysis_nonce += 1
        if st.session_state.proposal_file:
            run_proposal_analysis()
        st.toast("Obrolan dan lampiran dimuat ulang ✅")
        st.rerun()

    if st.session_state.proposal_error:
        st.error(f"Analisis gagal: {st.session_state.proposal_error}")
        if st.button("🔁 Coba analisis lagi", use_container_width=True):
            st.cache_data.clear()
            st.session_state.analysis_nonce += 1
            run_proposal_analysis()
            st.rerun()

    st.markdown("---")
    st.subheader("📚 Basis Pengetahuan")
    pdf_count = len(list(DOCUMENTS_DIR.glob("**/*.pdf"))) if DOCUMENTS_DIR.exists() else 0
    st.caption(
        f"{pdf_count} dokumen PDF · index FAISS: "
        + ("tersedia" if VECTORSTORE_DIR.exists() else "belum dibuat")
    )
    if st.button(
        "♻️ Muat ulang basis pengetahuan",
        help="Muat ulang model & index (gunakan setelah dokumen diperbarui).",
        use_container_width=True,
    ):
        initialize_rag.clear()
        st.session_state.chain = None
        st.session_state.chain_error = None
        st.toast("Basis pengetahuan dimuat ulang")
        st.rerun()

    if st.button(
        "🧹 Mulai percakapan baru",
        help="Hapus riwayat obrolan (lampiran dan kuota pertanyaan tetap).",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.session_state.last_failed_prompt = None
        st.toast("Riwayat obrolan dibersihkan")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 8. HASIL ANALISIS PROPOSAL
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.proposal_analysis:
    head_left, head_right = st.columns([4, 1])
    with head_left:
        st.subheader("🔍 Hasil Analisis Proposal Bisnis")
        if st.session_state.analysis_at:
            st.caption(
                "Dianalisis pada "
                + datetime.fromisoformat(st.session_state.analysis_at).strftime(
                    "%d-%m-%Y %H:%M"
                )
            )
    with head_right:
        if st.button("🧠 Analisis ulang", use_container_width=True):
            st.cache_data.clear()
            st.session_state.analysis_nonce += 1
            run_proposal_analysis()
            st.rerun()

    if st.session_state.proposal_truncated:
        st.warning(
            f"Teks proposal lebih dari {MAX_PROPOSAL_CHARS:,} karakter, jadi hanya bagian awal "
            "yang dianalisis. Untuk analisis menyeluruh, unggah proposal yang lebih ringkas "
            "atau pecah per bab."
        )

    with st.container(border=True):
        st.markdown(st.session_state.proposal_analysis)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 9. INISIALISASI SISTEM
# ─────────────────────────────────────────────────────────────────────────────
ensure_chain()

chain_ready = st.session_state.chain is not None
if st.session_state.chain_error:
    st.error(f"Gagal memuat basis pengetahuan: {st.session_state.chain_error}")
    if st.button("🔁 Coba muat ulang"):
        initialize_rag.clear()
        st.session_state.chain = None
        st.session_state.chain_error = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 10. ANTARMUKA CHAT
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💬 Diskusi Interaktif dengan **TEDIWIRA**")
st.markdown(
    "*Ajukan pertanyaan tentang kewirausahaan, ide bisnis, strategi, atau tantangan yang Anda hadapi*"
)
if st.session_state.proposal_analysis and st.session_state.include_proposal_context:
    st.caption(
        "🧠 Konteks aktif: jawaban mentor mempertimbangkan hasil analisis proposal "
        "yang Anda lampirkan."
    )

# 10.1 Riwayat chat
for message in st.session_state.chat_history:
    render_message(message)

# 10.2 Retry jika permintaan terakhir gagal
if st.session_state.last_failed_prompt and chain_ready:
    st.warning("Jawaban terakhir gagal dibuat.")
    if st.button("🔁 Coba jawab ulang pertanyaan terakhir"):
        question = st.session_state.last_failed_prompt
        with st.chat_message("assistant"):
            try:
                answer, sources = ask_mentor(question)
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                st.session_state.last_failed_prompt = None
                st.rerun()
            except Exception as exc:
                st.error(f"Maaf, terjadi kesalahan: {exc}")

# 10.3 Chat input
if chain_ready:
    questions_left = MAX_QUESTIONS - st.session_state.questions_asked
    if questions_left <= 0:
        st.warning(
            f"Batas {MAX_QUESTIONS} pertanyaan per sesi sudah tercapai. "
            "Unduh transkrip dulu bila perlu, lalu muat ulang halaman untuk memulai sesi baru."
        )
    elif questions_left <= 3:
        st.caption(
            f"ℹ️ Sisa {questions_left} dari {MAX_QUESTIONS} pertanyaan untuk sesi ini."
        )

    prompt = st.chat_input(
        "✍️tuliskan pertanyaan Anda tentang KEWIRAUSAHAAN disini",
        disabled=questions_left <= 0,
    )
    if prompt and questions_left > 0:
        st.session_state.questions_asked += 1
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.last_failed_prompt = None
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Mencari referensi & menyusun jawaban..."):
                try:
                    answer, sources = ask_mentor(prompt)
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.session_state.last_failed_prompt = prompt
                    st.error(f"Maaf, terjadi kesalahan: {exc}")

        if st.session_state.questions_asked >= MAX_QUESTIONS:
            st.rerun()

# 10.4 Unduh transkrip di akhir percakapan agar isinya selalu terbaru
if transcript_available():
    st.markdown("")
    col_spacer, col_download = st.columns([3, 1])
    with col_download:
        st.download_button(
            "⬇️ Unduh chat (.html)",
            data=build_transcript_html(),
            file_name=f"transkrip_tediwira_{datetime.now():%Y%m%d_%H%M}.html",
            mime="text/html",
            use_container_width=True,
            help="Menyimpan seluruh percakapan beserta hasil analisis proposal ke file HTML.",
        )

# ─────────────────────────────────────────────────────────────────────────────
# 11. FOOTER & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **Disclaimer:**
    - Sistem ini menggunakan **AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.**
    - Jawaban chat memakai referensi dari dokumen di folder `documents`; sumber tiap jawaban bisa dilihat pada bagian **📚 Sumber**.
    - Ketik: LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya.
    """
)
