# TEDI-WIRA
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI API & HALAMAN
# ─────────────────────────────────────────────────────────────────────────────

# Groq_API KEY
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="TEDIWIRA",
    page_icon="📓",
    layout="wide"
)

# CSS Styling
st.markdown(
    """
    <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #f0f2f6; }
        .bot-message { background-color: #e8f0fe; }
        .analysis-box { background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; margin: 1rem 0; }
        .startup-info { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("📓TEMAN DISKUSI KEWIRAUSAHAAN")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan Tentang Kewirausahaan
    **Chat Bot ini akan membantu Anda memahami lebih dalam tentang dunia KEWIRAUSAHAAN** dan berbagai hal-hal yang perlu di perhatikan baik pada masa persiapan, pelaksanaan, pengembangan,dan bahkan exit strategy. **Pastikanlah Anda memiliki koneksi internet yang baik dan stabil.**
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE DAN INISIALISASI
# ─────────────────────────────────────────────────────────────────────────────
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'proposal_analysis' not in st.session_state:
    st.session_state.proposal_analysis = None

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_INDONESIA = """\
Anda adalah seorang Ahli ENTREPRENEURSHIP yang KREATIF dan berpengalaman lebih dari 25 tahun . Gunakan informasi konteks berikut untuk menjawab berbagai pertanyaan pengguna dalam bahasa Indonesia yang baik dan terstruktur.
Selalu berikan jawaban terbaik yang dapat kamu berikan dengan tone memotivasi dengan gaya santai dan informal tapi tetap santun.

Konteks: {context}
Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_INDONESIA
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNGSI UNTUK REVIEW PROPOSAL BISNIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_business_proposal(pdf_file):
    """
    Menganalisis proposal bisnis dari file PDF dan memberikan ringkasan, keuntungan,
    risiko, serta pertanyaan lanjutan untuk investor.
    """
    try:
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Gabungkan semua teks
        full_text = "\n".join([doc.page_content for doc in documents])

        # Prompt untuk analisis proposal
        analysis_prompt = f"""
        Anda adalah seorang ahli bisnis dan investor profesional. Analisis proposal bisnis berikut dan berikan:

        1. **Ringkasan Proposal (5-9 kalimat)**: Berikan gambaran komprehensif tentang ide bisnis, model bisnis, target pasar, dan rencana implementasi.
        2. **Potensi Keuntungan**: Sebutkan hal-hal yang membuat proposal ini menarik atau potensial menguntungkan.
        3. **Risiko atau Hal yang Perlu Diwaspadai**: Sebutkan risiko utama atau kelemahan dalam proposal ini.
        4. **Pertanyaan Investor (3-5 pertanyaan)**: Buat daftar pertanyaan penting yang harus diajukan oleh investor kepada pengusul bisnis.

        Proposal Bisnis:
        {full_text}

        Jawaban:
        """

        # Inisialisasi LLM
        llm = ChatGroq(
            temperature=0.3,
            model_name="gemma2-9b-it",
            max_tokens=2048
        )

        # Dapatkan jawaban
        response = llm.invoke(analysis_prompt)
        
        # Hapus file sementara
        os.unlink(tmp_file_path)
        
        return response.content

    except Exception as e:
        # Hapus file sementara jika ada error
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        return f"Terjadi kesalahan saat menganalisis proposal: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNGSI INISIALISASI RAG DENGAN OPTIMASI
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=3600, show_spinner=False)  # Cache selama 1 jam
def initialize_rag():
    """
    Memuat dokumen PDF dari folder 'documents', memecah menjadi chunk,
    membuat FAISS vector store, dan membentuk ConversationalRetrievalChain.
    Menggunakan persistence untuk mempercepat startup.
    """
    try:
        # Path untuk menyimpan vector store
        vectorstore_path = "faiss_index"
        
        # Cek apakah vector store sudah ada
        if os.path.exists(vectorstore_path):
            # Load vector store yang sudah ada
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}  
            )
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            st.info("✅ Vector store dimuat dari cache disk")
        else:
            # Tampilkan info startup
            startup_placeholder = st.empty()
            startup_placeholder.markdown(
                '<div class="startup-info">🚀 Pertama kali startup: Membuat vector store dari dokumen... Ini mungkin memakan waktu beberapa menit.</div>', 
                unsafe_allow_html=True
            )
            
            # 1. Load Dokumen PDF
            loader = DirectoryLoader("documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()

            # 2. Split Dokumen (dioptimasi)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            texts = text_splitter.split_documents(documents)

            # 3. Embedding (model yang lebih cepat)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}  
            )

            # 4. Membuat Vector Store FAISS
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # 5. Simpan vector store ke disk
            vectorstore.save_local(vectorstore_path)
            
            # Hapus info startup
            startup_placeholder.empty()
            st.success("✅ Vector store berhasil dibuat dan disimpan!")

        # 6. Menginisialisasi LLM (ChatGroq)
        llm = ChatGroq(
            temperature=0.45,
            model_name="gemma2-9b-it",
            max_tokens=2048
        )

        # 7. Membuat Memory
        memory = ConversationBufferWindowMemory(
            k=3,
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        # 8. Membuat Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                'prompt': INDO_PROMPT_TEMPLATE,
                'output_key': 'answer'
            }
        )

        return chain

    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR UNTUK UPLOAD DAN ANALISIS PROPOSAL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Review Proposal Bisnis")
    st.markdown("Upload file PDF proposal bisnis untuk dianalisis.")
    uploaded_file = st.file_uploader("📁 Pilih file PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Menganalisis proposal..."):
            summary = analyze_business_proposal(uploaded_file)
            st.session_state.proposal_analysis = summary

# ─────────────────────────────────────────────────────────────────────────────
# 7. TAMPILKAN HASIL ANALISIS PROPOSAL (JIKA ADA)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.proposal_analysis:
    st.subheader("🔍 Hasil Analisis Proposal Bisnis")
    st.markdown(f'<div class="analysis-box">{st.session_state.proposal_analysis}</div>', unsafe_allow_html=True)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 8. INISIALISASI SISTEM
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain is None:
    with st.spinner("Memuat sistem..."):
        st.session_state.chain = initialize_rag()

# ─────────────────────────────────────────────────────────────────────────────
# 9. ANTARMUKA CHAT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain:
    # 9.1 Tampilkan riwayat chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 9.2 Chat Input
    prompt = st.chat_input("✍️tuliskan pertanyaan Anda tentang KEWIRAUSAHAAN disini")
    if prompt:
        # Tambahkan pertanyaan user ke riwayat chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 9.3 Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    # Panggil chain
                    result = st.session_state.chain({"question": prompt})
                    # Ambil jawaban
                    answer = result.get('answer', '')
                    st.write(answer)
                    # Tambahkan ke riwayat
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# ─────────────────────────────────────────────────────────────────────────────
# 10. FOOTER & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **Disclaimer:**
    - Sistem ini menggunakan **AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.**
    - Ketik: LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya.
    """
)
