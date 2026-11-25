# wirausaha — TEDIWIRA (RAG Kewirausahaan)

Chatbot mentoring kewirausahaan berbasis Streamlit + Groq dengan **RAG** (Retrieval-Augmented Generation) dari kumpulan eBook di folder `documents/`, plus review proposal bisnis PDF.

## Fitur

- Menjawab pertanyaan berdasarkan dokumen referensi (FAISS + embedding multibahasa) dan menampilkan **sumber jawaban** (file + halaman + kutipan).
- Review proposal bisnis PDF: ringkasan, potensi keuntungan, risiko, dan pertanyaan investor.
- Hasil analisis proposal otomatis dipakai sebagai konteks chat (bisa dimatikan lewat checkbox).
- Analisis proposal hanya dijalankan sekali per file (hash + cache).
- Prompt anti-halusinasi: model diminta menjawab hanya dari konteks dan mengaku bila informasi tidak ditemukan.
- Batas **7 pertanyaan per sesi browser** untuk menjaga kuota API.
- Unduh transkrip percakapan + analisis dalam format **HTML** yang rapi dan siap dicetak ke PDF.
- Tombol muat ulang basis pengetahuan tanpa menghapus riwayat chat.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
```

Buat `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_xxxxxxxx"
```

Jalankan:

```bash
streamlit run TD_WIRA.py
```

**Startup pertama** akan membaca seluruh PDF di `documents/`, memecahnya menjadi chunk, dan membuat index FAISS (bisa beberapa menit, terutama di CPU). Index disimpan di folder `faiss_index/` sehingga startup berikutnya lebih cepat.

## Struktur folder

```
TD_WIRA.py            # aplikasi utama
documents/            # PDF sumber pengetahuan
faiss_index/          # dibuat otomatis saat runtime (tidak di-commit)
requirements.txt
```

## Konfigurasi opsional (environment variable)

| Variabel | Default | Keterangan |
|---|---|---|
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-120b` | Model untuk chat RAG |
| `GROQ_PROPOSAL_MODEL` | `openai/gpt-oss-20b` | Model untuk analisis proposal |
| `GROQ_MAX_TOKENS_CHAT` | `3000` | Maksimum token jawaban chat |
| `GROQ_MAX_TOKENS_PROPOSAL` | `4096` | Maksimum token analisis proposal |
| `GROQ_TEMP_CHAT` | `0.45` | Temperature chat |
| `GROQ_TEMP_PROPOSAL` | `0.3` | Temperature analisis proposal |
| `RETRIEVER_K` | `3` | Jumlah chunk dokumen yang diambil per pertanyaan |
| `MAX_PROPOSAL_CONTEXT_CHARS` | `4000` | Panjang analisis yang dikirim ke konteks chat |

## Memperbarui basis pengetahuan

1. Tambah/ganti PDF di `documents/`.
2. Hapus folder `faiss_index/` (agar index dibangun ulang dari dokumen terbaru).
3. Klik **♻️ Muat ulang basis pengetahuan** atau jalankan ulang aplikasi.

## Catatan penting

- Percakapan hanya tersimpan di `st.session_state`. **Refresh/close tab akan menghapus riwayat** — unduh transkrip HTML bila ingin menyimpannya.
- Teks proposal, pertanyaan, dan potongan dokumen dikirim ke layanan Groq untuk diproses; tidak ada penyimpanan di server aplikasi.
- Jangan commit `.streamlit/secrets.toml` dan `faiss_index/` (sudah masuk `.gitignore`).
- Versi `streamlit`, `langchain*`, dan `markdown` dipin agar build reproducible. Library ML besar (`torch`, `transformers`, `sentence-transformers`, `faiss-cpu`) sengaja dibiarkan fleksibel mengikuti runtime.
- Untuk Streamlit Community Cloud: tambahkan `GROQ_API_KEY` melalui menu **Secrets**, lalu deploy dari repo ini. Perhatikan kebutuhan RAM karena memuat model embedding di CPU.
