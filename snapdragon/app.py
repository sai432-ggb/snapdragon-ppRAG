import os
import json
import math
import hashlib
import numpy as np
import streamlit as st
from cryptography.fernet import Fernet
import urllib.request

st.set_page_config(page_title="ppRAG - Privacy-Preserving RAG Dashboard", layout="wide")


class CAPRISEEngine:
    def __init__(self, s: float = 3.0, beta: float = 0.2, key: bytes = None):
        self.s = s
        self.beta = beta
        self.K = key if key is not None else os.urandom(32)

    def _prf_noise(self, vector_dim: int, seed_bytes: bytes, max_norm: float) -> np.ndarray:
        h = hashlib.sha256(self.K + seed_bytes).digest()
        seed_int = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed_int)
        n = rng.normal(0, 1, size=vector_dim)
        n_norm = np.linalg.norm(n)
        unit_direction = n / (n_norm if n_norm > 0 else 1.0)
        u = rng.uniform(0, 1)
        scaling = (u ** (1.0 / vector_dim)) * max_norm * 0.99
        return unit_direction * scaling

    def enc_db(self, e_i: np.ndarray) -> tuple[np.ndarray, bytes]:
        r = os.urandom(16)
        max_norm = (3.0 * self.s * self.beta) / 8.0
        lambda_e = self._prf_noise(len(e_i), r, max_norm)
        return self.s * e_i + lambda_e, r

    def dec_db(self, e_prime: np.ndarray, r: bytes) -> np.ndarray:
        max_norm = (3.0 * self.s * self.beta) / 8.0
        lambda_e = self._prf_noise(len(e_prime), r, max_norm)
        return (e_prime - lambda_e) / self.s

    def enc_q(self, e_q_tilde: np.ndarray) -> tuple[np.ndarray, bytes]:
        r = os.urandom(16)
        max_norm = (self.s * self.beta) / 8.0
        eta_q = self._prf_noise(len(e_q_tilde), r, max_norm)
        return self.s * e_q_tilde + eta_q, r


class DistanceDP:
    def __init__(self, perturbation_radius: float = 0.04):
        self.r = perturbation_radius

    def perturb_query(self, e_q: np.ndarray, k: int, total_db_size: int) -> tuple[np.ndarray, int]:
        dim = len(e_q)
        noise = np.random.normal(0, 1, size=dim)
        noise_norm = np.linalg.norm(noise)
        if noise_norm > 0:
            noise = (noise / noise_norm) * self.r
        e_q_tilde = e_q + noise
        norm_tilde = np.linalg.norm(e_q_tilde)
        if norm_tilde > 0:
            e_q_tilde /= norm_tilde
        k_prime = min(total_db_size, int(math.ceil(k * (1.5 + self.r * 5.0))))
        return e_q_tilde, max(k, k_prime)


class LocalEmbedding:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in words:
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            for i in range(4):
                idx = (h >> (i * 8)) % self.dim
                sign = 1.0 if ((h >> (i * 4)) & 1) == 0 else -1.0
                vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# Session state initialization
if "fernet_key" not in st.session_state:
    st.session_state.fernet_key = Fernet.generate_key()
    st.session_state.cipher_suite = Fernet(st.session_state.fernet_key)
    st.session_state.embedder = LocalEmbedding(dim=128)
    st.session_state.cloud_db = []

# Sidebar settings
st.sidebar.title("ppRAG Settings")
param_s = st.sidebar.slider("Scaling Factor (s)", 1.0, 10.0, 3.0, 0.5)
param_beta = st.sidebar.slider("Preservation Bound (β)", 0.05, 0.5, 0.2, 0.05)
param_dp_r = st.sidebar.slider("Query Perturbation Radius (r)", 0.01, 0.20, 0.04, 0.01)
ollama_url = st.sidebar.text_input("Ollama Endpoint", "http://localhost:11434/api/generate")
model_name = st.sidebar.text_input("Model Name", "qwen3-vl:4b-instruct")

caprise = CAPRISEEngine(s=param_s, beta=param_beta)
dp_engine = DistanceDP(perturbation_radius=param_dp_r)

st.title("ppRAG: Secure Cloud Vector Search & Local LLM Dashboard")

tab1, tab2, tab3 = st.tabs(["1. Document Ingestion", "2. Secure Vector Search", "3. Local LLM Generation"])

# Tab 1: Document ingestion
with tab1:
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.subheader("Local Client Input")
        doc_id_input = st.text_input("Document ID", f"doc_{len(st.session_state.cloud_db)+1}")
        doc_text_input = st.text_area("Plaintext Document Content", "CAPRISE encrypts vector embeddings while preserving query-to-database distance orderings.")
        if st.button("Encrypt & Upload to Cloud Store"):
            if doc_text_input.strip():
                ciphertext = st.session_state.cipher_suite.encrypt(doc_text_input.encode('utf-8'))
                e_i = st.session_state.embedder.embed(doc_text_input)
                e_prime_i, r_val = caprise.enc_db(e_i)
                st.session_state.cloud_db.append({
                    "id": doc_id_input,
                    "e_prime": e_prime_i,
                    "r": r_val,
                    "ciphertext": ciphertext,
                    "raw_text": doc_text_input
                })
                st.success(f"Document '{doc_id_input}' encrypted and uploaded!")

    with col_in2:
        st.subheader("Cloud Server View (Untrusted)")
        if st.session_state.cloud_db:
            for rec in st.session_state.cloud_db:
                with st.expander(f"Cloud Record: {rec['id']}"):
                    st.text(f"AES Ciphertext:\n{rec['ciphertext'][:40].decode('utf-8', errors='ignore')}...")
                    st.text(f"CAPRISE Vector Head:\n{np.round(rec['e_prime'][:5], 4)}")

# Tab 2: Secure search
with tab2:
    query_text = st.text_input("User Search Query", "How does CAPRISE protect vector embeddings?")
    target_k = st.number_input("Target Top-k", min_value=1, max_value=10, value=2)
    if st.button("Run Secure Search"):
        if st.session_state.cloud_db:
            e_q = st.session_state.embedder.embed(query_text)
            e_q_tilde, k_prime = dp_engine.perturb_query(e_q, target_k, len(st.session_state.cloud_db))
            e_prime_q, _ = caprise.enc_q(e_q_tilde)

            cloud_results = []
            for rec in st.session_state.cloud_db:
                dist = float(np.linalg.norm(e_prime_q - rec["e_prime"]))
                cloud_results.append({
                    "id": rec["id"],
                    "cloud_dist": dist,
                    "e_prime": rec["e_prime"],
                    "r": rec["r"],
                    "ciphertext": rec["ciphertext"]
                })
            cloud_results.sort(key=lambda x: x["cloud_dist"])

            client_results = []
            for item in cloud_results[:k_prime]:
                dec_vec = caprise.dec_db(item["e_prime"], item["r"])
                true_dist = float(np.linalg.norm(e_q - dec_vec))
                plaintext = st.session_state.cipher_suite.decrypt(item["ciphertext"]).decode('utf-8')
                client_results.append({"id": item["id"], "true_dist": true_dist, "text": plaintext})
            client_results.sort(key=lambda x: x["true_dist"])

            final_top_k = client_results[:target_k]
            st.session_state.last_query = query_text
            st.session_state.last_retrieved = final_top_k

            for rank, item in enumerate(final_top_k, 1):
                st.success(f"Rank {rank} [{item['id']}] (Dist: {item['true_dist']:.4f})\n\nText: {item['text']}")

# Tab 3: Local LLM generation
# Tab 3: Local LLM generation
with tab3:
    if "last_retrieved" in st.session_state and st.session_state.last_retrieved:
        # Build context string from retrieved docs
        context_str = "\n".join([f"- [{doc['id']}]: {doc['text']}" for doc in st.session_state.last_retrieved])
        st.subheader("Retrieved Context")
        st.text_area("Context for LLM", context_str, height=200)

        if st.button("Send Prompt to Local Ollama"):
            payload = {
                "model": model_name,
                "prompt": f"User Query: {st.session_state.last_query}\n\nContext:\n{context_str}"
            }
            try:
                req = urllib.request.Request(
                    ollama_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req) as resp:
                    responses = []
                    for line in resp:
                        try:
                            obj = json.loads(line.decode("utf-8"))
                            if "response" in obj:
                                responses.append(obj["response"])
                        except json.JSONDecodeError:
                            continue
                    final_answer = "".join(responses)
                    st.success(f"Ollama Response:\n\n{final_answer}")
            except Exception as e:
                st.error(f"Error contacting Ollama: {e}")
    else:
        st.info("Run a secure search first to populate context.")
