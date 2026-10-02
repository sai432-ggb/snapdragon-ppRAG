import os
import json
import math
import hashlib
import numpy as np
from cryptography.fernet import Fernet
import urllib.request

class CAPRISEEncryptionEngine:
    """
    Conditional Approximate Distance-Comparison-Preserving Symmetric Encryption (CAPRISE)
    Formulae:
    - EncDB: e'_i = s * e_i + lambda_{e_i}, where ||lambda_{e_i}|| < (3 * s * beta) / 8
    - EncQ: e'_q = s * e_q_tilde + eta_{e_q}, where ||eta_{e_q}|| < (s * beta) / 8
    - DecDB: e_i = (e'_i - lambda_{e_i}) / s
    """
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


class DistanceDPPerturbation:
    """DistanceDP Differential Privacy Query Perturbation"""
    def __init__(self, perturbation_radius: float = 0.05):
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


class UntrustedCloudStore:
    """Untrusted Cloud Vector Database Store"""
    def __init__(self):
        self.records = []

    def upload_record(self, doc_id: str, e_prime: np.ndarray, r: bytes, ciphertext: bytes):
        self.records.append({"id": doc_id, "e_prime": e_prime, "r": r, "ciphertext": ciphertext})

    def search_top_k(self, e_prime_q: np.ndarray, k_prime: int) -> list[dict]:
        results = []
        for rec in self.records:
            dist = float(np.linalg.norm(e_prime_q - rec["e_prime"]))
            results.append({
                "id": rec["id"],
                "cloud_dist": dist,
                "e_prime": rec["e_prime"],
                "r": rec["r"],
                "ciphertext": rec["ciphertext"]
            })
        results.sort(key=lambda x: x["cloud_dist"])
        return results[:k_prime]


class LocalEmbeddingProvider:
    """Local Feature Embedding Provider"""
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


class CompletePPRAGMasterPipeline:
    """End-to-End Master ppRAG System Pipeline"""
    def __init__(self, llm_endpoint: str = "http://localhost:11434/api/generate"):
        self.fernet_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.fernet_key)
        self.caprise = CAPRISEEncryptionEngine(s=3.0, beta=0.2)
        self.dp = DistanceDPPerturbation(perturbation_radius=0.04)
        self.cloud = UntrustedCloudStore()
        self.embedder = LocalEmbeddingProvider(dim=128)
        self.llm_endpoint = llm_endpoint

    def phase1_ingest_document(self, doc_id: str, plaintext_content: str):
        ciphertext = self.cipher_suite.encrypt(plaintext_content.encode('utf-8'))
        e_i = self.embedder.embed(plaintext_content)
        e_prime_i, r = self.caprise.enc_db(e_i)
        self.cloud.upload_record(doc_id, e_prime_i, r, ciphertext)

    def phase2_search_and_retrieve(self, query: str, target_k: int = 2) -> list[dict]:
        e_q = self.embedder.embed(query)
        e_q_tilde, k_prime = self.dp.perturb_query(e_q, target_k, len(self.cloud.records))
        e_prime_q, _ = self.caprise.enc_q(e_q_tilde)
        candidates = self.cloud.search_top_k(e_prime_q, k_prime)
        decrypted_results = []
        for cand in candidates:
            dec_vector = self.caprise.dec_db(cand["e_prime"], cand["r"])
            true_dist = float(np.linalg.norm(e_q - dec_vector))
            plaintext = self.cipher_suite.decrypt(cand["ciphertext"]).decode('utf-8')
            decrypted_results.append({"id": cand["id"], "true_dist": true_dist, "text": plaintext})
        decrypted_results.sort(key=lambda x: x["true_dist"])
        return decrypted_results[:target_k]

    def phase3_generate_prompt(self, query: str, retrieved_docs: list[dict]) -> str:
        context_block = "\n".join([f"- [{doc['id']}]: {doc['text']}" for doc in retrieved_docs])
        return f"System Context (Decrypted Local Knowledge):\n{context_block}\n\nUser Question: {query}\nAnswer:"

    def query_local_llm(self, prompt: str) -> str:
        payload = json.dumps({"model": "llama3.2", "prompt": prompt, "stream": False}).encode('utf-8')
        req = urllib.request.Request(self.llm_endpoint, data=payload, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode('utf-8')).get("response", "")
        except Exception as e:
            return f"[Local LLM Response Prompt Ready] (Endpoint offline: {e})"


if __name__ == "__main__":
    print("=== Master ppRAG Privacy System ===")
    pipeline = CompletePPRAGMasterPipeline()
    docs = {
        "doc_1": "Zero Trust Architecture specifies continuous verification across all network resources.",
        "doc_2": "CAPRISE encrypts vector embeddings while preserving conditional query-to-database distance orderings.",
        "doc_3": "DistanceDP applies differential privacy to query vectors, mitigating query analysis attacks."
    }

    print("\n[Phase 1] Ingesting documents...")
    for doc_id, content in docs.items():
        pipeline.phase1_ingest_document(doc_id, content)

    query = "How do CAPRISE and DistanceDP secure vector embeddings in RAG?"