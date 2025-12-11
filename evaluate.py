"""
Main evaluation script.
This script:
- Loads the chat and context JSON files
- Computes embeddings (or uses provided embeddings)
- Scores relevance, completeness, hallucination risk
- Estimates latency and cost
- Writes a JSON report
"""

import functools
import hashlib
import argparse
import json
import os
from datetime import datetime
from dateutil import parser as dateparser
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

_EMBED_CACHE_MAX = 50000 

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@functools.lru_cache(maxsize=_EMBED_CACHE_MAX)
def _embed_single_cached_tuple(model_name_and_text: str):
    model_name, text = model_name_and_text.split("||", 1)
    model = get_model()  
    emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    return tuple(emb.tolist())


def embed_single_text(text: str) -> np.ndarray:
    key = f"{MODEL_NAME}||{text}"
    tup = _embed_single_cached_tuple(key)
    return np.array(tup, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def save_json(obj: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


MODEL_NAME = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
EMB_MODEL = None
def get_model():
    global EMB_MODEL
    if EMB_MODEL is None:
        print(f"Loading embedding model: {MODEL_NAME} (this may take a few seconds)...")    
        EMB_MODEL = SentenceTransformer(MODEL_NAME)
    return EMB_MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        # dimension fallback; will be replaced on actual model use
        return np.zeros((0, 384), dtype=np.float32)  

    model = get_model()

    results = []
    to_batch = []
    batch_indices = []
    for i, t in enumerate(texts):
        key = f"{MODEL_NAME}||{t}"
        try:
            tup = _embed_single_cached_tuple(key)
            results.append(np.array(tup, dtype=np.float32))
        except Exception:
            results.append(None)
            to_batch.append(t)
            batch_indices.append(i)

    if to_batch:
        batch_embs = model.encode(to_batch, show_progress_bar=False, convert_to_numpy=True)
        for idx, emb in zip(batch_indices, batch_embs):
            # ensure cached tuple stored
            try:
                _embed_single_cached_tuple(f"{MODEL_NAME}||{texts[idx]}")
            except Exception:
                # caching function may raise if called concurrently; ignore
                pass
            results[idx] = np.array(emb, dtype=np.float32)

    return np.vstack(results)


def get_context_embeddings(context_snippets: List[Dict[str, Any]]) -> List[np.ndarray]:
    texts_to_embed = []
    idxs_to_embed = []
    pre_embs = [None] * len(context_snippets)

    for i, s in enumerate(context_snippets):
        emb = s.get('embedding')
        if emb:
            pre_embs[i] = np.array(emb, dtype=np.float32)
        else:
            texts_to_embed.append(s.get('text', ''))
            idxs_to_embed.append(i)

    if texts_to_embed:
        batch_embs = embed_texts(texts_to_embed)
        for j, idx in enumerate(idxs_to_embed):
            pre_embs[idx] = np.array(batch_embs[j], dtype=np.float32)

    return pre_embs


def score_relevance(reply: str, query: str, context_snippets: List[Dict[str, Any]], top_k: int = 5):
    # compute cached single embeddings for reply and query
    reply_emb = embed_single_text(reply)
    query_emb = embed_single_text(query)

    # get embeddings for top_k snippets (use precomputed if present)
    top_snippets = context_snippets[:top_k] if context_snippets else []
    context_embs = get_context_embeddings(top_snippets)
    sims_context = [cosine(reply_emb, e) for e in context_embs] if context_embs else []

    sim_query = cosine(reply_emb, query_emb)
    sim_top_context = max(sims_context) if sims_context else 0.0
    relevance = 0.4 * sim_query + 0.6 * sim_top_context
    return {
        'relevance_score': relevance,
        'sim_to_query': sim_query,
        'sim_to_top_context': sim_top_context,
        'top_k_context_sims': sims_context,
    }


def extract_expected_points(context_snippets: List[Dict[str, Any]], n_points: int = 6):
    pool = []
    for s in context_snippets:
        text = s.get('text', '')
        for sent in re.split(r'(?<=[.!?])\s+', text.strip()):
            if len(sent.split()) >= 4:
                pool.append(sent.strip())
    if not pool:
        return []
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vec.fit_transform(pool)
    scores = X.sum(axis=1).A1
    ranked_idx = np.argsort(-scores)
    selected = [pool[i] for i in ranked_idx[:n_points]]
    return selected


def is_point_covered(point: str, reply: str, threshold: float = 0.70) -> Dict[str, Any]:
    embs = embed_texts([point, reply])
    sim = cosine(embs[0], embs[1])
    return {'point': point, 'covered': sim >= threshold, 'similarity': sim}


def score_completeness(reply: str, context_snippets: List[Dict[str, Any]], n_points: int = 4, coverage_threshold: float = 0.70):
    expected = extract_expected_points(context_snippets, n_points=n_points)
    if not expected:
        # No expected points found -> return None to indicate "not applicable"
        return {'completeness_score': None, 'expected_points': [], 'covered_points': []}
    covered = [is_point_covered(p, reply, threshold=coverage_threshold) for p in expected]
    covered_count = sum(1 for c in covered if c['covered'])
    completeness = covered_count / len(expected)
    return {
        'completeness_score': completeness,
        'expected_points': expected,
        'covered_points': covered,
    }


def extract_claims(reply: str) -> List[str]:
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reply.strip()) if s.strip()]
    claims = []
    for sent in sents:
        if re.search(r"\d", sent):
            claims.append(sent)
            continue
        if _nlp:
            doc = _nlp(sent)
            ents = [e for e in doc.ents if e.label_ in ('PERSON','ORG','GPE','PRODUCT','DATE','NORP','EVENT','WORK_OF_ART')]
            if ents:
                claims.append(sent)
                continue
        else:
            tokens = sent.split()
            cap_count = sum(1 for t in tokens if t[:1].isupper())
            if cap_count >= 2:
                claims.append(sent)
                continue
    return claims


def verify_claims(claims: List[str], context_snippets: List[Dict[str, Any]], support_threshold: float = 0.68) -> List[Dict[str, Any]]:
    results = []
    if not claims:
        return results

    # batch-embed claims
    claim_embs = embed_texts(claims)

    # precompute context embeddings once
    context_embs = get_context_embeddings(context_snippets)

    for i, claim in enumerate(claims):
        claim_emb = claim_embs[i]
        best_sim = 0.0
        best_context = None

        # brute-force search over context embeddings (suitable for small snippet lists)
        for j, emb in enumerate(context_embs):
            if emb is None:
                continue
            s = cosine(claim_emb, emb)
            if s > best_sim:
                best_sim = s
                best_context = context_snippets[j]

        if best_sim >= support_threshold:
            status = 'supported'
        elif best_sim >= 0.4:
            status = 'weakly_supported'
        else:
            status = 'unsupported'

        results.append({
            'claim': claim,
            'status': status,
            'best_similarity': float(best_sim),
            'best_context_snippet': best_context,
        })
    return results


def hallucination_risk_from_verifications(verifications: List[Dict[str, Any]]) -> float:
    if not verifications:
        return 0.0
    bad = sum(1 for v in verifications if v['status'] in ('unsupported',))
    return bad / len(verifications)


def measure_latency_and_cost(chat: Dict[str, Any], reply_text: str) -> Dict[str, Any]:
    lat_ms = None
    if 'request_sent_ts' in chat and 'response_received_ts' in chat:
        try:
            a = dateparser.parse(chat['request_sent_ts'])
            b = dateparser.parse(chat['response_received_ts'])
            lat_ms = (b - a).total_seconds() * 1000.0
        except Exception:
            lat_ms = None

    if lat_ms is None and 'latency_ms' in chat:
        lat_ms = float(chat['latency_ms'])

    token_count = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        token_count = len(enc.encode(chat.get('prompt','') + '\n' + reply_text))
    except Exception:
        words = len((chat.get('prompt','') + ' ' + reply_text).split())
        token_count = int(words / 0.75)

    model = chat.get('model', 'gpt-like')
    pricing_per_1k = chat.get('pricing_per_1k', None)
    if pricing_per_1k is None:
        pricing_per_1k = 0.02 
    cost_usd = (token_count / 1000.0) * pricing_per_1k
    return {
        'latency_ms': lat_ms,
        'estimated_tokens': token_count,
        'estimated_cost_usd': cost_usd,
        'model': model,
    }


def evaluate(chat_path: str, contexts_path: str, out_path: str, args): 
    chat = load_json(chat_path)
    contexts = load_json(contexts_path)

    user_message = chat.get('user_message') or chat.get('prompt') or chat.get('query') or ""
    assistant_reply = chat.get('assistant_reply') or chat.get('reply') or chat.get('response') or ""
    context_snippets = contexts.get('snippets') if isinstance(contexts, dict) and 'snippets' in contexts else contexts
    if context_snippets is None:
        context_snippets = []

    rel = score_relevance(assistant_reply, user_message, context_snippets, top_k=args.top_k)

    comp = score_completeness(assistant_reply, context_snippets, n_points=args.n_points, coverage_threshold=args.coverage_threshold)

    claims = extract_claims(assistant_reply)
    verifs = verify_claims(claims, context_snippets, support_threshold=args.support_threshold)
    halluc_risk = hallucination_risk_from_verifications(verifs)

    latcost = measure_latency_and_cost(chat, assistant_reply)
    report = {
        'meta': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'chat_file': chat_path,
            'context_file': contexts_path,
        },
        'relevance': rel,
        'completeness': comp,
        'claims': verifs,
        'hallucination_risk': halluc_risk,
        'latency_and_cost': latcost,
    }
    save_json(report, out_path)
    print(f"Wrote evaluation report to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--chat', required=True, help='path to chat JSON file')
    p.add_argument('--contexts', required=True, help='path to contexts JSON file')
    p.add_argument('--out', default='report.json', help='output report JSON')
    p.add_argument('--top-k', dest='top_k', type=int, default=5)
    p.add_argument('--n-points', dest='n_points', type=int, default=6)
    p.add_argument('--support-threshold', dest='support_threshold', type=float, default=0.68)
    p.add_argument('--coverage-threshold', dest='coverage_threshold', type=float, default=0.70)
    args = p.parse_args()
    evaluate(args.chat, args.contexts, args.out, args)
