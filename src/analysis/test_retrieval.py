from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Model and parameters
model = SentenceTransformer('all-MiniLM-L6-v2')
phrase = "artificial intelligence"
fuzzy_threshold = 80   # 0 to 100, usually 70–90 for fuzzy matches
embedding_threshold = 0.6

sentences = [
    "Artificial intelligence is the future.",
    "Our company invests in artificial intelligence for efficiency.",
    "random sentence that mentions artificial objects that may or may not be related to intelligence.",
    "Artificial inteligence is a growing field.",
    "This year, artificial intelligence became our focus area. We developed many thins and used it across many sectors specifically chemical sector"
]

keyword_embedding = model.encode(phrase, convert_to_tensor=True)

for s in sentences:
    # Fuzzy match (can catch minor typos, spacing, etc.)
    fuzzy_score = fuzz.partial_ratio(phrase.lower(), s.lower())
    # Semantic match
    sent_embedding = model.encode(s, convert_to_tensor=True)
    emb_score = util.cos_sim(keyword_embedding, sent_embedding).item()
    # Hybrid condition: either fuzzy OR embedding threshold met
    if fuzzy_score >= fuzzy_threshold or emb_score >= embedding_threshold:
        print(f"[MATCH] {s}")
        print(f"   Fuzzy score: {fuzzy_score}, Embedding: {emb_score:.3f}")
    else:
        print(f"[NO]    {s}")
        print(f"   Fuzzy score: {fuzzy_score}, Embedding: {emb_score:.3f}")
