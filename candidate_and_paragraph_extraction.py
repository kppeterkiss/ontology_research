import json
import os
import re
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. NYELVI MODELL BETÖLTÉSE
# A könnyű modellt használjuk, ami másodpercek alatt lefut a laptopodon
nlp = spacy.load("en_core_web_sm")


def load_txt_articles(folder_path):
    """Beolvassa a mappában lévő összes .txt fájlt és bekezdésekre bontja őket."""
    all_paragraphs = []
    if not os.path.exists(folder_path):
        print(f"Hiba: A '{folder_path}' mappa nem található!")
        return all_paragraphs

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Bekezdésekre bontás az üres sorok mentén
                paragraphs = [p.strip() for p in f.read().split('\n') if p.strip()]
                for p in paragraphs:
                    all_paragraphs.append({
                        "doc_id": file_name,
                        "text": p
                    })
    return all_paragraphs


# =====================================================================
# 1. LÉPÉS: LINGUISTIC FILTERING (Szófaji szűrés spaCy-vel)
# =====================================================================
def extract_noun_phrases(paragraphs):
    """
    Végigmegy a szövegeken, és kigyűjti a potenciális szakkifejezéseket.
    Csak a főneveket (pl. propolis) és az összetett kifejezéseket (pl. queen excluder, American foulbrood) tartja meg.
    """
    candidate_counts = Counter()
    print("Szövegek nyelvi elemzése (főnevek és jelzős szerkezetek kigyűjtése)...")

    # Batch processing a spaCy-ben a gyorsaságért
    texts = [p['text'] for p in paragraphs]
    for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "textcat"]):
        # Kigyűjtjük a főnévi szerkezeteket (Noun Chunks)
        for chunk in doc.noun_chunks:
            # Tisztítás: kisbetűsítés, felesleges névelők és írásjelek eltávolítása
            clean_term = chunk.root.head.text.lower() if chunk.root.dep_ == "compound" else ""
            term = chunk.text.lower().strip()

            # Stopword-ök kiszűrése az elejéről (pl. "a hive" -> "hive")
            term = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', term)

            # Csak betűket és kötőjeleket tartalmazó, 2 karakternél hosszabb szavak
            if re.match(r'^[a-z\s\-]+$', term) and len(term) > 2:
                candidate_counts[term] += 1

    return candidate_counts


# =====================================================================
# 2. LÉPÉS: CONTRASTIVE TF-IDF PONTOZÁS (Termhood számítás)
# =====================================================================
def calculate_termhood(candidate_counts, paragraphs, top_k=50):
    """
    Kontrasztos pontozást végez. A scikit-learn TF-IDF-jét használja arra,
    hogy megnézze, mely kifejezések hordoznak egyedi információt a korpuszban,
    kiszűrve az általános tudományos kifejezéseket (pl. "results", "analysis").
    """
    print("Kontrasztos TF-IDF pontszámok (Termhood) kiszámítása...")
    texts = [p['text'] for p in paragraphs]

    # Olyan TF-IDF-et használunk, ami kiszűri a túl gyakori általános angol szavakat
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Átlagos TF-IDF pontszám kiszámítása minden szóra a 222 cikkben
    importance_scores = tfidf_matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(feature_names, importance_scores))

    # Kombináljuk a spaCy gyakoriságot a TF-IDF fontossággal
    scored_candidates = []
    for term, freq in candidate_counts.items():
        # Ha a kifejezés szerepel a TF-IDF mátrixban, lekérjük a pontszámát
        tfidf_score = tfidf_dict.get(term, 0.0)

        # Ha összetett kifejezés (több szóból áll), kap egy kis matematikai előnyt (C-value elv)
        word_count = len(term.split())
        final_score = tfidf_score * freq * (1.2 if word_count > 1 else 1.0)

        if final_score > 0:
            scored_candidates.append((term, final_score))

    # Rendezés pontszám alapján csökkenő sorrendben
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[:top_k]


# =====================================================================
# 3. LÉPÉS: RELEVÁNS BEKEZDÉSEK SZŰRÉSE
# =====================================================================
def filter_relevant_paragraphs(paragraphs, top_terms):
    """Visszaadja azokat a bekezdéseket, amik tartalmazzák a top kulcsszavak valamelyikét."""
    term_set = set([t[0] for t in top_terms])
    relevant_paragraphs = []

    for p in paragraphs:
        text_lower = p['text'].lower()
        # Megnézzük, hogy a bekezdésben szerepel-e bármelyik top kulcsszó
        matched_terms = [term for term in term_set if term in text_lower]

        if matched_terms:
            relevant_paragraphs.append({
                "doc_id": p['doc_id'],
                "text": p['text'],
                "matched_concepts": matched_terms
            })

    return relevant_paragraphs


# =====================================================================
# FŐPROGRAM FUTTATÁSA
# =====================================================================
if __name__ == "__main__":
    MAPPA_NEVE = "beekeeping_corpus"

    # 0. Cikkek beolvasása
    paragraphs = load_txt_articles(MAPPA_NEVE)
    print(f"Sikeresen beolvasva: {len(paragraphs)} bekezdés.")

    if paragraphs:
        # 1. Nyelvi szűrés
        raw_candidates = extract_noun_phrases(paragraphs)

        # 2. Kontrasztos pontozás (Kérjük a top 100 legjobb kulcsszót)
        top_keywords = calculate_termhood(raw_candidates, paragraphs, top_k=100)

        print("\n=== TOP 15 KINYERT KULCSSZÓ (KANDIDÁTUS) ===")
        for idx, (term, score) in enumerate(top_keywords[:15], 1):
            print(f"{idx}. {term} (Pontszám: {score:.4f})")

        # 3. Szűrés a kulcsszavak alapján
        filtered_data = filter_relevant_paragraphs(paragraphs, top_keywords)
        print(f"\nSzűrés kész: {len(paragraphs)} bekezdésből {len(filtered_data)} maradt meg mint releváns.")

        # Mentés JSON-ba, amit a következő lépésben (Llama / Ollama) azonnal be tudsz olvasni
        with open("filtered_paragraphs.json", "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print("A szűrt bekezdések elmentve a 'filtered_paragraphs.json' fájlba.")
