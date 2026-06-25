import os
import re
import json
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

# =====================================================================
# KUTATÓI ALAP-BEÁLLÍTÁSOK (Triviális és Tiltott szavak)
# =====================================================================
# 1. TRIVIÁLIS SZAVAK: Ezeket alapból tudja az ontológia, nem kérdezzük az LLM-et!
# A korábbi fix halmaz helyett egy üres halmazzal indítunk, amit fájlból töltünk be
TRIVIAL_WHITELIST = set()

def load_whitelist_from_ontology(ontology_file_path):
    """Beolvassa az összes meglévő fogalomnevet a Whitelistbe."""
    whitelist = set()
    if os.path.exists(ontology_file_path):
        with open(ontology_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                # Kisbetűsen adjuk hozzá a pontos illeszkedésért
                whitelist.add(item["name"].lower().strip())
    return whitelist

# 2. FEKETE LISTA: Általános szavak, amik nem hordoznak méhészeti jelentést
GENERAL_BLACKLIST = {
    "use", "need", "thing", "result", "results", "development", "analysis", "study",
    "paper", "method", "effect", "effects", "increase", "decrease", "data", "time",
    "number", "sample", "samples", "group", "groups", "test", "tests", "year", "years",
    "observation", "observations", "change", "changes", "system", "process", "activity","tion"
}



import re


def clean_academic_citations(text):
    """
    Eltávolítja a tudományos hivatkozásokat (szerzők, et al., évszámok, [számok])
    a nyers szövegből, hogy ne zavarják a kulcsszó-kinyerést.
    """
    # 1. 'Szerző(k) et al. [évszám]' vagy '(Szerző et al., évszám)' minták törlése
    # Pl: "Zhou et al.", "Smith et al., 2020", "(Brown et al., 2015)"
    text = re.sub(r'\(?[A-Z][a-zA-Z]*(?:\s+and\s+[A-Z][a-zA-Z]*)?\s+et\s+al\.?,?\s*\d{4}?\)?', '', text)
    text = re.sub(r'[A-Z][a-zA-Z]*\s+et\s+al\.', '', text)  # Évszám nélküli sima "Zhou et al."

    # 2. Zárójeles név + évszám hivatkozások törlése
    # Pl: "(Miller, 2019)" vagy "(Davis & Jones, 2022)"
    text = re.sub(r'\([A-Z][a-zA-Z]*,\s*\d{4}\)', '', text)
    text = re.sub(r'\([A-Z][a-zA-Z]*\s*&\s*[A-Z][a-zA-Z]*,\s*\d{4}\)', '', text)

    # 3. Szögletes zárójeles numerikus hivatkozások törlése
    # Pl: "[1]", "[14]", "[2-5]", "[10, 11, 15]"
    text = re.sub(r'\[\d+(?:[\s,\–\-]+\d+)*\]', '', text)

    # 4. Feleslegessé vált dupla szóközök takarítása
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


import os
import re
import json
from collections import Counter

# Manuális feketelista a hibás töredékeknek és nem kívánt szavaknak
MANUAL_BLACKLIST = {
    "tion", "ing", "ment", "able", "ance", "ence", "ive", # Tipikus elválasztási töredékek
    "fig", "table", "etc", "ibid", "vol", "pp", "al",     # Tudományos rövidítések
    "use", "need", "thing", "result", "results", "year","worker", "work","word","world", "wind", "wireless","wire"
         # Általános szavak (ha a max_df nem vinné el)
}


def load_txt_articles_v2(folder_path):
    all_paragraphs = []
    if not os.path.exists(folder_path):
        print(f"Hiba: A '{folder_path}' mappa nem található!")
        return all_paragraphs

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()

                # 1. AUTOMATIKUS ELVÁLASZTÁS JAVÍTÁS (A teljes fájlon futtatjuk a bontás előtt!)
                fixed_content = fix_hyphenation(raw_content)

                # 2. Bontás bekezdésekre (pl. dupla soremelés vagy üres helyek mentén)
                paragraphs = [p.strip() for p in fixed_content.split('  ') if p.strip()]
                if len(paragraphs) <= 1:
                    # Ha a fájlban nem voltak dupla szóközök/soremlések, mondatonként/bekezdésenként bontjuk
                    paragraphs = [p.strip() for p in fixed_content.split('. ') if p.strip()]

                for p in paragraphs:
                    # 3. ACADEMIC CITATION TISZTÍTÁS (a korábbi Zhou et al. szűrő)
                    cleaned_text = clean_academic_citations(p)

                    if len(cleaned_text) > 20:
                        all_paragraphs.append({
                            "doc_id": file_name,
                            "text": cleaned_text
                        })
    return all_paragraphs


def extract_noun_phrases_strict(paragraphs):
    candidate_counts = Counter()
    print("Szigorított nyelvi és manuális feketelistás szűrés...")

    texts = [p['text'] for p in paragraphs]
    for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "textcat"]):
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PROPN":
                continue

            term = chunk.text.lower().strip()
            term = re.sub(r'^(the|a|an|this|that|these|such|our|their|new|old)\s+', '', term)

            # Szigorú karakter-szűrés
            if not re.match(r'^[a-z\s\-]{3,40}$', term):
                continue

            term_lemma = " ".join([token.lemma_ for token in nlp(term)])

            # SZŰRÉS: Manuális feketelista ellenőrzése
            # Megnézzük, hogy a kifejezés szavai közül bármelyik rajta van-e a tiltólistán
            words_in_term = term_lemma.split()
            if any(w in MANUAL_BLACKLIST for w in words_in_term):
                continue

            if term_lemma in TRIVIAL_WHITELIST:
                continue

            candidate_counts[term_lemma] += 1

    return candidate_counts


# A kézi lista helyett bevezetünk egy dinamikus, spaCy alapú szűrést is
def extract_noun_phrases_advanced(paragraphs):
    candidate_counts = Counter()
    print("Kifinomult nyelvi zajszűrés futtatása...")

    texts = [p['text'] for p in paragraphs]
    for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "textcat"]):
        for chunk in doc.noun_chunks:
            # Csak akkor foglalkozunk vele, ha a kifejezés magja főnév,
            # és NEM tulajdonnév (kiszűri a kutatók neveit: pl. 'Smith', 'Johnson')
            if chunk.root.pos_ == "PROPN":
                continue

            term = chunk.text.lower().strip()
            term = re.sub(r'^(the|a|an|this|that|these|such|our|their|new|old|present)\s+', '', term)

            # Szigorú karakter-szűrés: csak betűk, nincs szám, nincs magányos karakter, min. 2 szótag vagy hosszabb szó
            if not re.match(r'^[a-z\s\-]{3,40}$', term):
                continue

            # Lemmatizáció (szótári tőre hozás): a 'developments' és 'development' egynek számítson
            term_lemma = " ".join([token.lemma_ for token in nlp(term)])
            words_in_term = term_lemma.split()
            if any(w in MANUAL_BLACKLIST for w in words_in_term):
                continue
            if term_lemma in TRIVIAL_WHITELIST:
                continue

            candidate_counts[term_lemma] += 1

    return candidate_counts


# A Termhood számításnál pedig bevetjük a matematikai max_df szűrőt:
def calculate_termhood_advanced(candidate_counts, paragraphs, top_k=100):
    print("Statisztikai zajszűrés (max_df=0.85)...")
    texts = [p['text'] for p in paragraphs]

    # max_df=0.85 -> Automatikusan eldobja azokat a szavakat, amik a cikkek több mint 85%-ában szerepelnek!
    # min_df=2    -> Eldobja azokat az elgépeléseket, amik az egész korpuszban csak egyszer fordulnak elő
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(texts)

    importance_scores = tfidf_matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(feature_names, importance_scores))

    scored_candidates = []
    for term, freq in candidate_counts.items():
        tfidf_score = tfidf_dict.get(term, 0.0)
        if tfidf_score == 0:  # Ha a max_df kiszűrte, itt 0 lesz az értéke
            continue

        word_count = len(term.split())
        final_score = tfidf_score * freq * (1.5 if word_count > 1 else 1.0)

        scored_candidates.append((term, final_score))

    scored_candidates.sort(key=lambda x: x, reverse=True)
    return [t for t in scored_candidates[:top_k]]


def calculate_termhood(candidate_counts, paragraphs, top_k=100):
    print("Termhood számítás...")
    texts = [p['text'] for p in paragraphs]

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)

    importance_scores = tfidf_matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(feature_names, importance_scores))

    scored_candidates = []
    for term, freq in candidate_counts.items():
        tfidf_score = tfidf_dict.get(term, 0.0)
        word_count = len(term.split())

        # Összetett szavak előnyben részesítése (pl. "varroa destructor" vs "destructor")
        final_score = tfidf_score * freq * (1.5 if word_count > 1 else 1.0)

        if final_score > 0:
            scored_candidates.append((term, final_score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in scored_candidates[:top_k]]


import re


def fix_hyphenation(raw_text):
    """
    Összeforrasztja az elválasztott szavakat a sorok végén.
    Pl: "infec-\ntion" -> "infection"
    Pl: "honey-\nbee" -> "honeybee"
    """
    # Eltávolítja a kötőjelet, ha azt közvetlenül soremelés (\n vagy \r) követi,
    # és a kötőjel utáni első karakter egy kisbetű (vagyis a szó folytatása).
    cleaned_text = re.sub(r'(\w+)-\s*\n\s*([a-z]\w+)', r'\1\2', raw_text)

    # Biztonsági okokból a bekezdésen belüli felesleges soremeléseket szóközre cseréljük
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
    return cleaned_text

def filter_relevant_paragraphs(paragraphs, top_terms):
    term_set = set(top_terms)
    relevant_paragraphs = []

    for p in paragraphs:
        text_lower = p['text'].lower()
        if not isinstance(text_lower, str):
            print('not a string')
            continue
        matched_terms = [term[0] for term in term_set if term[0] in text_lower]

        if matched_terms:
            relevant_paragraphs.append({
                "doc_id": p['doc_id'],
                "text": p['text'],
                "matched_concepts": matched_terms
            })
    return relevant_paragraphs


if __name__ == "__main__":
    #MAPPA_NEVE = "beekeeping_corpus/ontology_training"  # Cseréld a sajátodra
    MAPPA_NEVE = "beekeeping_corpus/nil_prediction"  # Cseréld a sajátodra

    SEED_ONTOLOGY_FILE = "beekeeping_corpus/glossaries/merged_glossary_terms.json"
    TRAINED_ONTOLOGY_FILE = "expanded_ontology_base.json"
    # DINAMIKUS WHITELIST BETÖLTÉS
    #TRIVIAL_WHITELIST = load_whitelist_from_ontology(SEED_ONTOLOGY_FILE)
    TRIVIAL_WHITELIST = load_whitelist_from_ontology(TRAINED_ONTOLOGY_FILE)
    print(f"Dinamikus Whitelist betöltve: {len(TRIVIAL_WHITELIST)} fogalom védve az ismételt LLM hívásoktól.")

    paragraphs = load_txt_articles_v2(MAPPA_NEVE)
    print(f"Beolvasva: {len(paragraphs)} bekezdés.")

    if paragraphs:
        #raw_candidates = extract_noun_phrases_strict(paragraphs)
        #top_keywords = calculate_termhood(raw_candidates, paragraphs, top_k=100)
        raw_candidates = extract_noun_phrases_advanced(paragraphs)
        top_keywords = calculate_termhood_advanced(raw_candidates, paragraphs, top_k=100)

        print("\n=== TOP 10 TISZTÍTOTT MÉHÉSZETI KULCSSZÓ ===")
        for idx, term in enumerate(top_keywords[:10], 1):
            print(f"{idx}. {term}")

        filtered_data = filter_relevant_paragraphs(paragraphs, top_keywords)
        print(f"\nSzűrés kész: {len(filtered_data)} releváns bekezdés maradt.")

        with open("filtered_paragraphs_test.json", "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
