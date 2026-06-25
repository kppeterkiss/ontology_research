import json
import os
import ollama
from sentence_transformers import SentenceTransformer, util

# KONFIGURÁCIÓ
#SEED_ONTOLOGY_FILE = "seed_oontology.json"
SEED_ONTOLOGY_FILE = "beekeeping_corpus/glossaries/merged_glossary_terms.json"
INPUT_JSON = "filtered_paragraphs_test.json"
ONTOLOGY_BASE_FILE = "expanded_ontology_base.json"
PERFORMANCE_LOG_FILE = "pipeline_performance_log.jsonl"
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'llama3.2'  # 8GB Mac-re optimalizált 3B modell

print("Modellek inicializálása...")
embedding_model = SentenceTransformer(BERT_MODEL_NAME)

# 1. ONTOLÓGIA BETÖLTÉSE
if os.path.exists(ONTOLOGY_BASE_FILE):
    with open(ONTOLOGY_BASE_FILE, "r", encoding="utf-8") as f:
        CURRENT_ONTOLOGY = json.load(f)
    print(f"Betöltve {len(CURRENT_ONTOLOGY)} meglévő koncepció.")
elif os.path.exists(SEED_ONTOLOGY_FILE):
    with open(SEED_ONTOLOGY_FILE, "r", encoding="utf-8") as f:
        CURRENT_ONTOLOGY = json.load(f)
    print(f"Betöltve {len(CURRENT_ONTOLOGY)} meglévő koncepció.")
else:
    CURRENT_ONTOLOGY = []


# ID Számláló
def get_next_id():
    if not CURRENT_ONTOLOGY: return "ONT_001"
    last_id = CURRENT_ONTOLOGY[-1]["id"]
    next_num = int(last_id.split("_")[1]) + 1
    return f"ONT_{str(next_num).zfill(3)}"


# =====================================================================
# JAVÍTÁS 1: LEXIKÁLIS KAPUŐR (Megakadályozza a duplikált új koncepciókat)
# =====================================================================
def check_lexical_exact_match(mention, ontology):
    """
    Karakter szinten ellenőrzi, hogy a szó létezik-e már.
    Ha igen, azonnal visszaadja, kikerülve az LLM-et és a BERT-et.
    """
    mention_clean = mention.lower().strip()
    for concept in ontology:
        # Ellenőrizzük a nevet
        if concept["name"].lower().strip() == mention_clean:
            return concept
        # Ellenőrizzük a szinonimákat (ha vannak)
        if "synonyms" in concept:
            if any(s.lower().strip() == mention_clean for s in concept["synonyms"]):
                return concept
    return None


# =====================================================================
# BERT HASONLÓSÁG KIFEJEZÉS + KONTEXTUS ALAPJÁN
# =====================================================================
def get_top_candidates_combined(mention, context, ontology, combined = True,top_n=2):
    if not ontology: return []
    scored_candidates = []

    mention_emb = embedding_model.encode(mention, convert_to_tensor=True)
    context_emb = embedding_model.encode(context, convert_to_tensor=True)

    for concept in ontology:
        name_emb = embedding_model.encode(concept["name"], convert_to_tensor=True)
        def_emb = embedding_model.encode(concept["definition"], convert_to_tensor=True)

        term_score = util.cos_sim(mention_emb, name_emb).item()
        context_score = util.cos_sim(context_emb, def_emb).item()

        # 60% súly a kifejezés alakjának, 40% a definíció környezetének
        combined_score = (term_score * 0.6) + (context_score * 0.4)

        scored_candidates.append({
            "concept": concept,
            "score": combined_score,
            "term_score": term_score,
            "context_score": context_score
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    if not combined: scored_candidates.sort(key=lambda x: x["term_score"], reverse=True)
    #else: scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:top_n]


# =====================================================================
# 3. FÁZIS: KÉT KÜLÖNBÖZŐ LLM LEKÉRDEZÉS A TESZTHEZ
# =====================================================================

# 1. VERZIÓ: KONTEXTUSOS (Az eddigi működés)
def ask_llama_nil_prediction(mention, context, candidates):
    if not candidates:
        return {"decision": "NEW_CONCEPT", "matched_concept_id": None, "reasoning": "Ontology empty."}

    candidates_str = ""
    for idx, c in enumerate(candidates, 1):
        candidates_str += f"{idx}. [ID: {c['concept']['id']}] Name: {c['concept']['name']} | Def: {c['concept']['definition']}\n"

    prompt = f"""
    You are an expert ontology engineer specializing in beekeeping.
    Evaluate the term "{mention}" found in this text context: "{context}"

    Existing database options:
    {candidates_str}

    Task: Is "{mention}" exactly one of the existing concepts above? 
    If the existing definitions are too broad, but this term represents a distinct sub-type or a completely different specific bee-related concept, select "NEW_CONCEPT".

    Respond strictly in JSON format:
    {{
      "decision": "EXISTING" or "NEW_CONCEPT",
      "matched_concept_id": "ID string or null",
      "reasoning": "One-sentence strict logical justification."
    }}
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 2048},
            keep_alive=0,
            format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": f"Llama error: {str(e)}"}


# 2. VERZIÓ: KONTEXTUS NÉLKÜLI TESZT PROMPT
def ask_llama_nil_prediction_no_context(mention, candidates):
    """
    Új teszt függvény: Csak a szót és a jelölteket kapja meg,
    de tudja, hogy méhészeti (apiculture) domainről van szó.
    """
    if not candidates:
        return {"decision": "NEW_CONCEPT", "matched_concept_id": None, "reasoning": "Ontology empty."}

    candidates_str = ""
    for idx, c in enumerate(candidates, 1):
        # A jelölteknek IS csak a nevét adjuk oda, a definíciót nem,
        # hogy tisztán a kifejezések közötti szemantikai kapcsolatot teszteljük!
        candidates_str += f"{idx}. [ID: {c['concept']['id']}] Name: {c['concept']['name']}\n"

    prompt = f"""
    You are an expert ontology engineer. We are building an ontology specifically for the APICULTURE AND BEEKEEPING domain.

    Target Term to evaluate: "{mention}"

    Existing concepts in our database:
    {candidates_str}

    Task:
    Based solely on your knowledge of apiculture, does the term "{mention}" match or represent the exact same concept as one of the existing database options listed above?
    If it represents a fundamentally different tool, pest, biological entity, or method in beekeeping, you MUST select "NEW_CONCEPT".

    Respond strictly in JSON format:
    {{
      "decision": "EXISTING" or "NEW_CONCEPT",
      "matched_concept_id": "ID string or null",
      "reasoning": "One-sentence strict logical justification based on apiculture."
    }}
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 2048},
            keep_alive=0,
            format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": f"Llama error (No-context): {str(e)}"}


# KIEGÉSZÍTETT NAPLÓZÁS: Mindkét eredményt beírja a JSONL-be
def log_performance_dual_test(source_doc, mention, context, bert_candidates, llama_with_ctx, llama_no_ctx):
    log_entry = {
        "doc": source_doc,
        "mention": mention,
        "context": context,
        "bert_top_suggestions": [c["concept"]["name"] for c in bert_candidates],

        # 1. Teszt eredmény (Környezettel és definícióval)
        "test_1_with_context": {
            "decision": llama_with_ctx.get("decision"),
            "matched_id": llama_with_ctx.get("matched_concept_id"),
            "reasoning": llama_with_ctx.get("reasoning")
        },

        # 2. Teszt eredmény (Csak a kifejezés és a domain infó)
        "test_2_no_context": {
            "decision": llama_no_ctx.get("decision"),
            "matched_id": llama_no_ctx.get("matched_concept_id"),
            "reasoning": llama_no_ctx.get("reasoning")
        }
    }
    with open(PERFORMANCE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
#=====================================================================
# JAVÍTÁS 2: DOKUMENTUM-SZINTŰ AGGREGÁCIÓS MOTOR + RECOVERY (CHECKPOINT)
# =====================================================================
PROGRESS_FILE = "pipeline_progress.json"


def load_progress():
    """Beolvassa a már teljesen feldolgozott dokumentumok listáját."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed_docs", []))
    return set()


def save_progress(processed_docs_set):
    """Elmenti a feldolgozott dokumentumok listáját a lemezre."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_docs": list(processed_docs_set)}, f, indent=2, ensure_ascii=False)

def generate_strict_definition(mention, context):
    """Generates a crisp, distinctive definition to prevent over-generalization."""
    prompt = f"""
    You are an expert apiculture lexicographer. Write a highly specific 1-sentence definition for "{mention}".
    Base it strictly on the functional domain knowledge in this context: "{context}"
    Avoid generic phrases. Start directly with the exact class type (e.g., "A specific species of mite...", "A mechanical extraction tool...").

    Respond strictly in JSON format:
    {{
      "generated_definition": "The crisp 1-sentence definition."
    }}
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 2048},
            keep_alive=0,
            format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"generated_definition": f"A specialized term representing {mention}."}



def run_optimized_pipeline():
    global new_concept_counter

    if not os.path.exists(INPUT_JSON):
        print(f"Hiba: Hiányzik a '{INPUT_JSON}'")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_paragraphs = json.load(f)

    # 1. RECOVERY: Korábbi előrehaladás betöltése
    processed_documents = load_progress()
    if processed_documents:
        print(
            f"[RECOVERY ACTIVATED] Találtam korábbi futási nyomot. {len(processed_documents)} cikk már kész, ezeket átugorjuk.")

    print("Adatok csoportosítása...")
    doc_mention_map = {}
    for p in all_paragraphs:
        doc_id = p["doc_id"]
        if doc_id in processed_documents: continue
        if doc_id not in doc_mention_map: doc_mention_map[doc_id] = {}
        for mention in p["matched_concepts"]:
            if mention not in doc_mention_map[doc_id] or len(p["text"]) > len(doc_mention_map[doc_id][mention]):
                doc_mention_map[doc_id][mention] = p["text"]

    total_jobs = sum(len(mentions) for mentions in doc_mention_map.values())
    if total_jobs == 0:
        print("\nNincs új feldolgozandó feladat.")
        return

    print(f"Csoportosítás kész. Összesen {total_jobs} feladat vár elvégzésre.")
    current_job_num = 1

    for doc_id, mentions in doc_mention_map.items():
        print(f"\n--- FÁJL FELDOLGOZÁSA: {doc_id} ---")

        for mention, context in mentions.items():
            print(f"[{current_job_num}/{total_jobs}] Kifejezés: '{mention}'")
            current_job_num += 1

            # Lexikális Kapuőr ellenőrzése
            lexical_match = check_lexical_exact_match(mention, CURRENT_ONTOLOGY)
            if lexical_match:
                print(f"   [->] Lexikális egyezés! LLM hívások kihagyva.")
                # Ide is beírhatsz egy üres tesztnaplót, ha akarod, de a lényeg az LLM-es eseteken van
                continue

            # BERT jelölt keresés
            top_candidates = get_top_candidates_combined(mention, context, CURRENT_ONTOLOGY, top_n=2)
            top_term_and_label_only_candidates = get_top_candidates_combined(mention, context, CURRENT_ONTOLOGY,combined=False, top_n=2)

            # --- AZ ABLÁCIÓS TESZT INDÍTÁSA ---
            # Hívás 1: Kontextussal és definícióval
            llama_with_ctx = ask_llama_nil_prediction(mention, context, top_candidates)

            # Hívás 2: CSAK a kifejezés és a domain információ (Kontextus NÉLKÜL)
            llama_no_ctx = ask_llama_nil_prediction_no_context(mention, top_term_and_label_only_candidates)

            print(
                f"   -> Teszt 1 (Ctx): {llama_with_ctx.get('decision')} | Teszt 2 (No-Ctx): {llama_no_ctx.get('decision')}")

            # Mentés a közös duális teljesítménynaplóba
            log_performance_dual_test(doc_id, mention, context, top_candidates, llama_with_ctx, llama_no_ctx)

            # --- ONTOLÓGIA ÉPÍTÉSE (Kizárólag az 1-es teszt döntése alapján!) ---
            decision = llama_with_ctx.get("decision", "NEW_CONCEPT")
            if decision == "NEW_CONCEPT":
                print(f"   [!] Új koncepció (kontextus alapján). Definíció generálása...")
                def_data = generate_strict_definition(mention, context)
                generated_def = def_data.get("generated_definition", f"A specialized term representing {mention}.")

                new_node = {
                    "id": get_next_id(),
                    "name": mention.title(),
                    "definition": generated_def,
                    "synonyms": []
                }
                CURRENT_ONTOLOGY.append(new_node)
                with open(ONTOLOGY_BASE_FILE, "w", encoding="utf-8") as f:
                    json.dump(CURRENT_ONTOLOGY, f, indent=2, ensure_ascii=False)

        processed_documents.add(doc_id)
        save_progress(processed_documents)
        print(f"=== {doc_id} kész. Checkpoint mentve. ===")

    print("\nFolyamat sikeresen véget ért.")


if __name__ == "__main__":
    run_optimized_pipeline()
