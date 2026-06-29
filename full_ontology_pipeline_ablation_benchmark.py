import json
import os
import torch
import ollama
import spacy

from sentence_transformers import SentenceTransformer, util

# =====================================================================
# 1. KONFIGURÁCIÓ ÉS BENCHMARK BEÁLLÍTÁSOK
# =====================================================================

# KONFIGURÁCIÓ
#SEED_ONTOLOGY_FILE = "seed_oontology.json"
PROGRESS_FILE = "pipeline_progress.json"

append_to_ontology = False
paul = True
INPUS_JSON = "filtered_paragraphs_test.json"
# Paul con fig:
if paul:
    SEED_ONTOLOGY_FILE = "ontology_rdf.json"

    INPUT_JSON = ["beekeeping_corpus/xls/gold_standard_multi_word_noun_phrase_contexts 1.json",
                  "beekeeping_corpus/xls/gold_standard_single_noun_contexts 1.json"]
    ONTOLOGY_BASE_FILE = "ontology_rdf_llama3_2.json"
    EXPANDED_ONTOLOGY_BASE_FILE = "p_expanded_ontology_base.json"

    PERFORMANCE_LOG_FILE = "paul_pipeline_performance_log_benchmarking.jsonl"
else:
    SEED_ONTOLOGY_FILE = "beekeeping_corpus/glossaries/merged_glossary_terms.json"
    INPUT_JSON = "filtered_paragraphs_test.json"
    ONTOLOGY_BASE_FILE = "expanded_ontology_base.json"
    PERFORMANCE_LOG_FILE = "pipeline_performance_log.jsonl"
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'

# A TESZTELNI KÍVÁNT MODELLEK LISTÁJA
TEST_MODELS = ["llama3.2", "mistral", "llama3.1"]

print("Modellek inicializálása...")
embedding_model = SentenceTransformer(BERT_MODEL_NAME)
nlp = spacy.load("en_core_web_sm") if 'spacy' in globals() else None
# Ha a spacy nincs importálva felül, itt pótoljuk a mondat-ablakozáshoz
if not nlp:
    import spacy

    nlp = spacy.load("en_core_web_sm")

# ONTOLÓGIA BETÖLTÉSE
if os.path.exists(ONTOLOGY_BASE_FILE):
    with open(ONTOLOGY_BASE_FILE, "r", encoding="utf-8") as f:
        CURRENT_ONTOLOGY = json.load(f)
    print(f"[Inkrementális mód megkezdett ontológiából] Betöltve {len(CURRENT_ONTOLOGY)} korábbi koncepció.")
elif os.path.exists(SEED_ONTOLOGY_FILE):
    with open(SEED_ONTOLOGY_FILE, "r", encoding="utf-8") as f:
        CURRENT_ONTOLOGY = json.load(f)
    print(f"[Inkrementális mód seed-ből] Betöltve {len(CURRENT_ONTOLOGY)} korábbi koncepció.")
else:
    print("[Kezdeti mód] Alap ontológia inicializálása...")
    CURRENT_ONTOLOGY = [
        {"id": "ONT_001", "name": "Honey Bee",
         "definition": "A stinging winged insect that lives in highly organized colonies and produces wax and honey."},
        {"id": "ONT_002", "name": "Bee Smoker",
         "definition": "A device used in beekeeping to calm honey bees by puffing smoke into the hive."},
        {"id": "ONT_003", "name": "Wax Moth",
         "definition": "A destructive hive pest whose larvae eat and ruin beeswax combs."}
    ]
    with open(ONTOLOGY_BASE_FILE, "w", encoding="utf-8") as f:
        json.dump(CURRENT_ONTOLOGY, f, indent=2, ensure_ascii=False)


def get_next_id():
    if not CURRENT_ONTOLOGY: return "ONT_001"
    last_id = CURRENT_ONTOLOGY[-1]["id"]
    next_num = int(last_id.split("_")[-1]) + 1
    return f"ONT_{str(next_num).zfill(3)}"


# PROGRESS MENEDZSMENT (RECOVERY)
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed_docs", []))
    return set()


def save_progress(processed_docs_set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_docs": list(processed_docs_set)}, f, indent=2, ensure_ascii=False)


# KAPUŐRÖK ÉS HASONLÓSÁGOK
def check_lexical_exact_match(mention, ontology):
    mention_clean = mention.lower().strip()
    for concept in ontology:
        if concept["name"].lower().strip() == mention_clean:
            return concept
        if "synonyms" in concept:
            if any(s.lower().strip() == mention_clean for s in concept["synonyms"]):
                return concept
    return None


def get_top_candidates_combined(mention, context, ontology, combined = True,top_n=5):
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

def extract_strict_sentence_window(mention, large_text, window_size=1):
    clean_large_text = large_text.replace('\n', ' ').replace('\r', ' ')
    doc = nlp(clean_large_text)
    sentences = list(doc.sents)
    matched_sentence_idx = -1
    for idx, sent in enumerate(sentences):
        if mention.lower() in sent.text.lower():
            matched_sentence_idx = idx
            break
    if matched_sentence_idx == -1:
        return large_text[:300] + "..."
    start_idx = max(0, matched_sentence_idx - window_size)
    end_idx = min(len(sentences), matched_sentence_idx + window_size + 1)
    return " ".join([s.text.strip() for s in sentences[start_idx:end_idx]])


# =====================================================================
# 2. FÁZIS: BENCHMARK LEKÉRDEZÉSEK (KONTEXTUSSAL ÉS ANÉLKÜL)
# =====================================================================
def ask_llama_with_context(model_name, mention, context, candidates):
    """1. TESZT VERZIÓ: Kontextussal és definícióval (Szigorú fogalmi azonosság teszt)"""
    candidates_str = "".join(
        [f"- [ID: {c['concept']['id']}] Name: {c['concept']['name']} | Def: {c['concept']['definition']}\n" for c in
         candidates])

    prompt = f"""
    You are a strict ontology engineer specializing in precision beekeeping.

    [TASK INSTRUCTIONS]
    Evaluate if the target term represents the EXACT SAME CONCEPT (is a direct synonym or equivalent terms) as one of the existing options.
    If the target term is a separate part, a component, or a related but distinct technology/metric, you MUST select "NEW_CONCEPT". Do NOT map related things together.
    Respond strictly in the required JSON format.

    [EXAMPLES]
    Example 1 (EXISTING - Direct Synonym):
    Target Term: "temperature probe"
    Context: "We inserted a digital temperature probe into the brood nest to monitor thermal changes."
    Existing database options:
    1. [ID: ONT_101] Name: Temperature Sensor | Def: A hardware sensor used to measure the thermal conditions inside a beehive.
    2. [ID: ONT_102] Name: Weight Scale | Def: An electronic scale placed under the hive to track honey yield.
    Expected Output:
    {{
      "decision": "EXISTING",
      "matched_concept_id": "ONT_101",
      "reasoning": "A temperature probe is functionally identical to a temperature sensor and represents the exact same concept."
    }}

    Example 2 (NEW_CONCEPT - Related but NOT identical):
    Target Term: "load cell"
    Context: "The automated scale system uses a 200kg load cell to measure structural deformation under hive weight."
    Existing database options:
    1. [ID: ONT_101] Name: Temperature Sensor | Def: A hardware sensor used to measure the thermal conditions inside a beehive.
    2. [ID: ONT_102] Name: Weight Scale | Def: An electronic scale placed under the hive to track honey yield.
    Expected Output:
    {{
      "decision": "NEW_CONCEPT",
      "matched_concept_id": null,
      "reasoning": "A load cell is a sub-component used inside a weight scale, but it is a distinct hardware entity, not the scale itself."
    }}

    [ACTUAL TASK TO EVALUATE]
    Target Term: "{mention}"
    Context from research paper: "{context}"

    Existing database options:
    {candidates_str}

    Task: Is "{mention}" the EXACT SAME CONCEPT as one of the existing options above? Select "NEW_CONCEPT" if it is merely a part, a component, or a related but distinct entity.

    Respond strictly in JSON format:
    {{
      "decision": "EXISTING" or "NEW_CONCEPT",
      "matched_concept_id": "ID string if existing, else null",
      "reasoning": "One-sentence strict logical justification."
    }}
    """
    try:
        response = ollama.chat(
            model=model_name, messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 2048},
            keep_alive=0, format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"decision": "ERROR", "matched_concept_id": None, "reasoning": str(e)}


def ask_llama_without_context(model_name, mention, candidates):
    """2. TESZT VERZIÓ: Kontextus NÉLKÜL (Szigorú fogalmi azonosság teszt)"""
    candidates_str = "".join([f"- [ID: {c['concept']['id']}] Name: {c['concept']['name']}\n" for c in candidates])

    prompt = f"""
    You are a strict ontology engineer building an ontology specifically for the PRECISION BEEKEEPING domain.

    [TASK INSTRUCTIONS]
    Based solely on your knowledge of smart apiculture, evaluate if the target term represents the EXACT SAME ENTITY/CONCEPT as one of the existing options.
    If the target term is a part, a sub-component, or just a related concept, you MUST select "NEW_CONCEPT".
    Respond strictly in the required JSON format.

    [EXAMPLES]
    Example 1 (EXISTING - Direct Synonym):
    Target Term: "telemetry module"
    Existing concepts in our database:
    1. [ID: ONT_104] Name: Communication Unit
    2. [ID: ONT_105] Name: Hive Scale
    Expected Output:
    {{
      "decision": "EXISTING",
      "matched_concept_id": "ONT_104",
      "reasoning": "A telemetry module acts as the communication unit responsible for transmitting data from the hive."
    }}

    Example 2 (NEW_CONCEPT - Related but NOT identical):
    Target Term: "internal battery"
    Existing concepts in our database:
    1. [ID: ONT_104] Name: Communication Unit
    2. [ID: ONT_105] Name: Hive Scale
    Expected Output:
    {{
      "decision": "NEW_CONCEPT",
      "matched_concept_id": null,
      "reasoning": "An internal battery provides power to these devices but is a distinct physical electronic component, not a scale or communication unit."
    }}

    [ACTUAL TASK TO EVALUATE]
    Target Term to evaluate: "{mention}"

    Existing concepts in our database:
    {candidates_str}

    Task: Based solely on your general knowledge of precision beekeeping, does "{mention}" represent the EXACT SAME CONCEPT as one of the options listed above? Select "NEW_CONCEPT" if it is a component, a part, or a related but separate entity.

    Respond strictly in JSON format:
    {{
      "decision": "EXISTING" or "NEW_CONCEPT",
      "matched_concept_id": "ID string or null",
      "reasoning": "One-sentence strict logical justification based on smart apiculture."
    }}
    """
    try:
        response = ollama.chat(
            model=model_name, messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 2048},
            keep_alive=0, format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"decision": "ERROR", "matched_concept_id": None, "reasoning": str(e)}


def generate_strict_definition(mention, context):
    prompt = f"""
    You are an expert precision beekeping lexicographer. Write a highly specific 1-sentence definition for "{mention}".
    Base it strictly on the functional domain knowledge in this context: "{context}"

    Respond strictly in JSON format:
    {{
      "generated_definition": "The crisp 1-sentence definition."
    }}
    """
    try:
        response = ollama.chat(
            model="llama3.2",  # A definíciógyártáshoz fixen a kis modellt használjuk a sebességért
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 2048},
            keep_alive=0, format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"generated_definition": f"A specialized term representing {mention}."}


# DUÁLIS BENCHMARK NAPLÓZÁS
def log_benchmark_result(source_doc, mention, context, bert_candidates,top_candidates_no_ctx, model_outputs):
    log_entry = {
        "doc": source_doc,
        "mention": mention,
        "context": context,
        "bert_top_suggestions_ctx": [c["concept"]["name"] for c in bert_candidates],
        "bert_top_suggestions_no_ctx":[c["concept"]["name"] for c in top_candidates_no_ctx],
        "benchmark_results": model_outputs  # Tartalmazza mind a 3 modell With/Without eredményeit
    }
    with open(PERFORMANCE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# =====================================================================
# 3. FÁZIS: FUTTATÓ MOTOR (6-IRÁNYÚ ABLÁCIÓS BENCHMARK)
# =====================================================================
def run_benchmark_pipeline():

    input_files = [INPUT_JSON] if isinstance(INPUT_JSON, str) else INPUT_JSON

    all_paragraphs=[]
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Hiba: Hiányzik a '{input_file}'")
            return

        with open(input_file, "r", encoding="utf-8") as f:
            all_paragraphs.extend(json.load(f))

    processed_documents = load_progress()
    if processed_documents:
        print(f"[RECOVERY] {len(processed_documents)} cikk már kész, átugorjuk őket.")

    # Dokumentum-szintű aggregáció az ismétlődések ellen
    doc_mention_map = {}
    for p in all_paragraphs:
        doc_id = p["doc_id"]
        if doc_id in processed_documents: continue
        if doc_id not in doc_mention_map: doc_mention_map[doc_id] = {}
        for mention in p["matched_concepts"]:
            if mention not in doc_mention_map[doc_id] or len(p["text"]) > len(doc_mention_map[doc_id][mention]):
                doc_mention_map[doc_id][mention] = p["text"]
    # Kiválasztunk maximum 100 feladatot a teszthez, hogy ne terheljük túl a laptopot egyszerre
    all_jobs = []
    for doc_id, mentions in doc_mention_map.items():
        for mention, large_context in mentions.items():
            all_jobs.append({"doc_id": doc_id, "mention": mention, "large_context": large_context})
    # Korlátozzuk a tesztet pontosan a kért mennyiségre (pl. az első 100 elérhető feladat)
    test_jobs = all_jobs[:100]
    total_jobs = len(test_jobs)
    if total_jobs == 0:
        print("\nNincs új feldolgozandó feladat.")
        return
    print(f"Csoportosítás kész. Elindítom a 6-irányú benchmarkot {total_jobs} egyedi kifejezésen...")
    for idx, job in enumerate(test_jobs, 1):
        doc_id = job["doc_id"]
        mention = job["mention"]
        large_context = job["large_context"]
        # Mondat-ablakozás alkalmazása (Tűéles kontextus)
        context = extract_strict_sentence_window(mention, large_context, window_size=1)
        print(f"\n[{idx}/{total_jobs}] Kifejezés: '{mention}' (Fájl: {doc_id})")

        # 1. Lexikális Kapuőr ellenőrzése
        lexical_match = check_lexical_exact_match(mention, CURRENT_ONTOLOGY)
        if lexical_match:
            print(f"   [->] Lexikális egyezés megtalálva! ID: {lexical_match['id']}. Modellek kihagyva.")
            log_benchmark_result(doc_id, mention, context, "", "",
                                 {m: {"with_context": "LEXICAL", "without_context": "LEXICAL"} for m in TEST_MODELS})

            continue
        # 2. BERT jelölt állítás
        top_candidates = get_top_candidates_combined(mention, context, CURRENT_ONTOLOGY, top_n=5)
        top_term_and_label_only_candidates = get_top_candidates_combined(mention, context, CURRENT_ONTOLOGY,
                                                                         combined=False, top_n=5)
        # 3. REKORDOK GYŰJTÉSE MINDEN MODELLRE (With vs Without Context)
        outputs_for_this_mention = {}
        for model_name in TEST_MODELS:
            print(f"   -> Futtatás: {model_name}...")
            # Kontextusos teszt
            res_with_ctx = ask_llama_with_context(model_name, mention, context, top_candidates)
            # Kontextus nélküli teszt
            res_no_ctx = ask_llama_without_context(model_name, mention, top_term_and_label_only_candidates)
            outputs_for_this_mention[model_name] = {"with_context": res_with_ctx,"without_context": res_no_ctx}
            print(f"      [{model_name}] Ctx: {res_with_ctx.get('decision')} | No-Ctx: {res_no_ctx.get('decision')}")
        # Mentés a duális benchmark naplóba
        log_benchmark_result(doc_id, mention, context, top_candidates,top_term_and_label_only_candidates, outputs_for_this_mention)

        # 4. ONTOLÓGIA FEJLESZTÉSE (Kizárólag a legerősebb modell - Llama 3.1 - KONTEXTUSOS döntése alapján!)
        llama_3_1_main_decision = outputs_for_this_mention["llama3.1"]["with_context"].get("decision", "NEW_CONCEPT")

        if append_to_ontology and llama_3_1_main_decision == "NEW_CONCEPT":
            print(f"   [!] Új koncepció elfogadva (Llama 3.1 alapján). Definíció generálása...")
            def_data = generate_strict_definition(mention, context)
            generated_def = def_data.get("generated_definition", f"A specialized term representing {mention}.")
            new_node = {"id": get_next_id(),"name": mention.title(),"definition": generated_def,"synonyms": []}
            CURRENT_ONTOLOGY.append(new_node)
            with open(ONTOLOGY_BASE_FILE, "w", encoding="utf-8") as f:
                json.dump(CURRENT_ONTOLOGY, f, indent=2, ensure_ascii=False)
    # Ha egy egész dokumentum összes szavát megvizsgáltuk az első 100-ból, elmentjük a progress-be# (A biztonság kedvéért a 100-as batch-en belül is mentünk haladást dokumentumonként)
    # Ez a rész ellenőrzi, hogy ez volt-e az utolsó feladat az adott cikkhez ebben a futásban
    remaining_in_doc = [j for j in test_jobs[idx:] if j["doc_id"] == doc_id]
    if not remaining_in_doc:
        processed_documents.add(doc_id)
        save_progress(processed_documents)
        print(f"=== {doc_id} szavai feldolgozva a batch-ben. Checkpoint mentve. ===")
        print("\nA 100-as ablációs benchmark sikeresen véget ért!")
        print(f"Az összehasonlító adatok a(z) '{PERFORMANCE_LOG_FILE}' fájlban érhetőek el.")

if __name__ == "__main__":
    run_benchmark_pipeline()