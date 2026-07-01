import gc
import json
import os
import torch
import ollama
import spacy
from transformers import AutoTokenizer, AutoModel

# =====================================================================
# 1. PARAMÉTEREK ÉS KONFIGURÁCIÓ
# =====================================================================
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

    PERFORMANCE_LOG_FILE = "pipeline_performance_log.jsonl"
else:
    SEED_ONTOLOGY_FILE = "beekeeping_corpus/glossaries/merged_glossary_terms.json"
    INPUT_JSON = "filtered_paragraphs_test.json"
    ONTOLOGY_BASE_FILE = "expanded_ontology_base.json"
    PERFORMANCE_LOG_FILE = "pipeline_performance_log.jsonl"
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'

# Az Ollama LLM modellek a benchmark teszthez
TEST_MODELS = ["llama3.2", "mistral", "llama3.1"]
# PARAMÉTEREZHETŐ MODELLEK
# Itt adhatod meg, melyik HuggingFace BERT modellt akarod használni (pl. SciBERT, BioBERT)
EMBEDDING_MODEL_PARAM = "allenai/scibert_scivocab_uncased"



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


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed_docs", []))
    return set()


def save_progress(processed_docs_set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_docs": list(processed_docs_set)}, f, indent=2, ensure_ascii=False)


def check_lexical_exact_match(mention, ontology):
    mention_clean = mention.lower().strip()
    for concept in ontology:
        if concept["name"].lower().strip() == mention_clean:
            return concept
        if "synonyms" in concept:
            if any(s.lower().strip() == mention_clean for s in concept["synonyms"]):
                return concept
    return None


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
# STAGE 0: PARAMÉTEREZETT ELŐRE-BEÁGYAZÁS ÉS OOV DIAGNOSZTIKA
# =====================================================================
def precompute_vectors_and_oov(unique_mentions, ontology, hf_model_name):
    """
    Paraméterként kapott HuggingFace modellel előre kiszámolja a vektorokat,
    és detektálja az OOV hibákat, majd kiüríti a RAM-ot.
    """
    print(f"\n--- STAGE 0: Előzetes vektorizáció a(z) '{hf_model_name}' modellel ---")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(hf_model_name)

    precomputed_mentions = {}
    precomputed_ontology = {}
    oov_log = {}

    def get_emb(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).flatten()

    # 1. Egyedi kifejezések kódolása + OOV teszt
    for mention in unique_mentions:
        tokens = tokenizer.tokenize(mention)
        token_count = len(tokens)
        # OOV ha egy kifejezést sok apró sub-tokenre szabdal szét a modell szótára
        is_oov = token_count >= 3 and len(mention.split()) < token_count

        oov_log[mention] = {
            "sub_tokens": tokens,
            "token_count": token_count,
            "is_oov_anomaly": is_oov
        }
        precomputed_mentions[mention] = get_emb(mention)

    # 2. Aktuális ontológia nevek kódolása
    for concept in ontology:
        precomputed_ontology[concept["id"]] = get_emb(concept["name"])

    # 3. DRASZTIKUS RAM FELSZABADÍTÁS
    print("[Optimalizálás] Vektorok elmentve. BERT eltávolítása a memóriából...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("-> RAM sikeresen felszabadítva a Llama számára!")

    return precomputed_mentions, precomputed_ontology, oov_log


# Tiszta matematikai hasonlóság-számítás az előre mentett vektorokból (Nulla extra RAM)
def get_top_candidates_from_cache(mention, ontology, precomputed_mentions, precomputed_ontology, top_n=2):
    if not ontology: return []
    mention_emb = precomputed_mentions[mention]
    scored_candidates = []

    for concept in ontology:
        # Ha időközben jött létre új koncepció a futás alatt, aminek nincs előre számított vektora,
        # azt egy alap nullás hasonlósággal kezeljük, hogy ne szálljon el a kód (vagy adhatunk neki egy alap értéket)
        if concept["id"] not in precomputed_ontology:
            continue

        concept_emb = precomputed_ontology[concept["id"]]
        similarity = torch.nn.functional.cosine_similarity(mention_emb, concept_emb, dim=0).item()

        scored_candidates.append({
            "concept": concept,
            "score": similarity
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:top_n]


# =====================================================================
# LLM KÉRÉSEK ÉS NAPLÓZÁS (A korábbi, szigorított precíziós promptok)
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
    You are an expert apiculture lexicographer. Write a highly specific 1-sentence definition for "{mention}" based on this context: "{context}".
    Respond strictly in JSON: {{"generated_definition": "The crisp 1-sentence definition."}}
    """
    try:
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}],
                               options={"temperature": 0.1, "num_ctx": 2048}, keep_alive=0, format="json")
        return json.loads(response['message']['content'])
    except:
        return {"generated_definition": f"A specialized precision beekeeping term representing {mention}."}


def log_benchmark_dual_test(source_doc, mention, context, bert_candidates, model_outputs, oov_info,
                            embedding_model_used):
    log_entry = {
        "doc": source_doc,
        "mention": mention,
        "context": context,
        "embedding_model_used": embedding_model_used,
        "oov_diagnostic": oov_info,
        "bert_top_suggestions": [{"id": c["concept"]["id"], "name": c["concept"]["name"], "score": round(c["score"], 4)}
                                 for c in bert_candidates],
        "benchmark_results": model_outputs
    }
    with open(PERFORMANCE_LOG_FILE, "a", encoding="utf-8") as f: f.write(
        json.dumps(log_entry, ensure_ascii=False) + "\n")


#=====================================================================
#3. FÁZIS: AZ OPTIMALIZÁLT FUTTATÓ MOTOR
#=====================================================================
def run_benchmark_pipeline(embedding_model_param):
    global CURRENT_ONTOLOGY
    input_files = [INPUT_JSON] if isinstance(INPUT_JSON, str) else INPUT_JSON

    all_paragraphs = []
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Hiba: Hiányzik a '{input_file}'")
            return

        with open(input_file, "r", encoding="utf-8") as f:
            all_paragraphs.extend(json.load(f))
    processed_documents = load_progress()
    # 1. Dokumentum szintű aggregáció (Zajszűrés és kötegelés előkészítése)
    doc_mention_map = {}
    unique_mentions_set = set()
    for p in all_paragraphs:
        doc_id = p["doc_id"]
        #if doc_id in processed_documents:
        #    continue
        if doc_id not in doc_mention_map:
            doc_mention_map[doc_id] = {}
        for mention in p["matched_concepts"]:
            unique_mentions_set.add(mention)
            if mention not in doc_mention_map[doc_id] or len(p["text"]) > len(doc_mention_map[doc_id][mention]):
                doc_mention_map[doc_id][mention] = p["text"]
    # Kiválasztjuk az első 100 feladatot
    all_jobs = []
    for doc_id, mentions in doc_mention_map.items():
        for mention, large_context in mentions.items():
            all_jobs.append({"doc_id": doc_id, "mention": mention, "large_context": large_context})
    test_jobs = all_jobs[:100]
    if not test_jobs:
        print("\nNincs új feldolgozandó feladat.")
        return
    # Csak azokat a kifejezéseket vektorozzuk előre, amik ténylegesen bekerültek a 100-as batch-be
    active_mentions = set([j["mention"] for j in test_jobs])

    # =====================================================================
    # STAGE 0 MEGHÍVÁSA: A PARAMÉTEREZETT BERT FUTTATÁSA ÉS TÖRLÉSE
    # =====================================================================
    precomputed_mentions, precomputed_ontology, oov_log = precompute_vectors_and_oov(active_mentions, CURRENT_ONTOLOGY, embedding_model_param)
    print(f"\n--- STAGE 1 & 2: 6-irányú benchmark indítása ({len(test_jobs)} feladat) ---")
    for idx, job in enumerate(test_jobs, 1):
        doc_id = job["doc_id"]
        mention = job["mention"]
        large_context = job["large_context"]
        # Tűéles ablakozás
        context = extract_strict_sentence_window(mention, large_context, window_size=1)
        print(f"\n[{idx}/{len(test_jobs)}] Kifejezés: '{mention}' (Fájl: {doc_id})")
        # Lexikális Kapuőr
        lexical_match = check_lexical_exact_match(mention, CURRENT_ONTOLOGY)
        if lexical_match:
            print(f" [->] Lexikális egyezés megtalálva! ID: {lexical_match['id']}. LLM-ek átugorva.")
            log_benchmark_dual_test(doc_id, mention, context, "", "",
                                 {m: {"with_context": "LEXICAL", "without_context": "LEXICAL"} for m in TEST_MODELS},"NONE")

            continue

        # Jelölt állítás TISZTÁN A GYORSÍTÓTÁRBÓL (Nulla extra RAM terhelés)
        top_candidates = get_top_candidates_from_cache(
            mention, CURRENT_ONTOLOGY, precomputed_mentions, precomputed_ontology, top_n=5
        )
        # 6-irányú LLM benchmark futtatása
        model_outputs = {}
        for model_name in TEST_MODELS:
            print(f" -> Futtatás: {model_name}...")
            res_with_ctx = ask_llama_with_context(model_name, mention, context, top_candidates)
            res_no_ctx = ask_llama_without_context(model_name, mention, top_candidates)
            model_outputs[model_name] = {
                "with_context": res_with_ctx,
                "without_context": res_no_ctx
            }
            print(f" [{model_name}] Ctx: {res_with_ctx.get('decision')} | No-Ctx: {res_no_ctx.get('decision')}")
            # Duális naplózás az OOV információkkal együtt!
        log_benchmark_dual_test(doc_id, mention, context, top_candidates, model_outputs, oov_log[mention],
                             embedding_model_param)
        # Ontológia növelése (Llama 3.1 8B kontextusos döntése alapján)
        main_decision = model_outputs["llama3.1"]["with_context"].get("decision", "NEW_CONCEPT")
        if append_to_ontology and main_decision == "NEW_CONCEPT":
            print(f" [!] Új koncepció elfogadva. Definíció generálása...")
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
        # Haladás mentése dokumentumonként
        remaining_in_doc = [j for j in test_jobs[idx:] if j["doc_id"] == doc_id]
        if not remaining_in_doc:
            processed_documents.add(doc_id)
        save_progress(processed_documents)
        print(f"=== {doc_id} kész. Checkpoint mentve. ===")
    print("\nA parametrizált, RAM-védett benchmark sikeresen lefutott!")
if __name__ == "__main__":
    # Itt tudod változtatni az embedding modellt paraméterként:
    # Kipróbálhatod a "allenai/scibert_scivocab_uncased" vagy "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" modelleket is.
    run_benchmark_pipeline(EMBEDDING_MODEL_PARAM)