import json
import os
import torch
import ollama
from sentence_transformers import SentenceTransformer, util


# =====================================================================
# 1. RUN MODE
# =====================================================================
RUN_MODE = "train"
MANUAL_WHITELIST = ["worker", "work","word","yield","wound"]

# =====================================================================
# 1. KONFIGURÁCIÓ ÉS FÁJLUTAK
# =====================================================================
INPUT_JSON = "filtered_paragraphs_train.json"
ONTOLOGY_BASE_FILE = "ontology_base.json"
EXPANDED_ONTOLOGY_BASE_FILE = "expanded_ontology_base.json"

PERFORMANCE_LOG_FILE = "pipeline_performance_log.jsonl"


BERT_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'llama3.2'

print("Modellek betöltése a memóriába...")
embedding_model = SentenceTransformer(BERT_MODEL_NAME)

# =====================================================================
# INKREMENTÁLIS LOGIKA: ONTOLÓGIA BETÖLTÉSE / INICIALIZÁLÁSA
# =====================================================================
if os.path.exists(EXPANDED_ONTOLOGY_BASE_FILE) and RUN_MODE!='train':
    with open(EXPANDED_ONTOLOGY_BASE_FILE, "r", encoding="utf-8") as f:
        CURRENT_ONTOLOGY = json.load(f)
elif os.path.exists(ONTOLOGY_BASE_FILE) and RUN_MODE!='train':
    with open(ONTOLOGY_BASE_FILE, "r", encoding="utf-8") as f:
        CURRENT_ONTOLOGY = json.load(f)
    print(f"[Inkrementális mód] Sikeresen beolvasva {len(CURRENT_ONTOLOGY)} korábbi koncepció.")
else:
    # Ha még sosem futott, létrehozzuk az induló (rudimentális) listát
    print("[Kezdeti mód] Nem található korábbi mentés. Alap ontológia inicializálása...")
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

# Számláló beállítása az új ID-k generálásához a meglévő elemek alapján
if CURRENT_ONTOLOGY:
    last_id = CURRENT_ONTOLOGY[-1]["id"]
    new_concept_counter = int(last_id.split("_")[1]) + 1
else:
    new_concept_counter = 1


# =====================================================================
# 2. FÁZIS: KOMBINÁLT BERT HASONLÓSÁG
# =====================================================================
def get_top_candidates_combined(mention, context, ontology, top_n=2):
    if not ontology:
        return []

    scored_candidates = []
    mention_emb = embedding_model.encode(mention, convert_to_tensor=True)
    context_emb = embedding_model.encode(context, convert_to_tensor=True)

    for concept in ontology:
        concept_name_emb = embedding_model.encode(concept["name"], convert_to_tensor=True)
        concept_def_emb = embedding_model.encode(concept["definition"], convert_to_tensor=True)

        term_score = util.cos_sim(mention_emb, concept_name_emb).item()
        context_score = util.cos_sim(context_emb, concept_def_emb).item()

        combined_score = (term_score * 0.6) + (context_score * 0.4)

        scored_candidates.append({
            "concept": concept,
            "score": combined_score,
            "term_score": term_score,
            "context_score": context_score
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:top_n]


# =====================================================================
# 3. FÁZIS: OLLAMA ÉS DEFINÍCIÓ GENERÁLÁS
# =====================================================================
def ask_llama_nil_prediction(mention, context, candidates):
    if not candidates:
        # Ha az ontológia teljesen üres, automatikusan új koncepcióként kezeljük
        return {"decision": "NEW_CONCEPT", "matched_concept_id": None, "reasoning": "Ontology is empty."}

    candidates_str = ""
    for idx, c in enumerate(candidates, 1):
        candidates_str += f"{idx}. [ID: {c['concept']['id']}] Name: {c['concept']['name']} | Definition: {c['concept']['definition']}\n"

    prompt = f"""
    You are an expert ontology engineer specializing in apiculture (beekeeping).
    Target Term to evaluate: "{mention}"
    Context sentence: "{context}"

    Top candidate concepts currently in our ontology:
    {candidates_str}

    Task: Does "{mention}" correspond exactly to one of the existing concepts listed above?
    If it represents a fundamentally different concept, tool, or pest/disease, you MUST select "NEW_CONCEPT".

    Respond ONLY with a raw JSON object matching this structure:
    {{
      "decision": "EXISTING" or "NEW_CONCEPT",
      "matched_concept_id": "The ID string if existing, else null",
      "reasoning": "A concise, one-sentence logical explanation for your choice."
    }}
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_ctx": 2048  # <--- EZ AZ OPTIMALIZÁLÁS: Lekorlátozza a kontextust 2048 tokenre
            },
            keep_alive=0,  # <--- EZ AZ EXTRA: Azonnal felszabadítja a RAM-ot a kérés után!
            format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": f"Ollama hiba (NIL): {str(e)}"}


def generate_definition_for_new_concept(mention, context):
    prompt = f"""
    You are an expert lexicographer and beekeeping specialist.
    Write a professional, objective, 1-sentence dictionary definition for the term "{mention}".
    Use this context: "{context}"

    Respond ONLY with a raw JSON object matching this structure:
    {{
      "concept_name": "{mention}",
      "generated_definition": "The professional 1-sentence definition here."
    }}
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_ctx": 2048  # <--- EZ AZ OPTIMALIZÁLÁS: Lekorlátozza a kontextust 2048 tokenre
            },
            keep_alive=0,  # <--- EZ AZ EXTRA: Azonnal felszabadítja a RAM-ot a kérés után!
            format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": f"Ollama hiba (Definíció): {str(e)}"}


# =====================================================================
# 4. KIMENET MENTÉS ÉS PERFORMANCE NAPLÓZÁS
# =====================================================================
def log_performance_entry(source_doc, mention, context, bert_candidates, llama_output):
    """
    Hozzáfűzi a döntést egy JSON Lines fájlhoz (.jsonl).
    Ez a formátum biztonságos, ha a script megszakadna, az eddigi logok megmaradnak.
    """

    try:
        log_entry = {
            "source_doc": source_doc,
            "mention": mention,
            "context": context,
            "bert_top_suggestions": [
                {
                    "id": c["concept"]["id"],
                    "name": c["concept"]["name"],
                    "combined_score": round(c["score"], 4),
                    "term_score": round(c["term_score"], 4),
                    "context_score": round(c["context_score"], 4)
                } for c in bert_candidates
            ],
            "llama_decision": llama_output["decision"],
            "llama_matched_id": llama_output.get("matched_concept_id"),
            "llama_reasoning": llama_output.get("reasoning")
        }
    except Exception as e:
        log_entry = {
            "source_doc": source_doc,
            "mention": mention,
            "context": context,
            "bert_top_suggestions": [
                {
                    "id": c["concept"]["id"],
                    "name": c["concept"]["name"],
                    "combined_score": round(c["score"], 4),
                    "term_score": round(c["term_score"], 4),
                    "context_score": round(c["context_score"], 4)
                } for c in bert_candidates
            ],
            "llama_decision":None,
            "llama_matched_id": None,
            "llama_reasoning": llama_output
        }

    # Hozzáfűzés (append) mód, hogy az új futások ne töröljék a régi logokat
    with open(PERFORMANCE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# =====================================================================
# 5. FUTTATÓ MOTOR
# =====================================================================
def run_pipeline():
    global new_concept_counter

    if not os.path.exists(INPUT_JSON):
        print(f"Hiba: Nem található a '{INPUT_JSON}' fájl.")
        return

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

    def get_last_processed_doc_id():
        """Returns the source_doc from the last valid row in PERFORMANCE_LOG_FILE."""
        if not os.path.exists(PERFORMANCE_LOG_FILE):
            return None

        last_doc_id = None
        with open(PERFORMANCE_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    log_entry = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore an incomplete/corrupted final line if the process was interrupted.
                    continue

                last_doc_id = log_entry.get("source_doc")

        return last_doc_id

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        filtered_paragraphs = json.load(f)

    # Filter paragraphs based on doc_id prefix
    train_paragraphs = []
    test_paragraphs = []

    import re
    for item in filtered_paragraphs:
        doc_id = item.get("doc_id", "")
        # Match 3-digit number at the start of doc_id
        match = re.match(r'^(\d{3})', doc_id)
        if match:
            doc_number = int(match.group(1))
            if doc_number <= 111:
                train_paragraphs.append(item)
            else:
                test_paragraphs.append(item)
        else:
            # If no 3-digit prefix, default to test
            test_paragraphs.append(item)

    # Select dataset based on RUN_MODE
    if RUN_MODE == "train":
        filtered_paragraphs = train_paragraphs
        print(f"[TRAIN MODE] Using {len(train_paragraphs)} training paragraphs.")
    else:
        filtered_paragraphs = test_paragraphs
        print(f"[TEST MODE] Using {len(test_paragraphs)} test paragraphs.")

    last_processed_doc_id = get_last_processed_doc_id()

    if last_processed_doc_id:
        resume_index = None

        for idx, item in enumerate(filtered_paragraphs):
            if item.get("doc_id") == last_processed_doc_id:
                resume_index = idx + 1
                break

        if resume_index is not None:
            print(
                f"[RESUME] Last processed doc_id in '{PERFORMANCE_LOG_FILE}': "
                f"{last_processed_doc_id}. Resuming from item #{resume_index + 1}."
            )
            filtered_paragraphs = filtered_paragraphs[resume_index:]
        else:
            print(
                f"[RESUME WARNING] Last processed doc_id '{last_processed_doc_id}' "
                f"was not found in the selected {RUN_MODE} dataset. Starting from the beginning."
            )
    else:
        print("[RESUME] No previous performance log found. Starting from the beginning.")

    print(f"Összesen feldolgozandó: {len(filtered_paragraphs)} bekezdés.")
    fn = EXPANDED_ONTOLOGY_BASE_FILE
    #if RUN_MODE == "train":
    #    fn = ONTOLOGY_BASE_FILE

    processed_rounds_since_save = 0

    for idx, item in enumerate(filtered_paragraphs):
        context = item["text"]
        for mention in item["matched_concepts"]:
            print(f"\n[{idx + 1}/{len(filtered_paragraphs)}] Kifejezés: '{mention}'")

            # 1. LÉPÉS: BERT szűrés (A dinamikusan növekvő CURRENT_ONTOLOGY-t használja!)
            top_candidates = get_top_candidates_combined(mention, context, CURRENT_ONTOLOGY, top_n=2)

            # 2. LÉPÉS: Ollama NIL predikció
            llama_output = ask_llama_nil_prediction(mention, context, top_candidates)

            decision = llama_output.get("decision")
            print(f"   -> Döntés: {decision} | Indok: {llama_output.get('reasoning')}")

            # 3. LÉPÉS: Naplózás a teljesítmény kiértékeléséhez (Azonnal kimentődik)
            log_performance_entry(item["doc_id"], mention, context, top_candidates, llama_output)

            processed_rounds_since_save += 1

            # 4. LÉPÉS: Ha ÚJ, akkor definíciót gyártunk és elmentjük
            if decision == "NEW_CONCEPT":
                print(f"   [!] Új fogalom. Definíció generálása...")
                definition_output = generate_definition_for_new_concept(mention, context)
                generated_def = definition_output.get("generated_definition", "No definition generated.")

                new_id = f"ONT_{str(new_concept_counter).zfill(3)}"
                new_concept_data = {
                    "id": new_id,
                    "name": mention.title(),
                    "definition": generated_def
                }

                # Azonnali hozzáadás a memóriához (a következő cikk már látni fogja)
                CURRENT_ONTOLOGY.append(new_concept_data)
                new_concept_counter += 1

            if processed_rounds_since_save >= 10:
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(CURRENT_ONTOLOGY, f, indent=2, ensure_ascii=False)

                print(f"   [SAVE] State saved after {processed_rounds_since_save} processed rounds.")
                processed_rounds_since_save = 0

    with open(fn, "w", encoding="utf-8") as f:
        json.dump(CURRENT_ONTOLOGY, f, indent=2, ensure_ascii=False)

    print(f"\nKész! Az ontológia frissítve az '{fn}' fájlban.")
    print(f"A teljesítmény-értékelési napló elérhető itt: '{PERFORMANCE_LOG_FILE}'")

if __name__ == "__main__":
    run_pipeline()