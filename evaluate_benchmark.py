import json
import os

PERFORMANCE_LOG_FILE_COMP = "results/minilm_pipeline_performance_log_llm_performance.jsonl"
PERFORMANCE_LOG_FILE = "results/minilm_pipeline_performance_log.jsonl"
LABEL_JSON = "labels.json"
ONTOLOGY_JSON = "ontology_rdf.json"


def load_ontology_mapping():
    """Reads ontology,and builds Name -> ID dictionary for checking."""
    if not os.path.exists(ONTOLOGY_JSON):
        print(f"ERROR: File '{ONTOLOGY_JSON}' missing. ")
        return {}
    with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        # { "temperature sensor": "ONT_002", "honey bee": "ONT_001" }
        return {item["name"].lower().strip(): item["id"] for item in data}


def load_gold_standard(ontology_mapping):
    """Reads Gold Standard and assigns expected IDs to mentions."""
    if not os.path.exists(LABEL_JSON):
        return {}
    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold_standard = {}
    for item in data:
        term = item["term"].lower().strip()
        label = item["label"].strip()
        name=str(item["name"]).strip()

        if label.lower() in ["no", "new_concept", "nil"]:
            gold_standard[term] = {"decision": "NEW_CONCEPT", "expected_id": None}
        else:
            # Megkeressük a várt osztály nevét az ontológia ID térképén
            expected_id = ontology_mapping.get(name.lower().strip())
            gold_standard[term] = {"decision": "EXISTING", "expected_id": expected_id, "expected_label_name": name}

    return gold_standard


def calculate_metrics(tp, tn, fp, fn, misclass):
    # Misclassified items should be counted among  errors (FP/FN).
    total = tp + tn + fp + fn + misclass
    accuracy = (tp + tn) / total if total > 0 else 0
    # Precision: How many exact matches from "EXISTING" class?
    precision = tp / (tp + fp + misclass) if (tp + fp + misclass) > 0 else 0
    recall = tp / (tp + fn + misclass) if (tp + fn + misclass) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy * 100, precision * 100, recall * 100, f1 * 100


def run_advanced_evaluation():
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        print(f"ERROR:  '{PERFORMANCE_LOG_FILE}' file missing!")
        return

    ontology_mapping = load_ontology_mapping()
    gold_standard = load_gold_standard(ontology_mapping)

    if not gold_standard:
        print(f"ERROR: Gold Standard could not be loaded from '{LABEL_JSON}'.")
        return

    total_records = 0
    stats = {}

    with open(PERFORMANCE_LOG_FILE, "r", encoding="utf-8") as f:
        lexical = []

        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            total_records += 1

            mention = record.get("mention", "").lower().strip()
            bench_results = record.get("benchmark_results", {})

            # Lekérjük az elvárt igazság adatcsomagot
            gold_item = gold_standard.get(mention, {"decision": "NEW_CONCEPT", "expected_id": None})
            gold_decision = gold_item["decision"]
            expected_id = gold_item["expected_id"]


            i=0
            for model_name, results in bench_results.items():
                if results.get("with_context", {}).get("decision","ERROR") == "LEXICAL_MATCH":
                    if i<1:
                        lexical.append(mention)
                    i+=1
                    continue
                elif results.get("with_context", {}).get("decision","ERROR") == "ERROR":
                    print("error")
                results.get("with_context", {})
                if model_name not in stats:
                    stats[model_name] = {
                        "agreement": 0, "total": 0,
                        "ctx_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "misclass_err": 0},
                        "no_ctx_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "misclass_err": 0},
                        "no_ctx_misclass" : [],
                        "ctx_misclass" : [],
                        "no_ctx_misclass_fp" : [],
                        "ctx_misclass_fp" : [],
                        "no_ctx_misclass_fn" : [],
                        "ctx_misclass_fn" : [],
                        "no_ctx_tn" : [],
                        "ctx_tn" : [],
                        "no_ctx_tp" : [],
                        "ctx_tp" : [],

                    }

                ctx_output = results.get("with_context", {})
                no_ctx_output = results.get("without_context", {})

                ctx_dec = ctx_output.get("decision")
                no_ctx_dec = no_ctx_output.get("decision")

                if ctx_dec == "ERROR" or no_ctx_dec == "ERROR":
                    continue

                stats[model_name]["total"] += 1
                if ctx_dec == no_ctx_dec:
                    stats[model_name]["agreement"] += 1

                # --- 1. KIÉRTÉKELÉS: KONTEXTUSOS ÁG (STRICT CONCEPT ALIGNMENT) ---
                ctx_matched_id = ctx_output.get("matched_concept_id")

                if gold_decision == "EXISTING":
                    if ctx_dec == "EXISTING":
                        # SZIGORÚ ELLENŐRZÉS: Elszúrtuk-e az ID-t meglévő döntésen belül?
                        if ctx_matched_id == expected_id and expected_id is not None:
                            stats[model_name]["ctx_matrix"]["tp"] += 1  # Valódi, tökéletes találat!
                            stats[model_name]["ctx_tp"].append("'"+mention+"'->"+ctx_matched_id)
                        else:
                            stats[model_name]["ctx_matrix"][
                                "misclass_err"] += 1  # Meglévőnek látta, de ROSSZ osztályt választott!
                            if not ctx_matched_id:
                                id="None"
                            else:
                                id=ctx_matched_id
                            stats[model_name]["ctx_misclass"].append("'"+mention+"': expected_id: " + str(expected_id) + "->" +id)

                    else:
                        stats[model_name]["ctx_matrix"]["fn"] += 1
                        stats[model_name]["ctx_misclass_fn"].append("'"+mention+"': expected_id: " + str(expected_id) + "->" +"None")
                else:  # gold_decision == "NEW_CONCEPT"
                    if ctx_dec == "NEW_CONCEPT":
                        stats[model_name]["ctx_matrix"]["tn"] += 1
                        stats[model_name]["ctx_tn"].append(mention)
                    else:
                        stats[model_name]["ctx_matrix"]["fp"] += 1
                        stats[model_name]["ctx_misclass_fp"].append("'"+mention+"': expected_id: " + "NONE" + "->" +ctx_matched_id)


                # --- 2. KIÉRTÉKELÉS: KONTEXTUS NÉLKÜLI ÁG (STRICT CONCEPT ALIGNMENT) ---
                no_ctx_matched_id = no_ctx_output.get("matched_concept_id")

                if gold_decision == "EXISTING":
                    if no_ctx_dec == "EXISTING":
                        if no_ctx_matched_id == expected_id and expected_id is not None:
                            stats[model_name]["no_ctx_matrix"]["tp"] += 1
                            stats[model_name]["no_ctx_tp"].append("'"+mention+"'->"+no_ctx_matched_id)

                        else:
                            stats[model_name]["no_ctx_matrix"]["misclass_err"] += 1
                            if not no_ctx_matched_id:
                                id = "None"
                            else:
                                id = no_ctx_matched_id
                            stats[model_name]["no_ctx_misclass"].append("'"+mention+"': expected_id: " + str(expected_id) + "->" +id)
                    else:
                        stats[model_name]["no_ctx_matrix"]["fn"] += 1
                        stats[model_name]["no_ctx_misclass_fn"].append("'"+mention+"': expected_id: " + str(expected_id) + "->" +"None")

                else:  # gold_decision == "NEW_CONCEPT"
                    if no_ctx_dec == "NEW_CONCEPT":
                        stats[model_name]["no_ctx_matrix"]["tn"] += 1
                        stats[model_name]["no_ctx_tn"].append(mention)

                    else:
                        stats[model_name]["no_ctx_matrix"]["fp"] += 1
                        stats[model_name]["no_ctx_misclass_fp"].append("'"+mention+"': expected_id: " + "NONE" + "->" +no_ctx_matched_id)


    # Print report
    print("\n" + "=" * 70)
    print("   FINAL PERFORMANCE REPORT: STRICT CONCEPT ALIGNMENT (WITH AND WITHOUT CONTEXT) ")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(f"   LEXICAL MATCH({len(lexical)}): {'; '.join(lexical)}) :")
    print("=" * 70)

    for model_name, s in stats.items():
        if s["total"] == 0: continue
        print(f"\n>>> MODELL: {model_name.upper()} ({s['total']} valid tasks) <<<")

        # 1. With context report
        cm = s["ctx_matrix"]
        c_acc, c_pr, c_rec, c_f1 = calculate_metrics(cm["tp"], cm["tn"], cm["fp"], cm["fn"], cm["misclass_err"])
        print(f"\n   [RUN 1: WITH CONTEXT + DEFINITION]")
        print(
            f"   Matrix: Clean_TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, Misclass={cm['misclass_err']}")
        print(f"   - Accuracy   : {c_acc:.2f}%")
        print(f"   - Precision  : {c_pr:.2f}% (Including misclassification!)")
        print(f"   - Recall   : {c_rec:.2f}%")
        print(f"   - F1-Score  : {c_f1:.2f}%")
        print("\n" + "." * 70)
        print(f"   - Misclass : {"; ".join(s['ctx_misclass'])}")
        print(f"   - FP: {"; ".join(s['ctx_misclass_fp'])}")
        print(f"   - FN: {"; ".join(s['ctx_misclass_fn'])}")
        print(f"   - TN: {"; ".join(s['ctx_tn'])}")
        print(f"   - TP: {"; ".join(s['ctx_tp'])}")


        # 2. Without context report
        nm = s["no_ctx_matrix"]
        n_acc, n_pr, n_rec, n_f1 = calculate_metrics(nm["tp"], nm["tn"], nm["fp"], nm["fn"], nm["misclass_err"])
        print(f"\n   [RUN 2: WITHOUT CONTEXT + DEFINITION]")
        print(
            f"   Matrix: Clean_TP={nm['tp']}, TN={nm['tn']}, FP={nm['fp']}, FN={nm['fn']}, Misclass={nm['misclass_err']}")
        print(f"   - Accuracy : {n_acc:.2f}%")
        print(f"   - Precision: {n_pr:.2f}%")
        print(f"   - Recall   : {n_rec:.2f}%")
        print(f"   - F1-Score : {n_f1:.2f}%")
        print("\n" + "." * 70)
        print(f"   - Misclass : {"; ".join(s['no_ctx_misclass'])}")
        print(f"   - FP: {"; ".join(s['no_ctx_misclass_fp'])}")
        print(f"   - FN: {"; ".join(s['no_ctx_misclass_fn'])}")
        print(f"   - TN: {"; ".join(s['no_ctx_tn'])}")
        print(f"   - TP: {"; ".join(s['no_ctx_tp'])}")
        print("\n" + "-" * 70)





def run_efficiency_evaluation():
    if not os.path.exists(PERFORMANCE_LOG_FILE_COMP):
        print(f"Hiba: A '{PERFORMANCE_LOG_FILE_COMP}' fájl nem található!")
        return

    # Adatszerkezet a mérőszámok gyűjtéséhez
    # { model_name: { "with_context": { "time": [], "prompt_tokens": [], "gen_tokens": [] }, ... } }
    metrics = {}
    total_records = 0

    with open(PERFORMANCE_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                # Csak azokat a sorokat dolgozzuk fel, amikben már benne van a fogyasztási mátrix
                if "consumption_metrics" not in record:
                    continue

                total_records += 1
                cons_data = record["consumption_metrics"]

                for model_name, modes in cons_data.items():
                    if model_name not in metrics:
                        metrics[model_name] = {
                            "with_context": {"time": [], "prompt_tokens": [], "gen_tokens": []},
                            "without_context": {"time": [], "prompt_tokens": [], "gen_tokens": []}
                        }

                    # Kontextusos adatok gyűjtése
                    w_ctx = modes.get("with_context", {})
                    if w_ctx.get("prompt_tokens", 0) > 0:  # Biztonsági szűrés a hibás futások ellen
                        metrics[model_name]["with_context"]["time"].append(w_ctx.get("total_time_sec", 0))
                        metrics[model_name]["with_context"]["prompt_tokens"].append(w_ctx.get("prompt_tokens", 0))
                        metrics[model_name]["with_context"]["gen_tokens"].append(w_ctx.get("generated_tokens", 0))

                    # Kontextus nélküli adatok gyűjtése
                    no_ctx = modes.get("without_context", {})
                    if no_ctx.get("prompt_tokens", 0) > 0:
                        metrics[model_name]["without_context"]["time"].append(no_ctx.get("total_time_sec", 0))
                        metrics[model_name]["without_context"]["prompt_tokens"].append(no_ctx.get("prompt_tokens", 0))
                        metrics[model_name]["without_context"]["gen_tokens"].append(no_ctx.get("generated_tokens", 0))

            except Exception as e:
                pass  # Átugorjuk a régebbi struktúrájú sorokat

    if total_records == 0:
        print("Nem találtam olyan naplóbejegyzést, ami tartalmazna fogyasztási adatokat.")
        return

    # =====================================================================
    # ERÖDMÉNYEK ÖSSZESÍTÉSE ÉS KIÍRATÁSA
    # =====================================================================
    print("\n" + "=" * 70)
    print("        LLM ERŐFORRÁS-FOGYASZTÁSI ÉS EFFEKTIVITÁSI JELENTÉS")
    print("=" * 70)
    print(f"Feldolgozott esetek száma: {total_records}")

    for model_name, modes in metrics.items():
        print("\n" + "#" * 60)
        print(f" MODELL: {model_name.upper()}")
        print("#" * 60)

        # 1. KONTEXTUSSAL ÁTLAGOK
        w_ctx = modes["with_context"]
        if w_ctx["prompt_tokens"]:
            avg_w_time = sum(w_ctx["time"]) / len(w_ctx["time"])
            avg_w_prompt = sum(w_ctx["prompt_tokens"]) / len(w_ctx["prompt_tokens"])
            avg_w_gen = sum(w_ctx["gen_tokens"]) / len(w_ctx["gen_tokens"])
            total_w_time = sum(w_ctx["time"])

            print(f"-> FUTTATÁS 1: KONTEXTUSSAL + PÉLDÁKKAL (Few-Shot)")
            print(f"   - Átlagos bemeneti (Prompt) méret : {avg_w_prompt:.1f} token / kérés")
            print(f"   - Átlagos legenerált válasz méret: {avg_w_gen:.1f} token / kérés")
            print(f"   - Átlagos válaszidő (Latency)    : {avg_w_time:.3f} másodperc")
            print(f"   - Összesített futási idő         : {total_w_time:.2f} másodperc (~{total_w_time / 60:.1f} perc)")

        # 2. KONTEXTUS NÉLKÜL ÁTLAGOK
        no_ctx = modes["without_context"]
        if no_ctx["prompt_tokens"]:
            avg_n_time = sum(no_ctx["time"]) / len(no_ctx["time"])
            avg_n_prompt = sum(no_ctx["prompt_tokens"]) / len(no_ctx["prompt_tokens"])
            avg_n_gen = sum(no_ctx["gen_tokens"]) / len(no_ctx["gen_tokens"])
            total_n_time = sum(no_ctx["time"])

            print(f"\n-> FUTTATÁS 2: KONTEXTUS NÉLKÜL (Tiszta gép-tudás)")
            print(f"   - Átlagos bemeneti (Prompt) méret : {avg_n_prompt:.1f} token / kérés")
            print(f"   - Átlagos legenerált válasz méret: {avg_n_gen:.1f} token / kérés")
            print(f"   - Átlagos válaszidő (Latency)    : {avg_n_time:.3f} másodperc")
            print(f"   - Összesített futási idő         : {total_n_time:.2f} másodperc (~{total_n_time / 60:.1f} perc)")

        # 3. GAZDASÁGOSSÁGI ÖSSZEHASONLÍTÁS (HEURISZTIKA)
        if w_ctx["prompt_tokens"] and no_ctx["prompt_tokens"]:
            token_saved_pct = ((avg_w_prompt - avg_n_prompt) / avg_w_prompt) * 100
            time_saved_pct = ((total_w_time - total_n_time) / total_w_time) * 100

            print(f"\n=> EFFEKTIVITÁSI MÉRLEG (Gazdaságossági mutatók):")
            print(f"   - A kontextus elhagyásával a bemeneti adatmennyiség {token_saved_pct:.1f}%-kal CSÖKKENT.")
            print(f"   - A teljes számítási idő a laptopodon {time_saved_pct:.1f}%-kal LETT RÖVIDEBB.")

    print("\n" + "=" * 70)




if __name__ == "__main__":
    run_advanced_evaluation()
    run_efficiency_evaluation()
