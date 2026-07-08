import json
import os

PERFORMANCE_LOG_FILE = "results/pipeline_performance_log.jsonl"
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


if __name__ == "__main__":
    run_advanced_evaluation()
