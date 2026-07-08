import json
import os
import torch
import gc
from transformers import AutoTokenizer, AutoModel

LABEL_JSON = "labels.json"
ONTOLOGY_JSON = "ontology_rdf.json"

nomber_of_tops=5

MODELS_TO_TEST = {
    "General BERT (MiniLM)": "sentence-transformers/all-MiniLM-L6-v2",
    "SciBERT (AllenAI)": "allenai/scibert_scivocab_uncased",
    "BiomedBERT (Microsoft)": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
}

# KÜSZÖBÉRTÉK A "NO" (NIL) ESETEKHEZ
# Ha a hasonlóság ez alatt van egy "no" kifejezésnél, a BERT helyesen ismerte fel, hogy új a fogalom.
THRESHOLD = 0.65


def load_data():
    if not os.path.exists(LABEL_JSON) or not os.path.exists(ONTOLOGY_JSON):
        print("ERROR:  'label.json' or 'expanded_ontology_base.json' missing!")
        return None, None
    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    return ground_truth, ontology


def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).flatten()


def evaluate_embedding_model(model_name, hf_path, ground_truth, ontology, top_n=5):
    print(f"\n-> {model_name} loading and encoding...")
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    model = AutoModel.from_pretrained(hf_path)

    onto_vectors = {}
    for concept in ontology:
        onto_vectors[concept["name"]] = get_embedding(concept["name"], tokenizer, model)

    hits = 0
    not_none_hits = 0
    total_existing= 0
    total_valid = 0
    detailed_results = []
    existing_label_results=[]

    for item in ground_truth:
        term = item["term"]
        expected_label = item["label"].strip()  # Lehet egy osztálynév, vagy "no"
        if expected_label.lower() == "no":
            expected_name = "no"
        else:
            #print(item["name"])
            expected_name = item["name"].strip()  # Lehet egy osztálynév, vagy "no"

        term_vector = get_embedding(term, tokenizer, model)

        scores = []
        for concept_name, concept_vec in onto_vectors.items():
            sim = torch.nn.functional.cosine_similarity(term_vector, concept_vec, dim=0).item()
            scores.append((concept_name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [cand_name for cand_name, score in scores[:top_n]]
        highest_score = scores[0][1] if scores else 0.0

        # --- UK ÉRTÉKELÉSI LOGIKA A "NO" CÍMKÉRE ---
        if expected_label.lower() == "no":
            # Ha "no", akkor az a SIKER (Hit), ha a legmagasabb hasonlóság a küszöbérték ALATT van
            # Vagyis a BERT helyesen jelzi, hogy ez a szó nem hasonlít semmire a meglévő adatbázisban
            is_hit = highest_score < THRESHOLD
        else:
            # Ha valódi osztálynév, akkor a megszokott módon a Top-N-ben kell lennie
            is_hit = expected_name in top_candidates
            not_none_hits+=is_hit
            total_existing+=1
            score_true = 0.0
            true_location=-1
            if is_hit:
                for i, (cand_name, score) in enumerate(scores):
                    if cand_name == expected_name:
                        score_true= score
                        true_location=i
            existing_label_results.append({
            "term": term, "expected_label": expected_label, "expected_name": expected_name,
            "top_candidates": top_candidates, "true_concept_score": score_true,"true_concept_rank":true_location, "is_hit": is_hit
        })


        if is_hit:
            hits += 1
        total_valid += 1

        detailed_results.append({
            "term": term, "expected_label": expected_label, "expected_name": expected_name,
            "top_candidates": top_candidates, "highest_score": highest_score, "is_hit": is_hit
        })

    del model;
    del tokenizer;
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    recall_score = (hits / total_valid) * 100 if total_valid > 0 else 0
    true_label_recall_score = (not_none_hits / total_existing) * 100 if total_existing > 0 else 0
    return recall_score, detailed_results,true_label_recall_score,existing_label_results


if __name__ == "__main__":
    ground_truth, ontology = load_data()
    if ground_truth and ontology:
        print(f" {len(ground_truth)} test expressions loaded. Size of ontology: {len(ontology)} concepts.")

        final_summary = {}
        final_summary_true_label={}
        all_model_failures = {item["term"]: [] for item in ground_truth}
        all_model_failures_true_label = {item["term"]: [] for item in ground_truth}

        for display_name, hf_path in MODELS_TO_TEST.items():
            recall, detailed, true_label_recall, existing_label_detailed = evaluate_embedding_model(display_name, hf_path, ground_truth, ontology, top_n=5)
            final_summary[display_name] = recall
            final_summary_true_label[display_name] = true_label_recall
            for res in detailed:
                if not res["is_hit"]:
                    all_model_failures[res["term"]].append(
                        (display_name, res["expected_label"], res["top_candidates"], res["highest_score"]))
            for res in existing_label_detailed:
                if not res["is_hit"]:
                    all_model_failures_true_label[res["term"]].append(
                        (display_name, res["expected_label"], res["top_candidates"], res["true_concept_score"], res["true_concept_rank"]))

        print("\n" + "=" * 65)
        print(f"     BERT EMBEDDING EVALUATION REPORT (Recall@5 / Threshold={THRESHOLD})")
        print("=" * 65)
        print("Global recall success of models (without NIL cases ):\n")
        for model_name, score in final_summary.items():
            print(f" - {model_name:<25}: {score:.2f}%")



        print("\n" + "-" * 65)
        print("   RESEARCH BLINDSPOT: WORDS ON WHICH ALL  3 MODELS FAILED")
        print("-" * 65)
        print("   All errors:")

        hard_failures_count = 0
        for term, failures in all_model_failures.items():
            print(f"{term} -> {failures}")
            if len(failures) == len(MODELS_TO_TEST):
                hard_failures_count += 1
                first_fail = failures[0]  # Első modell adatai a kiíratáshoz
                print(f"{hard_failures_count}. Expression: '{term}'")
                print(f"   - Expected state: '{first_fail[1]}'")
                if first_fail[1].lower() == "no":
                    print(
                        f"   - Reason of error: Too high score at the modell (for example. {first_fail[3]:.4f}) for an unrelated concept.")
                else:
                    print(f"   - Reason of error: Correct class not in  Top-5. Suggestions: {first_fail[2]}")
                print()

        print(f"Altogether {hard_failures_count} critical blindspot in the system.")
        print("=" * 65)


        print("\n" + "=" * 65)
        print(f"     BERT EMBEDDING EVALUATION REPORT ( True Label Recall@5 )")
        print("=" * 65)
        print("Global recall success of models (without NIL cases ):\n")
        for model_name, score in final_summary_true_label.items():
            print(f" - {model_name:<25}: {score:.2f}%")


        hard_failures_count = 0
        for term, failures in all_model_failures_true_label.items():
            print(f"{term} -> {failures}")
            if len(failures) == len(MODELS_TO_TEST):
                hard_failures_count += 1
                first_fail = failures[0]  # Első modell adatai a kiíratáshoz
                print(f"{hard_failures_count}. Expression: '{term}'")
                print(f"   - Expected state: '{first_fail[1]}'")

                print(f"   - Reason of error: Correct class not in  Top-5. Suggestions: {first_fail[2]}")
