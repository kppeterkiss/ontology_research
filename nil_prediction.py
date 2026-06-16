import json
import ollama
import torch
from sentence_transformers import SentenceTransformer, util

# =====================================================================
# 1. LOCAL INITIALIZATION (Perfect for Laptop, Scales to HPC)
# =====================================================================
# This embedding model is tiny (120MB) and runs instantly on a laptop CPU/GPU
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Tiny sample ontology
EXISTING_ONTOLOGY = [
    {"id": "ONT_001", "name": "Honey Bee",
     "definition": "A stinging winged insect that lives in organized colonies and produces wax and honey."},
    {"id": "ONT_002", "name": "Bee Smoker",
     "definition": "A device used in beekeeping to calm honey bees by puffing smoke into the hive."},
    {"id": "ONT_003", "name": "Wax Moth",
     "definition": "A destructive hive pest whose larvae eat and ruin beeswax combs."}
]

# Pre-calculate ontology embeddings (On HPC, you would load these from a vector database)
ontology_texts = [f"{item['name']}: {item['definition']}" for item in EXISTING_ONTOLOGY]
ontology_embeddings = embedding_model.encode(ontology_texts, convert_to_tensor=True)


# =====================================================================
# STAGE 1: DENSE VECTOR SEARCH (Realistic Candidate Retrieval)
# =====================================================================
def get_top_candidates_dense(new_mention, context, top_n=2):
    # Combine mention and context for a rich semantic query
    query_text = f"Term: {new_mention}. Context: {context}"
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True)

    # Compute cosine similarities via PyTorch tensors
    cosine_scores = util.cos_sim(query_embedding, ontology_embeddings)[0]

    # Get top N indices
    top_results = torch.topk(cosine_scores, k=top_n)

    candidates = []
    for score, idx in zip(top_results.values, top_results.indices):
        item = EXISTING_ONTOLOGY[idx.item()]
        candidates.append(item)
    return candidates


import torch
from sentence_transformers import SentenceTransformer, util

# Előre betanított, univerzális modell betöltése (nem kell finomhangolni!)
model = SentenceTransformer('all-MiniLM-L6-v2')


def find_candidates_combined(new_mention, context_sentence, ontology, top_n=2):
    """
    Kombinálja a Kifejezés-Kifejezés és Kontextus-Definíció hasonlóságot.
    """
    candidates_with_scores = []

    # 1. Beágyazzuk a keresett kifejezést és a környező kontextust
    mention_embedding = model.encode(new_mention, convert_to_tensor=True)
    context_embedding = model.encode(context_sentence, convert_to_tensor=True)

    for concept in ontology:
        # 2. Beágyazzuk az ontológia aktuális elemét (Név + Definíció)
        concept_name_embedding = model.encode(concept["name"], convert_to_tensor=True)
        concept_def_embedding = model.encode(concept["definition"], convert_to_tensor=True)

        # 3. Kiszámoljuk a két különálló koszinusz-hasonlóságot
        term_score = util.cos_sim(mention_embedding, concept_name_embedding).item()
        context_score = util.cos_sim(context_embedding, concept_def_embedding).item()

        # 4. Kombináljuk a pontszámokat (pl. 60% súly a kifejezésnek, 40% a kontextusnak)
        combined_score = (term_score * 0.6) + (context_score * 0.4)

        candidates_with_scores.append({
            "concept": concept,
            "score": combined_score
        })

    # Rendezés a kombinált pontszám alapján csökkenő sorrendben
    candidates_with_scores.sort(key=lambda x: x["score"], reverse=True)

    # Visszaadjuk a Top N legmeghatározóbb koncepciót
    return [item["concept"] for item in candidates_with_scores[:top_n]]


# =====================================================================
# STAGE 2: OLLAMA EVALUATION (NIL Prediction Gatekeeper)
# =====================================================================
def evaluate_nil_prediction(model_name, new_mention, context_sentence, candidates):
    candidates_string = "".join(
        [f"- [ID: {c['id']}] Name: {c['name']} | Definition: {c['definition']}\n" for c in candidates])

    prompt = f"""
    You are an expert ontology engineer specializing in beekeeping.
    Target Term: "{new_mention}"
    Context: "{context_sentence}"

    Existing options:
    {candidates_string}

    Task: Does "{new_mention}" map to an existing option? If it is completely different, select "NEW_CONCEPT".
    Respond strictly in JSON:
    {{
      "decision": "EXISTING" or "NEW_CONCEPT",
      "matched_concept_id": "ID string or null"
    }}
    """

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
        format="json"
    )
    return json.loads(response['message']['content'])


# =====================================================================
# RUN THE LAPTOP TEST
# =====================================================================
if __name__ == "__main__":
    # Use the lightweight 8B model on your laptop
    # On the HPC server, you will just change this string to "llama3.1:70b" or "mixtral"
    LOCAL_LLM = "llama3.1"

    test_mention = "Apis mellifera"
    test_context = "We observed the Apis mellifera returning to the hive with heavy pollen baskets."

    # 1. Retrieve candidates using dense embeddings
    candidates = get_top_candidates_dense(test_mention, test_context)

    # 2. Gatekeep with local LLM
    final_decision = evaluate_nil_prediction(LOCAL_LLM, test_mention, test_context, candidates)
    print(json.dumps(final_decision, indent=2))
