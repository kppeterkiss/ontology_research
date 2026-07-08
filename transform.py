import json

import pandas as pd
from rdflib import Graph, RDF, RDFS, OWL, SKOS

def camel_case_split(str):
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    res= [''.join(word) for word in words]
    return ' '.join(res)

def rdf_to_concept_list(file_path, file_format="xml"):
    # Grafikon inicializálása és fájl beolvasása
    g = Graph()
    g.parse(file_path, format=file_format)

    concepts_list = []

    # Minden olyan elem keresése, ami Osztály (Class) vagy Fogalom (Concept)
    target_types = [OWL.Class, RDFS.Class, SKOS.Concept]
    counter = 0
    for s, p, o in g.triples((None, RDF.type, None)):
        if o in target_types:
            # URI konvertálása stringgé
            concept_uri = str(s)

            # Alapértelmezett értékek
            definition = "Nincs definíció"
            label = concept_uri.split('#')[-1].split('/')[-1]  # Egyszerűsített név az URI-ból

            # Definíció vagy leírás keresése (RDFS comment vagy SKOS definition)
            for _, _, desc in g.triples((s, RDFS.comment, None)):
                definition = str(desc)
            for _, _, desc in g.triples((s, SKOS.definition, None)):
                definition = str(desc)

            # Ember által olvasható név keresése (opcionális)
            for _, _, lbl in g.triples((s, RDFS.label, None)):
                label = str(lbl)

            # Szótár hozzáadása a listához
            concepts_list.append({
                "id": label,#f"RDF_{str(counter).zfill(3)}" ,
                "name": camel_case_split(label).replace('_', '').title(),
                "uri": concept_uri,
                "definition": definition
            })
            counter += 1

    return concepts_list

def read_label_xls(label_file_path,ontology_json_path):
    # Read all sheets into a dictionary of DataFrames
    all_sheets = pd.read_excel(label_file_path, sheet_name=[1,2,3,4])

    label_name_df=pd.read_json(ontology_json_path)

    # Combine all DataFrames into a single DataFrame
    df = pd.concat(all_sheets.values(), ignore_index=True)
    df=df[['term','label_in_ontology_if_exist']]
    df = pd.merge(df,label_name_df, left_on='label_in_ontology_if_exist', right_on='id', how='left')
    df.rename(columns={"label_in_ontology_if_exist":"label"},inplace=True)
    df.drop(columns=['id','definition','uri'],inplace=True)
    df = df.to_dict(orient="records")

    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(df, f, indent=2, ensure_ascii=False)


def xls_to_json(file_path):
    qwe = pd.read_excel(file_path,sheet_name=1)

    qwe.rename(columns={"Gold-standard term":"matched_concepts","term":"matched_concepts","sentence_context":"text","Cleaned sentence context":"text","Filename":"doc_id","source_container":"doc_id"},inplace=True)
    qwe = qwe.to_dict(orient="records")
    for i in qwe:
        i['matched_concepts']=[i['matched_concepts']]
    with open(file_path.replace(".xlsx",".json"), "w", encoding="utf-8") as f:
        json.dump(qwe, f, indent=2, ensure_ascii=False)



# Használati példa:
# eredmény = rdf_to_concept_list("ontologia.owl", file_format="xml")
# print(eredmény)
if __name__ == "__main__":
    read_label_xls("beekeeping_corpus/xls/gold standard list for precision_beekeeping_benchmark_100_terms_labels.xlsx","ontology_rdf.json")
    '''
    xls_to_json("beekeeping_corpus/xls/gold_standard_multi_word_noun_phrase_contexts 1.xlsx")
    xls_to_json("beekeeping_corpus/xls/gold_standard_single_noun_contexts 1.xlsx")
    res = rdf_to_concept_list("beekeeping_corpus/rdf/v2PBO.rdf", file_format="xml")
    with open("ontology_rdf.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    '''