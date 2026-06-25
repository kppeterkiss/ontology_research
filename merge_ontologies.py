import json

PATHS=['expanded_ontology_base.json','beekeeping_corpus/glossaries/merged_glossary_terms.json']
OUTPUT_FILE='expanded_ontology_base.json'
def merge_ontologies():
    last_id=0
    l=[]
    for path in PATHS:
        json_data = json.load(open(path))
        if len(l)==0:
            l=json_data
            last_id=int(json_data[-1]['id'].replace("ONT_",""))

            continue
        for item in json_data:
            present=False
            for i in l:
                if item['name']==i['name']:
                    present=True
                    break
            if not present:
                item['id']="ONT_"+str(last_id+1).zfill(3)
                last_id+=1
                l.append(item)
    json.dump(l,open(OUTPUT_FILE,'w'),indent=2)


if __name__=="__main__":
    merge_ontologies()