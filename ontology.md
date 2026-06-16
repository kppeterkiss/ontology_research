# General pipeline

## Generating candidate terms: 
Automatic Term Extraction (ATE) or Termhood Estimation
1. **contrastive tf-idf:** Compute the relative frequency of a word in your beekeeping text and divide it by the frequency of the same word in the general text.
2.  **C-Value Method (For Multi-Word Terms)** -  C-Value algorithm is a standard ontology-learning formula designed specifically to extract multi-word technical terms by evaluating:

-  filter your text linguistically before running your contrastive math
```
[Raw Beekeeping Text]
         │
         ▼
[Step 1: Linguistic Filter] ───> Keep only Nouns, Adjective+Noun, and Noun+Noun phrases.
         │                       (e.g., discard verbs and adverbs like "slowly flying")
         ▼
[Step 2: Contrastive Match] ───> Compare the frequencies of these filtered phrases 
         │                       against a general background corpus.
         ▼
[High-Scoring Output]       ───> Candidate Terms: "wax moth", "royal jelly", "apis mellifera"
```
pre-calculated word frequency tables from public datasets:Google Books Ngram dataset: Free tables showing how often every English word appears across millions of books.The British National Corpus (BNC) or COCA: Ready-to-download frequency lists of general, everyday English vocabulary.

---

## Term Normalization and Grouping


To group different surfaces (raw text variants) that belong to the exact same term (canonical concept) in a highly specialized domain, you are essentially performing Synonym Extraction and Term Normalization. 
- surface grouping and definition retrieval at the same time.

**Step 1**: Structural Parsing & Co-occurrence Linking (Instant Grouping)

- Acronym Mapping: Scan your text for patterns like Candidate Term (Acronym) (e.g., "Layens Hive (LH)"). Group these surfaces immediately.
- Morphological Cleaning: Run a lemmatizer to merge singulars, plurals, and hyphen variants (e.g., merging "queen-less", "queenless", and "queenless colonies").

**Step 2**: Context-Aware Definition & Synonym Harvesting (Simultaneous Step)
 Beekeeping literature is scarce online -> Targeted Retrieval-Augmented Generation (RAG) approach with an LLM, feeding it your local documents.
 - For every unique candidate term left on your list, write a script to find the top 3 sentences in your document collection where that term appears

 ```
 [Context from your Beekeeping Documents]
"...The beekeeper noticed a heavy infestation of Varroa destructor. This parasitic mite attaches to the thorax of the honey bee..."

[Prompt]
Based on the text above, provide:
1. A precise, 1-sentence definition for the term "Varroa destructor".
2. Any exact synonyms or alternative surface forms used for this term in the text.

 ```
(
How to Get Precise Definitions for Concepts - in general
)
**Step 3**: Semantic Embedding ClusteringFor any remaining terms that couldn't be grouped by rules or LLM prompts, convert the generated definitions and terms into vectors using a sentence transformer model (like all-MiniLM-L6-v2).


---
## Building the Ontology

 moving from the Concepts to the Relationships and Attributes.

**Step 1**: Document Re-Indexing (Semantic Search Setup)
 - digestible chunks (paragraphs or chunks of 3–5 sentences).
 - scan for your concepts and their known synonyms.
 -Tag each paragraph with the concepts it contains. For example:Paragraph 42 contains: [Queen Excluder, Worker Bee, Brood Nest]

 **Step 2**: The Co-Occurrence Filter (Finding Relations) - paragraphs that contain two or more distinct concepts.
  multi-concept paragraphs to your local Llama model using a highly structured prompt to extract Triplets (Subject → Predicate → Object)

  ```
  [Context]
"The queen excluder is a selective barrier that prevents the queen bee from entering the honey supers, while allowing smaller worker bees to pass through."

[Prompt]
Analyze the text above. Extract semantic relationships between the following concepts: [Queen Excluder, Queen Bee, Honey Super, Worker Bee].
Format your output strictly as a JSON list of triplets: {"subject": "", "relation": "", "object": ""}.

[Expected Local Llama Output]
[
  {"subject": "Queen Excluder", "relation": "prevents_entry_of", "object": "Queen Bee"},
  {"subject": "Queen Excluder", "relation": "allows_passage_of", "object": "Worker Bee"},
  {"subject": "Queen Bee", "relation": "lays_eggs_in", "object": "Brood Nest"}
]

  ```

  **Step 3: Classifying the Relationships**

  relations  fall into two distinct buckets:
  *A. Taxonomic Relations (is-a Hierarchy)* - These define your ontology structure.
  where to place the node in your graph tree. 
  *B. Non-Taxonomic Relations (Domain Properties)* - connect completely different parts of your tree together.
   Object Properties. They link an instance of one class to an instance of another class

**Step 4: Extracting Attributes (Data Properties)**
   paragraphs that only mention one specific concept, you are not looking for relationships to other concepts. Instead, you look for Attributes (in OWL, these are called Data Properties). These are concrete data values like numbers, colors, sizes, or life spans.


___
# Common tasks in ontlogy learning

- **Entity Linking (Semantic Annotation)**
   -   which ontology entity do *text mention* refer to (apple fruit or company - from context)

  - can we enrich the extraction phase?

  -  A-box of ontology : concrete examples

 - **(Taxonomy )Completion** = **Concept placement** : "placing newly discovered concepts into their correct location within an existing hierarchical parent-child structure " - grow/refine hierarchical backbone
   - add in the middle of and edge, or insert as new leaf in taxonomy.
 - **ontology matching/ alignment:** semantic correspondence, equivalent concepts like employe and staff_member at 2 companies - equivalence relationship
 - concept merging:
   - there might be a hierarchical cluctering of concepts - so new concept will be a leaf - it will definietley have a parent but not necessary a child.
   - edge enrichment is used to improve concept placement - strucural path into semantic vectors - unclear

# Complex concepts:

# Article 1: A Language Model based Framework for New Concept Placement in Ontologies
goal:inserting new concepts - **concept placement**
steps: 
1. edge search to find candidates
2. edge formation and enrichment
3. edge selection
RW: [31,8] entity linnking and new entity discovery: fine tuned BERTencoding  - how for the new mention? for the ont. probably using definition
LLMs are used for entity linking[29] and ont. matching[14]

Contribution: placement - potentially multiple edges

- mention with context window - encode
- concept search : each connect encode - they talk about complex concepts iwth logical operators -> ?
- edge search bi-encoder - mentiion and edge separetely
  - mention is represented as ```[CLS] ctxtl [M_s] mention [M_e] ctxtr [SEP]```
  - edge : ```“[CLS] parent tokens [P-TAG] child tokens [C-TAG] [SEP]```, that is, for each con cepts we encode the endpoints of the edges the concept lies on.
- Fine tuning:contrastive loss to push mentions  to close of one of the edges and far from the wrong ones
- Edge search: for a candidate concept A: all edges from parents to A, from A to children, and from A's parents to A's children. 
- Edge set is ranked by average distance of endpoints to mention embedding - *seed edges*.
  - edge enrichment: for these seed edges we add parents and children of the edge.
- Zero shot LLM version: list candidate edges then ask LLM to pick the *best ones* - seems to be relying on LLM *general knowledge*

- 
## Summary 

So the paper is testing multiple versions of the above steps, with various encodings and architectures, to improve **taxonomy**. 
"In all steps, we propose to
leverage neural methods, where we apply embedding-based methods and
contrastive learning with Pre-trained Language Models (PLMs) such as
BERT for edge search, and adapt a BERT fine-tuning-based multi-label
Edge-Cross-encoder, and Large Language Models (LLMs)"
- We may say, that the philosop
# Article 2: Ontology enrichment using a large language model: Applying lexical, semantic, and knowledge network-based similarity for concept placement

Decent literature review, but a bad article.

triplets generated by GPT-4-1201:```<C,R,P>```   
- first lexical match (token-based, sequence-based, and edit
distance-based distances/Levenshtein
distance, Jaccard similarity, and Longest Common Subsequence),

- then semantic match if lexical did not work - encode with PubMedBERT-  using threshold for semantic match.
- if still no match - use Knowledge Network-based similarity. that is about encoding the parent of then existing concept and the new concept.
reasonable thresholds with  Kernel Density Estimate (KDE) for the distances 

## Summary:
GPT relationship extraction, then if i understand well wer only care about candidate's parent similarity to existing concepts.
