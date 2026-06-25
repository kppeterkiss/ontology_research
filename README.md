# Spacy model download

```bash
 uv run spacy  download en_core_web_sm
```
# Ollama local install
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

llama 3.1 qvantált: ```Q4_K_M``` 16 GB RAM to ~4 GB RAM

further optimization:
```python


        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_ctx": 2048 
            },
            keep_alive=0,  # <--- EZ AZ EXTRA: Azonnal felszabadítja a RAM-ot a kérés után!
            format="json"
        )
```
```bash
ollama pull llama3.1
```

```bash
ollama serve`
``

# llama.cpp

```bash
# Apple Silicon Mac-re (Metal gyorsítással):
CMAKE_ARGS="-GGUIDE -DLLAMA_METAL=on" pip install llama-cpp-python

# Windows/Linux Nvidia GPU-val (CUDA gyorsítással):
CMAKE_ARGS="-GGUIDE -DLLAMA_CUDA=on" pip install llama-cpp-python
```


TODO: seed: cross references, synonyms,plurals: larvae


For dynamic content:
playwright explicit install needed: ```playwright install```