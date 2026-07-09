import json
import os

PERFORMANCE_LOG_FILE = "results/minilm_pipeline_performance_log_llm_performance.jsonl"


# GYÁRI MODELL SPECIFIKÁCIOK A KV-CACHE KÉPLETHEZ
MODEL_SPECS = {
    "llama3.2": {"layers": 28, "kv_heads": 8, "head_dim": 128, "bytes": 2},
    "mistral": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes": 2},
    "llama3.1": {"layers": 32, "kv_heads": 8, "head_dim": 128, "bytes": 2}
}


def calculate_kv_cache_mb(model_name, total_tokens):
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        return 0
    bytes_size = 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"] * total_tokens * spec["bytes"]
    return bytes_size / (1024 * 1024)


def run_efficiency_evaluation():
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        print(f"Hiba: A '{PERFORMANCE_LOG_FILE}' fájl nem található!")
        return

    metrics = {}
    total_records = 0

    with open(PERFORMANCE_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                if "consumption_metrics" not in record: continue

                total_records += 1
                cons_data = record["consumption_metrics"]

                for model_name, modes in cons_data.items():
                    if model_name not in metrics:
                        metrics[model_name] = {
                            "with_context": {"time": [], "prompt_tokens": [], "gen_tokens": [], "alpha": [],
                                             "beta": []},
                            "without_context": {"time": [], "prompt_tokens": [], "gen_tokens": [], "alpha": [],
                                                "beta": []}
                        }

                    # === JAVÍTÁS: KIZÁRÓLAG A CONS_DATA ATTRIBÚTUMAIBÓL DOLGOZUNK ===
                    w_ctx = modes.get("with_context", {})
                    no_ctx = modes.get("without_context", {})

                    # 1. KONTEXTUSOS ÁG SZÁMÍTÁSAI
                    if w_ctx.get("prompt_tokens", 0) > 0:
                        t_time = w_ctx.get("total_time_sec", 0)
                        p_tok = w_ctx.get("prompt_tokens", 0)
                        g_tok = w_ctx.get("generated_tokens", 0)
                        tps = w_ctx.get("tokens_per_second", 0)

                        metrics[model_name]["with_context"]["time"].append(t_time)
                        metrics[model_name]["with_context"]["prompt_tokens"].append(p_tok)
                        metrics[model_name]["with_context"]["gen_tokens"].append(g_tok)

                        # Elméleti béta kiszámítása a TPS-ből (ms/token)
                        if tps > 0:
                            beta_val = (1.0 / tps) * 1000
                            metrics[model_name]["with_context"]["beta"].append(beta_val)

                            # Elméleti alfa kiszámítása (Megmaradt idő / prompt tokenek) * 1000
                            gen_time = g_tok / tps
                            prompt_time = max(0.0, t_time - gen_time)
                            metrics[model_name]["with_context"]["alpha"].append((prompt_time / p_tok) * 1000)

                    # 2. KONTEXTUS NÉLKÜLI ÁG SZÁMÍTÁSAI
                    if no_ctx.get("prompt_tokens", 0) > 0:
                        t_time_no = no_ctx.get("total_time_sec", 0)
                        p_tok_no = no_ctx.get("prompt_tokens", 0)
                        g_tok_no = no_ctx.get("generated_tokens", 0)
                        tps_no = no_ctx.get("tokens_per_second", 0)

                        metrics[model_name]["without_context"]["time"].append(t_time_no)
                        metrics[model_name]["without_context"]["prompt_tokens"].append(p_tok_no)
                        metrics[model_name]["without_context"]["gen_tokens"].append(g_tok_no)

                        if tps_no > 0:
                            beta_val_no = (1.0 / tps_no) * 1000
                            metrics[model_name]["without_context"]["beta"].append(beta_val_no)

                            gen_time_no = g_tok_no / tps_no
                            prompt_time_no = max(0.0, t_time_no - gen_time_no)
                            metrics[model_name]["without_context"]["alpha"].append((prompt_time_no / p_tok_no) * 1000)

            except Exception as e:
                # print(f"Sor hiba: {e}") # Debug esetén visszakapcsolható
                pass

    if total_records == 0:
        print("Nem találtam feldolgozható fogyasztási adatot a logfájlban.")
        return

    print("\n" + "=" * 75)
    print("     LLM EFFEKTIVITÁSI JELENTÉS: JAVÍTOTT KLASSZIKUS EGYÜTTHATÓK")
    print("=" * 75)
    print(f"Feldolgozott esetek száma: {total_records}")

    for model_name, modes in metrics.items():
        print("\n" + "#" * 65)
        print(f" MODELL: {model_name.upper()}")
        print("#" * 65)

        # 1. KONTEXTUSSAL ÁTLAGOK
        w_ctx = modes["with_context"]
        if w_ctx["prompt_tokens"]:
            avg_w_time = sum(w_ctx["time"]) / len(w_ctx["time"])
            avg_w_prompt = sum(w_ctx["prompt_tokens"]) / len(w_ctx["prompt_tokens"])
            avg_w_gen = sum(w_ctx["gen_tokens"]) / len(w_ctx["gen_tokens"])
            avg_w_seq = avg_w_prompt + avg_w_gen
            avg_w_kv = calculate_kv_cache_mb(model_name, avg_w_seq)

            avg_alpha = sum(w_ctx["alpha"]) / len(w_ctx["alpha"]) if w_ctx["alpha"] else 0
            avg_beta = sum(w_ctx["beta"]) / len(w_ctx["beta"]) if w_ctx["beta"] else 0

            print(f"-> FUTTATÁS 1: KONTEXTUSSAL + PÉLDÁKKAL (Few-Shot)")
            print(
                f"   - Átlagos szekvencia hossz       : {avg_w_seq:.1f} token (Prompt: {avg_w_prompt:.1f}, Gen: {avg_w_gen:.1f})")
            print(f"   - Átlagos Dinamikus KV-Cache     : {avg_w_kv:.4f} MB / kérés")
            print(f"   - Átlagos válaszidő (Latency)    : {avg_w_time:.3f} másodperc")
            print(f"   - Becsült Alfa (Prefill sebesség): {avg_alpha:.3f} ms / input token")
            print(f"   - Becsült Béta (Decoding seb.)   : {avg_beta:.3f} ms / output token")

        # 2. KONTEXTUS NÉLKÜL ÁTLAGOK
        no_ctx = modes["without_context"]
        if no_ctx["prompt_tokens"]:
            avg_n_time = sum(no_ctx["time"]) / len(no_ctx["time"])
            avg_n_prompt = sum(no_ctx["prompt_tokens"]) / len(no_ctx["prompt_tokens"])
            avg_n_gen = sum(no_ctx["gen_tokens"]) / len(no_ctx["gen_tokens"])
            avg_n_seq = avg_n_prompt + avg_n_gen
            avg_n_kv = calculate_kv_cache_mb(model_name, avg_n_seq)

            avg_alpha_no = sum(no_ctx["alpha"]) / len(no_ctx["alpha"]) if no_ctx["alpha"] else 0
            avg_beta_no = sum(no_ctx["beta"]) / len(no_ctx["beta"]) if no_ctx["beta"] else 0

            print(f"\n-> FUTTATÁS 2: KONTEXTUS NÉLKÜL (Tiszta gép-tudás)")
            print(
                f"   - Átlagos szekvencia hossz       : {avg_n_seq:.1f} token (Prompt: {avg_n_prompt:.1f}, Gen: {avg_n_gen:.1f})")
            print(f"   - Átlagos Dinamikus KV-Cache     : {avg_n_kv:.4f} MB / kérés")
            print(f"   - Átlagos válaszidő (Latency)    : {avg_n_time:.3f} másodperc")
            print(f"   - Becsült Alfa (Prefill sebesség): {avg_alpha_no:.3f} ms / input token")
            print(f"   - Becsült Béta (Decoding seb.)   : {avg_beta_no:.3f} ms / output token")

        # 3. HATÉKONYSÁGI ÖSSZEHASONLÍTÁS
        if w_ctx["prompt_tokens"] and no_ctx["prompt_tokens"]:
            kv_saved_pct = ((avg_w_kv - avg_n_kv) / avg_w_kv) * 100
            time_saved_pct = ((sum(w_ctx["time"]) - sum(no_ctx["time"])) / sum(w_ctx["time"])) * 100

            print(f"\n=> ERŐFORRÁS MEGTAKARÍTÁSI MÉRLEG:")
            print(f"   - A kontextus elhagyásával a RAM KV-Cache terhelés {kv_saved_pct:.1f}%-kal CSÖKKENT.")
            print(f"   - A számítási idő a laptopodon {time_saved_pct:.1f}%-kal LETT RÖVIDEBB.")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_efficiency_evaluation()
