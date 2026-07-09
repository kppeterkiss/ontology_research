import json
import os

PERFORMANCE_LOG_FILE = "results/minilm_pipeline_performance_log_llm_performance.jsonl"

# GYÁRI MODELL SPECIFIKÁCIÓK (Layers, Embedding Dim, Head Dim)
MODEL_ARCHITECTURES = {
    "llama3.2": {"layers": 28, "embedding_dim": 3072, "head_dim": 128},
    "mistral": {"layers": 32, "embedding_dim": 4096, "head_dim": 128},
    "llama3.1": {"layers": 32, "embedding_dim": 4096, "head_dim": 128}
}


def calculate_exact_flops(model_name, N, M):
    """Kiszámítja a pontos FLOP igényt a megadott N (input) és M (output) tokenek alapján."""
    arch = MODEL_ARCHITECTURES.get(model_name)
    if not arch or N <= 0:
        return 0, 0

    L = arch["layers"]
    d = arch["embedding_dim"]
    d_head = arch["head_dim"]

    # Prefill FLOP: 2 * L * d * N (Linear) + 2 * L * N^2 * d_head (Attention)
    prefill_flops = (2 * L * d * N) + (2 * L * (N ** 2) * d_head)
    # Decoding FLOP: 2 * L * d * M
    decoding_flops = 2 * L * d * M

    return prefill_flops, decoding_flops


def run_actual_flops_evaluation():
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        print(f"Hiba: A '{PERFORMANCE_LOG_FILE}' fájl nem található!")
        return

    # Adatgyűjtő szótár a statisztikákhoz
    flop_stats = {}
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
                    if model_name not in flop_stats:
                        flop_stats[model_name] = {
                            "with_context": {"prefill": [], "decoding": [], "total": []},
                            "without_context": {"prefill": [], "decoding": [], "total": []}
                        }

                    # 1. KONTEXTUSOS ÁG VALÓS ADATAI
                    w_ctx = modes.get("with_context", {})
                    n_in = w_ctx.get("prompt_tokens", 0)
                    m_out = w_ctx.get("generated_tokens", 0)

                    if n_in > 0:
                        p_flop, d_flop = calculate_exact_flops(model_name, n_in, m_out)
                        flop_stats[model_name]["with_context"]["prefill"].append(p_flop)
                        flop_stats[model_name]["with_context"]["decoding"].append(d_flop)
                        flop_stats[model_name]["with_context"]["total"].append(p_flop + d_flop)

                    # 2. KONTEXTUS NÉLKÜLI ÁG VALÓS ADATAI
                    no_ctx = modes.get("without_context", {})
                    n_in_no = no_ctx.get("prompt_tokens", 0)
                    m_out_no = no_ctx.get("generated_tokens", 0)

                    if n_in_no > 0:
                        p_flop_no, d_flop_no = calculate_exact_flops(model_name, n_in_no, m_out_no)
                        flop_stats[model_name]["without_context"]["prefill"].append(p_flop_no)
                        flop_stats[model_name]["without_context"]["decoding"].append(d_flop_no)
                        flop_stats[model_name]["without_context"]["total"].append(p_flop_no + d_flop_no)

            except Exception as e:
                pass

    if total_records == 0:
        print("Nem találtam feldolgozható token-fogyasztási adatot.")
        return

    print("\n" + "=" * 75)
    print("     VALÓS FUTÁSI ADATOKON ALAPULÓ ELMÉLETI MŰVELETIGÉNY (GFLOPs)")
    print("=" * 75)
    print(f"Kiértékelt feladatok száma: {total_records}")

    for model_name, modes in flop_stats.items():
        print("\n" + "#" * 65)
        print(f" MODELL: {model_name.upper()}")
        print("#" * 65)

        # Átlagok számítása a kontextusos ágra (Osztunk 1 milliárddal -> GFLOP)
        ctx = modes["with_context"]
        if ctx["total"]:
            avg_p = (sum(ctx["prefill"]) / len(ctx["prefill"])) / 1e9
            avg_d = (sum(ctx["decoding"]) / len(ctx["decoding"])) / 1e9
            avg_t = (sum(ctx["total"]) / len(ctx["total"])) / 1e9
            print(f"-> FUTTATÁS 1: KONTEXTUSSAL (Few-Shot)")
            print(f"   - Átlagos Prefill (Input)  : {avg_p:.3f} GFLOPs")
            print(f"   - Átlagos Decoding (Output): {avg_d:.3f} GFLOPs")
            print(f"   - Átlagos Összesített Igény: {avg_t:.3f} GFLOPs")

        # Átlagok számítása a kontextus nélküli ágra
        no_ctx = modes["without_context"]
        if no_ctx["total"]:
            avg_p_no = (sum(no_ctx["prefill"]) / len(no_ctx["prefill"])) / 1e9
            avg_d_no = (sum(no_ctx["decoding"]) / len(no_ctx["decoding"])) / 1e9
            avg_t_no = (sum(no_ctx["total"]) / len(no_ctx["total"])) / 1e9
            print(f"\n-> FUTTATÁS 2: KONTEXTUS NÉLKÜL")
            print(f"   - Átlagos Prefill (Input)  : {avg_p_no:.3f} GFLOPs")
            print(f"   - Átlagos Decoding (Output): {avg_d_no:.3f} GFLOPs")
            print(f"   - Átlagos Összesített Igény: {avg_t_no:.3f} GFLOPs")

        # Tiszta elméleti megtakarítás a valós tokenek alapján
        if ctx["total"] and no_ctx["total"]:
            saved_pct = ((sum(ctx["total"]) - sum(no_ctx["total"])) / sum(ctx["total"])) * 100
            print(f"\n=> ELMÉLETI MATEMATIKAI MEGTAKARÍTÁS:")
            print(f"   - A kontextus elhagyása átlagosan {saved_pct:.1f}%-kal csökkentette a FLOP-igényt.")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_actual_flops_evaluation()
