import json

# GYÁRI MODELL SPECIFIKÁCIÓK A FLOP SZÁMÍTÁSHOZ
# Az LLM-ek teljes embedding dimenziója (d) megegyezik a fejek száma * fej dimenzió értékkel.
# Llama 3.2 3B:  32 Query Head * 128 Head Dim = 3072 dimenzió (d)
# Mistral 7B:    32 Query Head * 128 Head Dim = 4096 dimenzió (d)
# Llama 3.1 8B:  32 Query Head * 128 Head Dim = 4096 dimenzió (d)
MODEL_ARCHITECTURES = {
    "llama3.2": {"layers": 28, "embedding_dim": 3072, "head_dim": 128},
    "mistral": {"layers": 32, "embedding_dim": 4096, "head_dim": 128},
    "llama3.1": {"layers": 32, "embedding_dim": 4096, "head_dim": 128}
}


def calculate_theoretical_flops(model_name, num_input_tokens, num_output_tokens):
    """
    Kiszámítja az elméleti FLOP-igényt Kaplan et al. és Hoffmann et al. képletei alapján.
    Separálva adja vissza a Prefill és a Decoding fázisokat.
    """
    arch = MODEL_ARCHITECTURES.get(model_name)
    if not arch:
        return None

    L = arch["layers"]
    d = arch["embedding_dim"]
    d_head = arch["head_dim"]
    N = num_input_tokens
    M = num_output_tokens

    # 1. PREFILL FÁZIS (Input feldolgozás) - Négyzetes komplexitás az Attention miatt
    # Lineáris rétegek műveletei (súlyok szorzása) + Self-Attention mátrixszorzások
    prefill_linear_flops = 2 * L * d * N
    prefill_attention_flops = 2 * L * (N ** 2) * d_head
    total_prefill_flops = prefill_linear_flops + prefill_attention_flops

    # 2. DECODING FÁZIS (Output generálás) - Lineáris komplexitás tokenenként
    # Mivel autoregresszív, minden egyes generált tokennél (M darab) újra lefut a hálózat
    # Az Attention itt elhanyagolható FLOP szempontból, mert a KV-Cache-ből olvassuk a múltat
    total_decoding_flops = 2 * L * d * M

    return {
        "prefill_flops": total_prefill_flops,
        "decoding_flops": total_decoding_flops,
        "total_flops": total_prefill_flops + total_decoding_flops
    }


# =====================================================================
# TESZT FUTTATÁS A KÍSÉRLETED ÁTLAGOS TOKEN-SZÁMAIVAL
# =====================================================================
if __name__ == "__main__":
    # Tételezzük fel az alábbi átlagos token-számokat a korábbi méréseid alapján:
    # Futtatás 1 (Kontextussal + Példákkal): hosszú prompt (~650 token), rövid JSON válasz (~45 token)
    # Futtatás 2 (Kontextus nélkül): nagyon rövid prompt (~80 token), rövid JSON válasz (~45 token)

    RUN_1_INPUT, RUN_1_OUTPUT = 650, 45
    RUN_2_INPUT, RUN_2_OUTPUT = 80, 45

    print("=" * 70)
    print("      ELMÉLETI SZÁMÍTÁSI IGÉNY KIÉRTÉKELÉSE (FLOP / GFLOP)")
    print("=" * 70)
    print(f"Kísérleti beállítások:")
    print(f" - Run 1 (Kontextussal): Input = {RUN_1_INPUT} token, Output = {RUN_1_OUTPUT} token")
    print(f" - Run 2 (No-Context) : Input = {RUN_2_INPUT} token, Output = {RUN_2_OUTPUT} token\n")

    for model_name in MODEL_ARCHITECTURES.keys():
        print(f"#" * 60)
        print(f" MODELL: {model_name.upper()}")
        print(f"#" * 60)

        # RUN 1 Számítás
        res_1 = calculate_theoretical_flops(model_name, RUN_1_INPUT, RUN_1_OUTPUT)
        # GigaFLOP-ra váltás (/ 1,000,000,000) a szebb olvashatóságért
        p1_gflops = res_1["prefill_flops"] / 1e9
        d1_gflops = res_1["decoding_flops"] / 1e9
        t1_gflops = res_1["total_flops"] / 1e9

        print(f"-> RUN 1 (Kontextussal):")
        print(f"   - Prefill (Input) igény  : {p1_gflops:.2f} GFLOPs")
        print(f"   - Decoding (Output) igény: {d1_gflops:.2f} GFLOPs")
        print(f"   - Összesített elméleti szám: {t1_gflops:.2f} GFLOPs")

        # RUN 2 Számítás
        res_2 = calculate_theoretical_flops(model_name, RUN_2_INPUT, RUN_2_OUTPUT)
        p2_gflops = res_2["prefill_flops"] / 1e9
        d2_gflops = res_2["decoding_flops"] / 1e9
        t2_gflops = res_2["total_flops"] / 1e9

        print(f"\n-> RUN 2 (Kontextus nélkül):")
        print(f"   - Prefill (Input) igény  : {p2_gflops:.2f} GFLOPs")
        print(f"   - Decoding (Output) igény: {d2_gflops:.2f} GFLOPs")
        print(f"   - Összesített elméleti szám: {t2_gflops:.2f} GFLOPs")

        # Megtakarítás számítása
        flops_saved_pct = ((res_1["total_flops"] - res_2["total_flops"]) / res_1["total_flops"]) * 100
        print(f"\n=> ELMÉLETI MŰVELET-MEGTAKARÍTÁS:")
        print(
            f"   - A kontextus elhagyása {flops_saved_pct:.1f}%-kal csökkentette a szükséges matematikai műveletek számát!")
        print("-" * 60)


