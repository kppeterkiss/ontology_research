import json
import os

PERFORMANCE_LOG_FILE = "pipeline_performance_log.jsonl"


def run_evaluation():
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        print(f"Hiba: A '{PERFORMANCE_LOG_FILE}' naplófájl nem található!")
        return

    total_records = 0
    oov_records = 0

    # Adatszerkezetek a statisztikákhoz modellek szerint
    # Felépítése: { model_name: { "with_context": {"NEW_CONCEPT": 0, "EXISTING": 0}, "without_context": {...}, "agreement": 0 } }
    stats = {}

    # OOV specifikus statisztikák
    oov_stats = {}

    print(f"Naplófájl beolvasása: {PERFORMANCE_LOG_FILE}...")

    with open(PERFORMANCE_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                total_records += 1

                # OOV diagnosztika kinyerése
                is_oov = record.get("oov_diagnostic", {}).get("is_oov_anomaly", False)
                if is_oov:
                    oov_records += 1

                bench_results = record.get("benchmark_results", {})

                for model_name, results in bench_results.items():
                    # Inicializálás, ha még nem láttuk a modellt
                    if model_name not in stats:
                        stats[model_name] = {
                            "with_ctx_new": 0, "with_ctx_exist": 0,
                            "no_ctx_new": 0, "no_ctx_exist": 0,
                            "agreement": 0, "errors": 0
                        }
                        oov_stats[model_name] = {"oov_agreement": 0, "oov_total": 0}

                    ctx_dec = results.get("with_context", {}).get("decision")
                    no_ctx_dec = results.get("without_context", {}).get("decision")

                    # Ha valamelyik ágon hiba történt, kihagyjuk a statisztikából
                    if ctx_dec == "ERROR" or no_ctx_dec == "ERROR":
                        stats[model_name]["errors"] += 1
                        continue

                    # Kontextusos számlálók
                    if ctx_dec == "NEW_CONCEPT":
                        stats[model_name]["with_ctx_new"] += 1
                    elif ctx_dec == "EXISTING":
                        stats[model_name]["with_ctx_exist"] += 1

                    # Kontextus nélküli számlálók
                    if no_ctx_dec == "NEW_CONCEPT":
                        stats[model_name]["no_ctx_new"] += 1
                    elif no_ctx_dec == "EXISTING":
                        stats[model_name]["no_ctx_exist"] += 1

                    # Egyezőség vizsgálata (Döntési konzisztencia)
                    if ctx_dec == f"{no_ctx_dec}":
                        stats[model_name]["agreement"] += 1
                        if is_oov:
                            oov_stats[model_name]["oov_agreement"] += 1

                    if is_oov:
                        oov_stats[model_name]["oov_total"] += 1

            except Exception as e:
                print(f"Hiba egy sor parszolásakor: {e}")

    if total_records == 0:
        print("A logfájl üres vagy nem tartalmazott érvényes adatot.")
        return

    # =====================================================================
    # ERÖDMÉNYEK KIÍRATÁSA (TUDOMÁNYOS RENDSZEREZÉSBEN)
    # =====================================================================
    print("\n" + "=" * 60)
    print("      PRECÍZIÓS MÉHÉSZETI ONTOLÓGIA ABLÁCIÓS BENCHMARK")
    print("=" * 60)
    print(f"Összes feldolgozott egyedi kifejezés száma: {total_records}")
    print(f"Ebből SciBERT OOV (Out-of-Vocabulary) anomália: {oov_records} ({oov_records / total_records * 100:.1f}%)")

    for model_name, s in stats.items():
        valid_runs = total_records - s["errors"]
        if valid_runs <= 0: continue

        agreement_pct = (s["agreement"] / valid_runs) * 100

        print("\n" + "-" * 50)
        print(f" MODELL: {model_name.upper()}")
        print("-" * 50)
        print(f"-> Kontextussal (Tűéles ablak):")
        print(f"   - NEW_CONCEPT (NIL): {s['with_ctx_new']} eset")
        print(f"   - EXISTING (Azonosság): {s['with_ctx_exist']} eset")

        print(f"-> Kontextus NÉLKÜL (Csak domain infó):")
        print(f"   - NEW_CONCEPT (NIL): {s['no_ctx_new']} eset")
        print(f"   - EXISTING (Azonosság): {s['no_ctx_exist']} eset")

        print(f"\n=> DÖNTÉSI EGYEZŐSÉG (Konzisztencia): {agreement_pct:.2f}%")
        print(f"   (A kontextus elhagyása az esetek {100 - agreement_pct:.2f}%-ában változtatta meg a döntést!)")

        # OOV specifikus hatás elemzése
        o_stat = oov_stats.get(model_name, {})
        if o_stat.get("oov_total", 0) > 0:
            oov_agree_pct = (o_stat["oov_agreement"] / o_stat["oov_total"]) * 100
            print(f"\n=> EBBŐL AZ OOV (ISMERETLEN) SZAVAKNÁL AZ EGYEZŐSÉG: {oov_agree_pct:.2f}%")
            print(
                f"   (Az OOV szavaknál a kontextus hiánya {100 - oov_agree_pct if 'oov_agree_pct' in locals() else 100 - oov_agree_pct:.2f}%-os döntési szórást okozott.)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_evaluation()
