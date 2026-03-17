# 1. Vanilla (only once)
echo "[Phase 1] Vanilla"
python measure_speedup_micro_benchmark.py 0.0 --device cpu --method vanilla

# 2. WASI
echo "[Phase 2] WASI"
for i in 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "  -> WASI eps=$i"
    python measure_speedup_micro_benchmark.py "$i" --device cpu --method WASI
done

# # 3. ASI
# echo "[Phase 3] ASI"
# for i in 0.733501434 2.071178436 5.206390381 13.06269073 36.20851517 123.1485367; do
#     echo "  -> ASI budget=$i"
#     python measure_speedup_micro_benchmark.py "1.0" --device cpu --method ASI --budget_ASI "$i"
# done