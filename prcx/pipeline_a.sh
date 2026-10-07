#!/bin/sh
# Option A pipeline on the Mac: zero-shot eval -> small PR fine-tune -> re-test.
cd ~/systemone/prcx
while pgrep -f predict_mlx.py >/dev/null; do sleep 30; done
echo "=== ZERO-SHOT (full test) ===" > option_a_results.txt
python3 evaluate.py --dataset data/dataset --model-preds moe_zeroshot=preds/moe_zeroshot_test.jsonl >> option_a_results.txt 2>&1
cd ~/systemone/v2
../mlxenv/bin/python train_mlx.py --model ../models/gemma4-26b-a4b-qat-mlx4 --init-adapter adapters/run1-moe/best/adapters.safetensors \
  --train ../prcx/data/dataset/ft_small_train.jsonl --valid ../prcx/data/dataset/ft_valid.jsonl --out adapters/moe-pr-small \
  --batch 16 --lr 3e-5 --val-n 400 --eval-every 40 > ../prcx/ft_small.log 2>&1
cd ~/systemone/prcx
../mlxenv/bin/python predict_mlx.py --model ../models/gemma4-26b-a4b-qat-mlx4 --adapter ../v2/adapters/moe-pr-small/best \
  --records data/dataset/test_records.jsonl --out preds/moe_pr_small_test.jsonl > preds_ft_small.log 2>&1
echo "=== AFTER SMALL PR FINE-TUNE ===" >> option_a_results.txt
python3 evaluate.py --dataset data/dataset --model-preds moe_zeroshot=preds/moe_zeroshot_test.jsonl \
  --model-preds moe_pr_finetuned=preds/moe_pr_small_test.jsonl >> option_a_results.txt 2>&1
echo "PIPELINE A DONE" >> option_a_results.txt
