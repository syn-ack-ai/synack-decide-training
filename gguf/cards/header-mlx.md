---
license: apache-2.0
base_model: SkyPanther/synack-decide-26b-a4b
library_name: mlx
tags: [mlx, gemma4, moe, decision, system-one, jev, classification, routing]
---

# SynACK Decide 26B-A4B (MLX 8-bit)

8-bit MLX conversion of [SkyPanther/synack-decide-26b-a4b](https://huggingface.co/SkyPanther/synack-decide-26b-a4b) (8.5 bits per weight). **Checked against the bf16 weights:** on 500 held-out PRs it makes the same routing decision on 97.6% of them at the 0.27 threshold (98.4% at 0.30), with AUC 0.774 vs 0.777 and rank correlation 0.985; on the ordinal probes it scores 82.5% / 77.8% vs 82.3% / 78.1% for bf16. Converted with Google's original `config.json`, because transformers 5.18 writes fields that mlx-lm 0.32 misreads.

