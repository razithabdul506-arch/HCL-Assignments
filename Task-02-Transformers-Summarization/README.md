# Abstractive Headline Generation Using a Transformer Built From Scratch

A complete, end-to-end **Encoder–Decoder Transformer** implemented **100% from scratch** in TensorFlow/Keras for **abstractive news headline generation** on the Gigaword corpus. No pretrained language models, pretrained weights, pretrained embeddings, or pretrained tokenizers are used at any stage.

---

## Project Highlights
* **100% From-Scratch Pipeline:** Custom SentencePiece BPE tokenizer, Sinusoidal Positional Encoding, Multi-Head Self/Cross-Attention, Pre-LayerNorm blocks, Weight-Tied Embeddings, and a custom `@tf.function` + `tf.GradientTape` training loop.
* **Verified Attention Masking:** Explicitly tested to confirm **zero PAD-token cross-attention leakage (`0.00000000`)** before training.
* **Strong Generalization on Single GPU:** Trained on 120,000 Gigaword pairs on a single Google Colab **NVIDIA Tesla T4 GPU**, achieving **49.51% masked validation token accuracy** and **0.3256 ROUGE-1** (Beam Search).
* **Full Interpretability & Demo:** Includes training/validation curves, head-averaged Cross-Attention heatmaps, a 3-panel evaluation dashboard, and a live custom-text inference demo.

---

## Dataset Overview
* **Source:** [`walzen/gigaword`](https://huggingface.co/datasets/walzen/gigaword) on Hugging Face (Rush et al. Gigaword Headline-Generation Corpus).
* **Task:** Compress the lead sentence of a news article (`article`) into a concise, abstractive headline (`summary`).

| Split | Samples Used | Mean Word Length | Max Token Length | Token Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | 120,000 | Input: 31.3 \| Summary: 8.2 | `64` \| `20` | 97.88% \| 98.28% |
| **Validation** | 3,000 | — | `64` \| `20` | — |
| **Test (Full Official)** | 1,951 | — | `64` \| `20` | — |

---

## Architecture & Training Configuration

| Component | Configuration |
| :--- | :--- |
| **Tokenizer** | Custom SentencePiece BPE (`vocab_size=8,000`, `PAD=0`, `UNK=1`, `BOS=2`, `EOS=3`) |
| **Model Architecture** | 3-Layer Pre-LN Encoder–Decoder Transformer |
| **Dimensions** | `d_model=256`, `num_heads=8`, `dff=512`, `dropout=0.2` |
| **Weight Tying** | Shared embedding matrix across Encoder, Decoder, and Output Projection |
| **Trainable Parameters** | **6,010,688** |
| **Loss Function** | Masked Label-Smoothed Cross-Entropy (`label_smoothing=0.1`) |
| **Optimizer & Schedule** | `AdamW` (`peak_lr=6e-4`, Linear Warmup + Cosine Decay, `clipnorm=1.0`) |
| **Batch Size / Epochs** | `128` / `15` epochs (with Early Stopping) |

---

## Evaluation Results

### 1. Training & Validation Convergence
* **Best Validation Loss:** `3.9466` (Epoch 13)
* **Best Validation Masked Token Accuracy:** `49.51%` (Epoch 15)

### 2. Test-Set ROUGE Scores

| Evaluation Split & Decoding Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :--- | :--- |
| **Full Official Test Set (1,951 samples) — Greedy** | **0.3108** | **0.1342** | **0.2896** |
| **200-Sample Benchmark — Greedy** | 0.3128 | 0.1288 | 0.2960 |
| **200-Sample Benchmark — Beam Search (`k=4`, `α=0.6`)** | **0.3256** | **0.1474** | **0.3073** |

---

## Qualitative Examples (Unseen Inputs)

### Test-Set Samples
| Input Article (Truncated) | Reference Headline | Beam Search (`k=4`) Prediction |
| :--- | :--- | :--- |
| *the german government and red cross have decided to give ###,### dollars in humanitarian aid to victims of the earthquake...* | `germany gives ###,### dollars in aid for iran quake victims` | `germany to give humanitarian aid to quake victims` |
| *teenager michaela krajicek claimed her first grasscourt title with a #-# , #-# victory over third seed dinara safina...* | `krajicek defeats safina in dutch final` | `krajicek wins first grasscourt title` |
| *at least ## people were killed as typhoon sepat hit the chinese mainland after more than ###,### were evacuated...* | `storms lash chinese coast killing ## ; ###,### evacuated` | `typhoon sepat hits chinese mainland` |

### Live Custom Input Demo
* **Input:** *"India's space agency successfully launched a new weather satellite into orbit on Monday from its southern spaceport."*
  * **Output (`Beam k=4`):** `india launches new weather satellite`
* **Input:** *"Oil prices rose sharply on Tuesday after major oil-producing nations agreed to cut daily production."*
  * **Output (`Beam k=4`):** `oil prices rise sharply`

---

## Repository Structure
```text
├── Gigaword_Transformer_Scratch.ipynb      # Complete end-to-end Colab notebook (Markdown + Code)
├── README.md                               # Project documentation
└── artifacts/                              # Exported training & evaluation outputs (optional)
    ├── best_gigaword_transformer.weights.h5
    ├── gigaword_bpe.model
    ├── gigaword_bpe.vocab
    ├── training_history.json
    ├── final_evaluation_results.json
    └── evaluation_comparison_dashboard.png