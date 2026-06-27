# PRODIGY_ML_01 — GPT-2 Text Generation

> **ProDigy Infotech · Machine Learning Internship · Task 01**

Fine-tune GPT-2 on a custom text corpus and generate coherent, contextually relevant text from any prompt.

---

## 📂 Repository Structure

```
PRODIGY_ML_01/
├── gpt2_text_generation.py   # Main script (fine-tune + generate)
├── GPT2_Text_Generation.ipynb# Interactive Jupyter notebook
├── train_data.txt            # Sample training corpus (AI/ML domain)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/PRODIGY_ML_01.git
cd PRODIGY_ML_01

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Quick Demo (no training needed)
```bash
python gpt2_text_generation.py demo
```

### Generate Text from a Prompt
```bash
python gpt2_text_generation.py generate \
    --prompt "Artificial intelligence is transforming" \
    --model gpt2 \
    --max_tokens 200 \
    --num_sequences 3 \
    --temperature 0.9
```

### Fine-Tune on Your Own Dataset
```bash
# Prepare a plain .txt file with your training text
python gpt2_text_generation.py finetune \
    --train_file train_data.txt \
    --output_dir ./gpt2-finetuned \
    --model gpt2 \
    --epochs 3
```

### Generate with Fine-Tuned Model
```bash
python gpt2_text_generation.py generate \
    --prompt "Deep learning enables" \
    --model ./gpt2-finetuned \
    --max_tokens 150
```

---

## 📓 Notebook

Open `GPT2_Text_Generation.ipynb` in Jupyter or Google Colab for a fully interactive walkthrough including:

- Base GPT-2 text generation
- Decoding strategy comparison (greedy, beam search, top-K, top-P, combined)
- Fine-tuning on a custom corpus
- Interactive generation cell

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🔑 Key Concepts

| Concept | Description |
|---|---|
| **GPT-2** | Autoregressive transformer language model by OpenAI |
| **Fine-tuning** | Adapting a pre-trained model to a specific domain/style |
| **Greedy decoding** | Always pick the highest-probability next token |
| **Beam search** | Keep top-n candidate sequences at each step |
| **Top-K sampling** | Sample from the K most likely tokens |
| **Top-P (nucleus)** | Sample from the smallest set of tokens whose cumulative prob ≥ P |
| **Temperature** | Scale logits — higher = more creative, lower = more deterministic |
| **Repetition penalty** | Penalise already-generated tokens to reduce repetition |

---

## 🧠 Model Variants

| Model | Parameters | Notes |
|---|---|---|
| `gpt2` | 117M | Fast, good for experimentation |
| `gpt2-medium` | 345M | Better coherence |
| `gpt2-large` | 774M | High quality, needs more RAM |
| `gpt2-xl` | 1.5B | Best quality, needs GPU |

---

## 📚 References

- [How to Generate Text — HuggingFace Blog](https://huggingface.co/blog/how-to-generate)
- [GPT-2 Fine-Tuning Colab](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [OpenAI GPT-2 Paper](https://openai.com/research/language-unsupervised)

---

## 📋 LinkedIn Post Template

> Just completed **Task 01** of my Machine Learning Internship at **ProDigy Infotech**! 🚀
>
> 🤖 **Project:** GPT-2 Text Generation
>
> In this task, I:
> ✅ Loaded and explored OpenAI's GPT-2 transformer model
> ✅ Compared 5 decoding strategies (greedy, beam search, top-K, top-P, nucleus sampling)
> ✅ Fine-tuned GPT-2 on a custom AI/ML domain corpus
> ✅ Generated coherent, contextually relevant text from custom prompts
>
> 💡 Key learnings: transformer architecture, autoregressive language modeling, transfer learning, and the impact of decoding hyperparameters on text quality.
>
> 🔗 GitHub: github.com/<your-username>/PRODIGY_ML_01
>
> #MachineLearning #NLP #GPT2 #DeepLearning #ProdigyInfotech #Internship #AI #Transformers

---

*Made with ❤️ during the ProDigy Infotech ML Internship*
