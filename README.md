# 🗓️ 30-Day Project Plan: Build a Reasoning LLM

## Week 1: Neural Network & Language Basics
**Focus:** Fundamentals of neural nets & word representation

- **Day 1:** Implement perceptron (AND/OR/XOR)  
  [Perceptron](./perceptron/)
- **Day 2:** Build a small feedforward NN on MNIST digits (only 1 hidden layer)  
  [Feedforward](./feedforward)
- **Day 3:** Implement backpropagation manually (tiny dataset)  
  [Backpropagation](./backpropagation)
- **Day 4:** Implement word embeddings (Word2Vec skip-gram with toy corpus)
- **Day 5:** Visualize embeddings with PCA/t-SNE
- **Day 6:** Build an MLP for text classification (movie reviews tiny dataset)
- **Day 7:** Reflection & mini-report: compare embeddings vs one-hot

## Week 2: Sequence Models & Attention
**Focus:** Sequence learning, RNN → Transformer motivation

- **Day 8:** Implement a vanilla RNN (char-level text generator)
- **Day 9:** Add LSTM, train on small text dataset (predict next char)
- **Day 10:** Compare RNN vs LSTM outputs on long sequences
- **Day 11:** Implement self-attention (Q, K, V matrices) from scratch
- **Day 12:** Visualize attention weights on a toy sentence
- **Day 13:** Implement multi-head self-attention
- **Day 14:** Mini reflection: why attention > RNNs?

## Week 3: Transformers & Tiny GPT
**Focus:** Build a decoder-only transformer

- **Day 15:** Implement transformer block (attention + feedforward + layer norm)
- **Day 16:** Stack multiple transformer blocks → mini encoder
- **Day 17:** Add causal masking → decoder-only transformer
- **Day 18:** Train on a toy text corpus (fairy tales, Wikipedia snippets)
- **Day 19:** Generate text with sampling (top-k, nucleus sampling)
- **Day 20:** Implement positional embeddings & compare with/without them
- **Day 21:** Reflection: How GPT is built from these blocks

## Week 4: Reasoning & Alignment
**Focus:** Reasoning tasks + preference alignment

- **Day 22:** Create toy reasoning dataset (math: 2+3=?, logical: “A→B, B→C”)
- **Day 23:** Train GPT-mini on this reasoning dataset
- **Day 24:** Add chain-of-thought fine-tuning (teach model to “think step by step”)
- **Day 25:** Evaluate on GSM8k-style toy problems
- **Day 26:** Implement PPO (basic version) for reward tuning (reward = correct reasoning)
- **Day 27:** Try DPO with reasoning preferences (good vs bad chains)
- **Day 28:** Add tool-use module (model can call calculator)
- **Day 29:** Run evaluation: compare plain GPT-mini vs CoT vs PPO/DPO tuned
- **Day 30:** Final report + presentation notebook (plots, samples, what worked, what didn’t)

---

## ✅ Deliverables by End of Month
- Mini GPT-like model (decoder-only, trained on toy reasoning data)
- Chain-of-thought reasoning fine-tuned model
- PPO/DPO aligned version
- A final notebook showing reasoning examples