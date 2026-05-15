---
title: "Counting Every Parameter in GPT-2"
author: "Chaitanya"
pubDatetime: 2026-04-26T00:00:00Z
slug: counting-gpt2-parameters
featured: true
draft: false
tags:
  - llm
  - transformers
  - gpt-2
  - architecture
description: "A hand-count of every parameter in GPT-2 small — embeddings, attention, MLP, LayerNorms, the tied LM head — deriving 124,439,808 from the hyperparameters alone."
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaitanya-19/llm-architecture-notebooks/blob/main/01_counting_gpt2_parameters.ipynb)Every number in this post is verified in the companion notebook.

In this series of blogs I'll take you from 0 to 1 on how modern LLMs actually work. I'll start with GPT-2 — what happens under the hood, what calculations run when you prompt it, and where its 124 million parameters actually live. From there we'll keep complicating things, post by post, until we've worked through the architecture of open-source models like Llama 3, Mixtral, and DeepSeek-V3.

I'm assuming you've read ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) and have a working mental model of transformers. If not, [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), or [the paper itself](https://arxiv.org/abs/1706.03762) are good places to catch up before continuing.

Onto this first post.

GPT-2 small has 124 million parameters. That number gets quoted constantly. The breakdown of where those 124M actually sit — how many in the embeddings, how many per transformer block, how many attention heads — is much less common knowledge.

This post counts every parameter in GPT-2 by hand. Embeddings, the QKV projection, the output projection, both MLP matrices, every LayerNorm, the tied LM head. Exact shapes, exact counts, no approximations. By the end you'll derive 124,439,808 from the hyperparameters alone — and you'll have a clean baseline for the rest of the series. When we get to GQA in the next post and I tell you Llama 3 cuts its KV cache by 75%, you'll know which matrices that 75% comes from, because you'll have counted them here.

<img src="/diagrams/01-counting-gpt2/01-architecture-stack.png" alt="GPT-2 full architecture: from input string through 12 transformer blocks to next-token probabilities. Embeddings are 31.6% of the model; the 12-block stack is 68.3%; the LM head is tied to the token embedding and adds zero parameters." style="max-width: 500px; display: block; margin: 0 auto;" />

_Note: the token IDs shown in the diagram (`[464, 2068, 7586, 21831]`) are illustrative placeholders, not the actual GPT-2 BPE encoding of "the quick brown fox." Run the companion notebook to see the real IDs._

## A few terms before we count

A handful of vocabulary that'll come up throughout. If you've trained or fine-tuned LLMs you can skim — these are the minimum prerequisites for the math.

**Token.** The atomic unit a language model operates on — not a word, not a character, but a _subword_ chunk produced by a tokenizer. The string `"tokenization is fun"` might split into `["token", "ization", " is", " fun"]` — four tokens, each represented internally as an integer ID into the model's vocabulary.

**BPE (Byte Pair Encoding).** The subword tokenization algorithm GPT-2 uses. BPE starts from individual bytes and iteratively merges the most frequent adjacent pairs into new tokens. The result is a vocabulary that handles common words as single tokens (`"the"`, `"and"`), rare words as a few subwords (`"tokenization"` → `"token"` + `"ization"`), and arbitrary byte sequences (emoji, code, non-Latin scripts) by falling back to byte-level tokens. The vocabulary size is fixed at training time.

**Context length.** The maximum number of tokens the model can attend to at once — its working memory window. GPT-2 caps at 1,024 tokens; Llama 3 starts at 8,192 and extends to 128,000; modern frontier models stretch into the millions. Context length is an architectural choice (baked into the position embedding table for GPT-2) and directly drives the memory cost of inference.

**Parameter.** A learned weight or bias in the model — a single scalar value that gradient descent has tuned during training. When we say "GPT-2 has 124M parameters," we mean 124M individual floating-point numbers stored in the model's weight matrices and bias vectors. Distinct from **hyperparameters** (`d_model`, `n_layer`, etc.), which are set by the architect before training and never updated.

**Embedding.** A lookup-table mapping from discrete token IDs to dense vectors. The token embedding `wte` in GPT-2 is a `(50257, 768)` matrix — row `i` is the 768-dim vector representation of token ID `i`. The model never sees the integer `i` after the first lookup; everything downstream operates on vectors.

**LM head (Language Modeling head).** The final layer of the model — a linear projection that maps each output vector back to the vocabulary, producing one logit per possible next token. For GPT-2 that's a `(768, 50257)` projection. GPT-2 specifically _ties_ the LM head to the token embedding (reuses `wte` transposed), so it adds zero new parameters. We'll prove that in the final-layer section.

**Logits.** Raw, un-normalized scores the model produces over the vocabulary. The LM head outputs a 50,257-dim logit vector at every position; softmax converts those logits into a probability distribution over the next token.

## The GPT-2 small spec sheet

GPT-2 ships in four sizes — small (124M), medium (355M), large (774M), and XL (1.5B). Throughout this series I'll use _GPT-2_ to mean _GPT-2 small_ unless I say otherwise. It's the canonical baseline, it's the one everyone's mental "124M" refers to, and the math generalizes cleanly to the larger variants.

Here are the hyperparameters we'll be working with:

| Symbol       | Name                                         | Value  |
| ------------ | -------------------------------------------- | ------ |
| `vocab_size` | BPE vocabulary size                          | 50,257 |
| `n_ctx`      | Max context length                           | 1,024  |
| `d_model`    | Hidden dimension                             | 768    |
| `n_layer`    | Number of transformer blocks                 | 12     |
| `n_head`     | Number of attention heads                    | 12     |
| `d_head`     | Dimension per head (= `d_model / n_head`)    | 64     |
| `d_ff`       | MLP intermediate dimension (= 4 × `d_model`) | 3,072  |

Quick orientation on what each of these means and why they're set the way they are.

**`vocab_size = 50,257`.** The size of GPT-2's BPE vocabulary — 256 byte-level base tokens + 50,000 learned BPE merges + 1 special `<|endoftext|>` token. Every input token is one of these 50,257 IDs.

**`d_model = 768`.** The width of the residual stream — the dimension of every token's representation as it flows through the network. After embedding, each token is a 768-dim vector. After every transformer block, still 768-dim. The LM head receives 768-dim vectors. Every internal activation shape in GPT-2 ends in `768`. (Modern open-source models are much wider — Llama 3 405B uses `d_model = 16,384`, more than 20× GPT-2 small.)

**`d_head = d_model / n_head = 64`.** Each attention head operates on a 64-dim slice of the 768-dim representation. The reason `d_head` is calculated this way rather than chosen freely is dimensional: the per-head outputs get concatenated back together before the output projection, and we want that concatenation to land at `d_model` so the residual stream stays the same width throughout the network. Twelve heads × 64 dims = 768. (Modern models with GQA break this clean symmetry — Q, K, and V no longer all have the same number of heads. That's the whole next post.)

**`d_ff = 4 × d_model = 3,072`.** The MLP between attention layers expands each token's representation 4× before projecting back down. The 4× ratio is a convention from "Attention Is All You Need" — MLPs need to be wider than the residual stream to learn useful non-linear transformations, and 4× empirically hits a good balance between capacity and parameter cost. Modern models with SwiGLU change this ratio (we'll see ~8/3 × later in the series), but for vanilla GPT-2 it's 4×.

A few architectural choices to flag before we count anything:

GPT-2 is **pre-norm** — LayerNorm sits _before_ attention and _before_ the MLP, not after. Position information comes from a **learned position embedding table**, not the sin/cos formulation from the original Transformer paper. The MLP uses **GELU**. Attention is plain multi-head — twelve query heads, twelve key heads, twelve value heads, all the same width. And the LM head is **tied** to the token embedding.

One implementation note worth knowing now, because it'll show up in the count. We _conceptually_ think of attention as "12 heads, each with its own W*Q, W_K, W_V" — but none of that is physically separate in the actual code. GPT-2 stores **one big weight matrix** of shape `(768, 2304)` called `c_attn` that handles Q, K, \_and* V for all 12 heads in a single matmul:

```
X @ c_attn → (T, 768) @ (768, 2304) = (T, 2304)
```

That `(T, 2304)` output is then split three ways into Q, K, V tensors of shape `(T, 768)` each, and each of those is reshaped to `(T, 12, 64)` to expose the head dimension. The "multi-head split" is a reshape, not a separate set of weights. The 12 heads are a _view_ on the same underlying matrix. This is the same total parameter count as if you'd defined 36 separate matrices (3 projections × 12 heads × 64 dims each), just packed into one tensor for matmul efficiency — we'll count it both ways in the attention section to make the equivalence concrete.

Every one of these architectural choices changes in modern LLMs, and every one gets its own post later in this series. For now, they're just facts about GPT-2.

Now we count.

## Embeddings

The moment you send text to GPT-2, the first thing that happens is BPE tokenization — your input string gets split into subword tokens. Each token is then mapped to its integer ID in the 50,257-entry vocabulary, and each ID is used to look up a 768-dim vector from a table that was learned during pretraining. The string `"the quick brown fox"` becomes a sequence of 4 token IDs, which becomes a sequence of 4 vectors, each of shape `(768,)`.

That table is the **token embedding**. There's a second table — the **position embedding** — that adds positional information to each vector. Two embedding tables total, both fully learned during pretraining, both pure lookup at inference time.

**Token embedding (`wte`).** A matrix of shape `(50257, 768)`. Row `i` is the 768-dim vector for token ID `i`. Looking up a sequence of token IDs is just a row-lookup — no matmul, no learned function, just `wte[token_ids]`.

Parameters: 50,257 × 768 = **38,597,376**.

That's already the largest single tensor in GPT-2 — bigger than any individual matrix inside the transformer blocks. We'll see this matrix again at the very end of the network: GPT-2 reuses it as the LM head, which is why the LM head adds zero new parameters. (More on weight tying when we get there.)

**Position embedding (`wpe`).** A matrix of shape `(1024, 768)`. Row `i` is the 768-dim vector for _being at position `i`_. The first dimension is 1024 because that's GPT-2's context length — the table needs exactly one row per possible position, and the model can't see past position 1023.

Parameters: 1,024 × 768 = **786,432**.

This one's worth pausing on. GPT-2's position embeddings are **learned**, not sinusoidal. The original Transformer paper used fixed sin/cos functions of position; GPT-2 (following GPT-1) replaced them with a learned lookup table. Each row of `wpe` is gradient-updated like any other parameter. The practical implication is real: GPT-2 _cannot_ run on sequences longer than 1,024 tokens without retraining or extending the table — there's literally no `wpe[1024]` to look up. RoPE, which we'll cover in post 3, is one answer to this rigidity.

**Combining the two.** Pure element-wise addition. No concat, no projection. For a sequence of length `T`:

```
x = wte[token_ids] + wpe[0:T]
   (T, 768)        (T, 768)    →    (T, 768)
```

The actual forward-pass code reads exactly like the math:

```python
token_ids = ...                       # (B, T) integer IDs
positions = torch.arange(T)           # (T,)

x = wte(token_ids) + wpe(positions)   # (B, T, 768)
```

After this point every tensor flowing through GPT-2 has its last dimension fixed at 768.

**Embeddings subtotal:**

| Component        | Shape          | Parameters     |
| ---------------- | -------------- | -------------- |
| `wte` (token)    | `(50257, 768)` | 38,597,376     |
| `wpe` (position) | `(1024, 768)`  | 786,432        |
| **Total**        |                | **39,383,808** |

**Running total: 39,383,808 parameters — 31.6% of the model, and we haven't done a single matmul yet.**

## Inside one transformer block

A GPT-2 transformer block does two things to its input: it lets each token attend to other tokens (via attention), and it transforms each token's representation independently (via the **MLP**, short for _Multi-Layer Perceptron_). Both operations are wrapped in residual connections, and each gets a LayerNorm in front.

Here's the structure:

```python
def block(x):
    x = x + attention(layer_norm_1(x))   # LN → attention → residual
    x = x + mlp(layer_norm_2(x))         # LN → MLP → residual
    return x
```

![One transformer block: two LayerNorms, two sub-layers (attention and MLP), two residual connections. The residual stream stays 768-dim throughout.](/diagrams/01-counting-gpt2/02-transformer-block.png)

Two sub-layers, two LayerNorms, two residual connections. Now we understand each piece — and count what's in it.

### Residual connections and the residual stream

The `x + ...` in those two lines is doing more than it looks. Each sub-layer doesn't _replace_ the token's representation; it computes an _update_ and adds it to what was already there. The token's vector flows from block to block as a running sum: original embedding + block 1's update + block 2's update + ... + block 12's update.

This running 768-dim vector is called the **residual stream**. Every component in the network reads from it and writes back to it. Attention reads, computes its update, adds it. The MLP reads, computes its update, adds it. Twelve blocks, twenty-four sub-layers, all writing into the same shared 768-dim space.

The crucial property: the residual stream's width never changes. It starts at 768 (after embedding) and stays at 768 all the way through. Every weight matrix in GPT-2 either reads from the 768-dim stream, processes internally, and writes back to 768 — or expands to a wider intermediate space and projects back. This is why `d_model = 768` shows up as a dimension in almost every weight matrix in the model.

(Without residual connections, training a 12-block transformer would be brutal — gradients would vanish before they reach early layers. The residual path keeps gradient signal flowing during backprop. But for our counting purposes, the architectural fact matters more than the training intuition.)

### LayerNorm

LayerNorm normalizes activations across the feature dimension. For each token's 768-dim vector, it computes the mean and variance, normalizes to zero mean and unit variance, then applies a learned affine transformation:

```
y = γ * (x - μ) / √(σ² + ε) + β
```

The mean μ and variance σ² are computed from the input at runtime — no parameters. The learned parameters are γ (scale) and β (shift), each of shape `(768,)` — one γ and one β per feature dimension.

Per LayerNorm: 768 + 768 = **1,536 parameters**.

Each block has two — `ln_1` before attention, `ln_2` before MLP. Per block: **3,072 parameters** in LayerNorms.

This is the smallest line item per block, but every block has it. (Modern models replace LayerNorm with RMSNorm, which drops β and the mean-centering step. That's post 4.)

### Attention: what Q, K, V actually mean

This is the part everyone hand-waves. Let's not.

Self-attention is a mechanism that lets each token look at every previous token (the causal mask blocks future tokens) and pull in relevant information. The question is _how_ the model decides which tokens are relevant to which.

Every token gets three vectors — a **query**, a **key**, and a **value** — computed by passing the token's embedding through three different learned matrices. These three vectors play distinct roles:

- **Query (Q):** "What am I looking for?" Encodes the kind of context this token wants from other tokens.
- **Key (K):** "What do I represent?" Encodes the kind of context this token _offers_ to others.
- **Value (V):** "What information do I actually carry?" The content that gets mixed into other tokens' representations if they decide to attend to me.

To decide how much attention token _i_ pays to token _j_, we compute the dot product of _i_'s query and _j_'s key. High dot product → high attention. We then use these attention scores as weights to take a weighted sum of all the value vectors. The result becomes token _i_'s updated representation.

There's actually a _fourth_ matrix in attention beyond Q, K, V — the **output projection (W_O)** — that we'll cover after walking through the attention math. It doesn't have a question/key/value role. Its job is to mix the per-head outputs back into a single 768-dim vector after attention has done its work. Mentioning it now so you're not surprised when a fourth matrix shows up in the count.

Three things to internalize before we move on:

1. Q, K, V are not different pieces of data. They're three different _projections_ of the same input vector.
2. Each token has its own Q, K, V, computed in parallel for the whole sequence.
3. The model learns the projection matrices during training. It learns _what kind of question each token should ask_ (Q), _what each token should advertise about itself_ (K), and _what content each token should share_ (V).

Now let's see how this is actually computed.

### Computing Q, K, V

Our input: the 4-token sequence "the quick brown fox" has been tokenized, embedded, and combined with position embeddings. The input to the first transformer block is a matrix `X` of shape `(4, 768)` — 4 rows (one per token), each row a 768-dim vector.

```
X.shape = (4, 768)

       ←————— 768 dims —————→
     ┌                        ┐
     │  the's vector          │  ← row 0
X =  │  quick's vector        │  ← row 1
     │  brown's vector        │  ← row 2
     │  fox's vector          │  ← row 3
     └                        ┘
```

To compute Q, K, V we use three weight matrices, each of shape `(768, 768)`. These are learned during training:

- `W_Q`: maps each 768-dim token vector to its query vector. Shape `(768, 768)`.
- `W_K`: maps each 768-dim token vector to its key vector. Shape `(768, 768)`.
- `W_V`: maps each 768-dim token vector to its value vector. Shape `(768, 768)`.

The math is just matrix multiplication. The `@` symbol in Python means matmul (matrix multiplication). When we write `X @ W_Q`, we mean: multiply matrix `X` of shape `(4, 768)` by matrix `W_Q` of shape `(768, 768)`, producing a matrix of shape `(4, 768)`. The inner dimensions (768 and 768) match and disappear; the outer dimensions (4 and 768) survive.

```
Q = X @ W_Q    →    (4, 768) @ (768, 768) = (4, 768)
K = X @ W_K    →    (4, 768) @ (768, 768) = (4, 768)
V = X @ W_V    →    (4, 768) @ (768, 768) = (4, 768)
```

After this step we have three matrices, each of shape `(4, 768)`. Row 0 of Q is the's query, row 1 is quick's query, row 2 is brown's, row 3 is fox's. Same row layout for K and V.

Each weight matrix has 768 × 768 = 589,824 weights, plus a 768-dim bias — total 590,592 parameters per matrix. Three matrices: 3 × 590,592 = **1,771,776 parameters** for the QKV projection.

(Implementation note for when you read the source code: GPT-2 doesn't actually keep these as three separate matrices. It packs them into a single weight tensor of shape `(768, 2304)` called `c_attn` and does one matmul instead of three. The parameter count is identical — 768 × 2304 + 2304 = 1,771,776 — and the math works out exactly the same. Three matrices is the cleaner mental model; the packed version is purely a GPU optimization.)

### Multi-head: 12 parallel attentions

There's one more twist before we do the actual attention math. We don't run _one_ attention computation on these `(4, 768)` matrices. We run _twelve_ parallel attention computations, each on a 64-dim slice.

The 768-dim Q, K, V vectors are reshaped into 12 heads of 64 dims each:

```
Q: (4, 768) → reshape → (4, 12, 64)
K: (4, 768) → reshape → (4, 12, 64)
V: (4, 768) → reshape → (4, 12, 64)
```

Each head sees the same 4 tokens, but operates on its own 64-dim slice of the Q, K, V vectors. Twelve attention computations run in parallel; their outputs get concatenated back together at the end.

Why bother with multiple heads? Different heads can learn to attend to different things. One head might learn syntactic dependencies (subject ↔ verb), another might track coreferences (pronoun ↔ antecedent), another might attend to nearby tokens vs. distant ones. Twelve specialized "attention sub-mechanisms" instead of one generalist.

Importantly: **no new parameters are introduced by the multi-head split.** The reshape is just rearranging the same numbers. The 12 heads are a _view_ on the same underlying weight matrices — that's why `d_head = d_model / n_head = 64` falls out so cleanly. The 768-dim Q/K/V was already implicitly partitioned; reshaping makes the partition explicit.

One thing worth being explicit about: unlike the `c_attn` packing we mentioned earlier (which is a code optimization — the conceptual three-matrix view is mathematically equivalent), the multi-head reshape _genuinely happens_ in the actual implementation. PyTorch really does call `.reshape()` and `.transpose()` on the Q, K, V tensors to expose the head dimension before running attention. But it's a _view_ operation: no memory is copied, no data is moved, just the same underlying numbers indexed differently. So while the reshape is real, it's free — zero compute, zero parameters.

### Walking through the attention math (one head)

Let's walk through the math for _one_ head — head 0 — using our 4-token example. `Q_h`, `K_h`, `V_h` are each `(4, 64)` tensors (the head-0 slice).

**Step 1: Score every query against every key.**

```
scores = Q_h @ K_h^T    →    (4, 64) @ (64, 4) = (4, 4)
```

`K_h^T` is K transposed — shape flips from `(4, 64)` to `(64, 4)`. The result is a `(4, 4)` matrix where entry `(i, j)` is the dot product of query _i_ and key _j_ — i.e., how strongly token _i_ wants to attend to token _j_ (according to head 0):

```
                  ←——— keys ———→
                 the  quick brown fox
              ┌                        ┐
   q_the      │  s00   s01   s02   s03 │
   q_quick    │  s10   s11   s12   s13 │
scores  =     │  s20   s21   s22   s23 │
   q_brown    │  s30   s31   s32   s33 │
   q_fox      └                        ┘
```

Entry `(3, 2)` is fox's query dotted with brown's key — i.e., how strongly fox at position 3 wants to attend to brown at position 2 (in head 0's view).

**Step 2: Scale.**

```
scores = scores / √d_head = scores / √64 = scores / 8
```

A stabilization trick. Dot products in 64-dim space can grow large; dividing by √d_head keeps them in a range where softmax is well-behaved (otherwise softmax saturates and gradients vanish during training).

**Step 3: Apply the causal mask.**

GPT-2 is autoregressive — token _i_ can only attend to tokens at positions ≤ _i_. The causal mask sets all scores at positions _j > i_ to `-∞`, which becomes 0 after softmax:

```
                       ┌                        ┐
                       │  s00   -∞    -∞    -∞  │
                       │  s10   s11   -∞    -∞  │
masked scores  =       │  s20   s21   s22   -∞  │
                       │  s30   s31   s32   s33 │
                       └                        ┘
```

The diagonal stays — token _i_ can attend to itself (i = j). Only the _strictly_ upper triangle (j > i) gets masked. This is one of the most common bugs to ship: masking the diagonal too is wrong.

**Step 4: Softmax (row-wise).**

Softmax converts each row of scores into a probability distribution that sums to 1. These are the attention weights:

```
weights = softmax(scores)    →    (4, 4)
```

Row 3 of `weights` tells us how fox distributes its attention across positions 0, 1, 2, 3. If brown (position 2) is the most relevant, `weights[3, 2]` will be the largest value in that row.

This softmax happens _inside every attention sub-layer in every block_, once per head. With 12 heads and 12 blocks, that's 12 × 12 = **144 attention softmaxes per forward pass** for our 4-token sequence. Don't confuse this with the _final_ softmax that runs at the very end of the network — the one that converts the LM head's vocabulary logits into next-token probabilities. That's a separate, single softmax over 50,257 dims, run only once per generated token. Two different softmaxes, two different roles.

**Step 5: Weighted sum of values.**

```
out_h = weights @ V_h    →    (4, 4) @ (4, 64) = (4, 64)
```

This is the punchline. Each row of `out_h` is a weighted sum of _all_ the value vectors, weighted by the attention scores. Row 3 (fox's output) is a mix of value-the, value-quick, value-brown, value-fox — heavily weighted toward whichever ones fox's query matched best.

That's one head. The exact same five steps run in parallel for heads 1 through 11, each producing its own `(4, 64)` output.

![The five-step attention recipe applied to one head on the 4-token sequence: score, scale, mask, softmax, weighted sum.](/diagrams/01-counting-gpt2/03-attention-math.png)

### Concatenating heads and the output projection

After all 12 heads have produced their outputs, we have 12 tensors of shape `(4, 64)`. We concatenate them along the feature dimension to get back to `(4, 768)`:

```
concat(out_0, out_1, ..., out_11)    →    (4, 768)
```

Then this concatenated output is passed through the **attention output projection** — the fourth matrix we flagged earlier. It's called `c_proj` in the GPT-2 source code; we'll call it `W_O` here:

```
attn_out = concat @ W_O    →    (4, 768) @ (768, 768) = (4, 768)
```

`W_O` has shape `(768, 768)`. Its job is to mix information _across_ heads. Each output position can blend signal from all 12 heads' outputs into a final 768-dim vector. This is the "we ran 12 parallel computations, now combine them" step.

`W_O` has 768 × 768 + 768 = **590,592 parameters**.

The result `attn_out` has shape `(4, 768)` — same as the input `X`. This gets _added_ back to the residual stream: `x = x + attn_out`. The residual stream stays 768-dim throughout.

**Attention parameters per block:**

| Component                        | Shape            | Parameters    |
| -------------------------------- | ---------------- | ------------- |
| QKV projection (W_Q + W_K + W_V) | 3 × `(768, 768)` | 1,771,776     |
| Output projection (W_O)          | `(768, 768)`     | 590,592       |
| **Total attention**              |                  | **2,362,368** |

### The MLP

After attention, the second sub-layer is the MLP — also called the feed-forward network. While attention mixes information _across_ tokens, the MLP transforms each token's representation _independently_. There's no token-to-token interaction here; just a per-token nonlinear transformation.

GPT-2's MLP has exactly **two** weight matrices, with a GELU activation in between. Worth marking now because this is one of the cleanest contrasts with modern models — SwiGLU (post 4) splits this into three matrices to support a gating mechanism. For vanilla GPT-2, the count is two.

The first matrix `c_fc` ("fully connected") expands each token from 768 dims up to 3072 dims:

```
hidden = X @ W_fc    →    (4, 768) @ (768, 3072) = (4, 3072)
```

Parameters: 768 × 3072 + 3072 = 2,359,296 + 3,072 = **2,362,368**.

GELU is applied element-wise — no parameters, just a smooth nonlinearity (a smoother variant of ReLU).

The second matrix `c_proj` projects back down from 3072 to 768:

```
mlp_out = gelu(hidden) @ W_proj    →    (4, 3072) @ (3072, 768) = (4, 768)
```

Parameters: 3072 × 768 + 768 = 2,359,296 + 768 = **2,360,064**.

Total MLP per block: 2,362,368 + 2,360,064 = **4,722,432 parameters**.

The MLP is the largest single component in the block — bigger than all of attention combined. This isn't coincidence. Mechanistic interpretability studies suggest the MLP is where most of the model's stored knowledge lives — attention does the routing, the MLP does the lookup. When we get to MoE in post 7, the gating is applied to the MLP for exactly this reason: that's where the parameters are.

### Block total

Tallying everything in one block:

| Component                      | Parameters    |
| ------------------------------ | ------------- |
| `ln_1` + `ln_2` (LayerNorms)   | 3,072         |
| QKV projection                 | 1,771,776     |
| Attention output projection    | 590,592       |
| MLP up projection (`c_fc`)     | 2,362,368     |
| MLP down projection (`c_proj`) | 2,360,064     |
| **One block**                  | **7,087,872** |

GPT-2 has 12 such blocks: 12 × 7,087,872 = **85,054,464 parameters** in the transformer stack.

Running tally:

|                       | Parameters      | % of model |
| --------------------- | --------------- | ---------- |
| Embeddings            | 39,383,808      | 31.6%      |
| 12 transformer blocks | 85,054,464      | 68.3%      |
| **So far**            | **124,438,272** | **99.99%** |

We're 1,536 parameters short of the full 124,439,808 — and the next section accounts for exactly that.

## The final LayerNorm and the tied LM head

We're at 124,438,272 parameters. The transformer stack is done. What's left is the very end of the network — the path from the last block's output to the next-token probability distribution.

After the 12th transformer block, the residual stream has shape `(T, 768)` — the same shape it had after embedding, just with 12 blocks' worth of accumulated updates added to it. Two things happen next: a final LayerNorm, then the LM head.

### The final LayerNorm

GPT-2 applies one more LayerNorm after the last transformer block, before the LM head. It's identical in structure to the LayerNorms inside the blocks — γ and β, each of shape `(768,)`, computed over the feature dimension.

Parameters: 768 + 768 = **1,536**.

This brings our running total to **124,439,808** — the canonical 124M GPT-2 small parameter count, exactly.

The reason this final LayerNorm exists is a consequence of pre-norm architecture. In pre-norm, every sub-layer reads from the residual stream _after_ a LayerNorm — so the residual stream itself never gets normalized along the way. By the time we reach the end of the 12th block, the stream has accumulated 24 sub-layer updates (12 attention + 12 MLP) without any cleanup. The final LayerNorm is the cleanup pass before we project to vocabulary.

### The LM head and weight tying

The last operation in the network is the **LM head** — the linear projection that converts each 768-dim output vector into a 50,257-dim logit vector, one logit per vocabulary token. The token with the highest logit (after softmax) is the model's prediction for the next token.

Conceptually, this is a weight matrix of shape `(768, 50257)`:

```
logits = final_hidden @ W_lm_head
       = (T, 768) @ (768, 50257)
       = (T, 50257)
```

That matrix would have 768 × 50257 = **38,597,376 parameters** if GPT-2 trained it from scratch.

But GPT-2 doesn't. It uses a trick called **weight tying**: instead of a separate `W_lm_head` matrix, the LM head is just the token embedding `wte` transposed. The same `(50257, 768)` matrix that maps `token_id → vector` at the input gets reused as `vector → logits` at the output:

```python
# Conceptually:
logits = final_hidden @ wte.T         # (T, 768) @ (768, 50257) = (T, 50257)
```

No new weights are introduced. The LM head adds **0 parameters** to our count.

This is a deliberate architectural choice that goes back to ["Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)](https://arxiv.org/abs/1608.05859). The intuition: the embedding matrix has already learned a mapping between tokens and the 768-dim semantic space. The reverse direction is the same problem, so the same matrix can do both jobs. Tying the weights forces the input and output representations to live in a shared geometry, which improves both training efficiency and output quality.

The savings are substantial. Without tying, the LM head would add 38,597,376 parameters to the model — that's 31% of GPT-2 small's parameter budget. The tied version reuses what's already there. For a model this small, with vocab roughly 65× larger than `d_model`, weight tying is essentially free quality and 31% fewer parameters.

(Modern open-source models don't all tie. Llama 3 _unties_ — its LM head is a separate `(d_model, vocab_size)` matrix. Why? When `d_model` gets large — Llama 3 405B has `d_model = 16,384` while vocab is 128,256 — the embedding matrix is no longer the dominant cost, and the input vs. output representations benefit from being learned independently. Weight tying is a sensible default at GPT-2's scale and a questionable choice at Llama 3's. We'll quantify this when we count Llama 3 later.)

## The 124M tally

Every parameter counted. Here's the full breakdown:

| Component                       | Parameters      | % of total |
| ------------------------------- | --------------- | ---------- |
| Token embedding (`wte`)         | 38,597,376      | 31.0%      |
| Position embedding (`wpe`)      | 786,432         | 0.6%       |
| 12 × transformer blocks         | 85,054,464      | 68.3%      |
| └─ LayerNorms (24 total)        | 36,864          | 0.03%      |
| └─ QKV projections              | 21,261,312      | 17.1%      |
| └─ Attention output projections | 7,087,104       | 5.7%       |
| └─ MLPs                         | 56,669,184      | 45.5%      |
| Final LayerNorm                 | 1,536           | 0.001%     |
| LM head (tied to `wte`)         | 0               | 0.0%       |
| **Total**                       | **124,439,808** | **100%**   |

![Where GPT-2's 124M parameters live: MLPs (45.5%), token embedding (31%), attention QKV (17%), other components.](/diagrams/01-counting-gpt2/04-parameter-breakdown.png)

A few things worth pulling out of this table.

**The embeddings are 31.6% of the model.** That's enormous. For every two parameters inside the 12-block transformer stack, there's almost one parameter just sitting in the token and position embedding tables. This ratio is a GPT-2 small artifact — at this scale, vocab × d_model (50,257 × 768 ≈ 38.6M) is competitive with the per-block parameter count. But it doesn't generalize.

For Llama 3 70B: vocab is 128,256 (about 2.5× GPT-2's), but `d_model` is 8,192 (10.7× GPT-2's). The token embedding alone is 128,256 × 8,192 ≈ 1.05B parameters — sounds like a lot until you realize the model has 70 billion total parameters. **Embeddings are 1.5% of Llama 3 70B.** They go from a third of the model to a rounding error.

This is why frontier models can untie their LM head without flinching. At GPT-2 small's scale, untying would _double_ the embedding cost — disastrous. At Llama 3's scale, untying adds 1.5% to the parameter count for the freedom of learning input and output representations independently. Easy trade.

**The MLP is 45.5% of the entire model.** Just the MLPs. Bigger than embeddings, attention, and LayerNorms combined. Per block, the MLP is 4.7M parameters versus attention's 2.4M — roughly 2:1, MLP-heavy. This pattern persists in modern dense models (the SwiGLU MLP in Llama 3 is also the largest component per block) and it's _the_ reason Mixture-of-Experts matters. When DeepSeek-V3 or Mixtral replaces the MLP with 8 experts and routes each token to 2 of them, they're targeting the largest line item in the parameter budget. Sparse the MLP, sparse the model. We'll work through that math in post 7.

**Attention's parameters are dominated by the QKV projection.** The output projection is one matrix — 590K parameters per block. The QKV projection is three matrices — 1.77M parameters per block. So 75% of attention's parameter cost is in producing Q, K, V. This is exactly where GQA strikes: by reducing the number of K and V heads (Llama 3 uses 8 K/V heads versus 32 Q heads), you shrink the K and V projections by 4× while leaving Q untouched. Less obvious from parameter count, the bigger win is at inference time — the KV cache. We'll quantify that next post.

**LayerNorms are negligible.** 36,864 + 1,536 = 38,400 parameters total — 0.03% of the model. They show up in every block, but they're tiny. Worth noting because LayerNorm parameters are sometimes excluded from "core model" parameter counts in papers; for GPT-2 it doesn't materially affect the headline number either way.

## Verify it yourself

Every count in this post can be checked against the actual GPT-2 weights in about 6 lines:

```python
from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2")

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")

# Per-tensor breakdown
for name, p in model.named_parameters():
    print(f"{name:40s}  {tuple(p.shape)!s:20s}  {p.numel():>12,}")
```

Run this and you'll see every shape and every count we derived by hand:

- `wte.weight` → `(50257, 768)` → 38,597,376
- `wpe.weight` → `(1024, 768)` → 786,432
- For each of 12 blocks: `ln_1`, `attn.c_attn`, `attn.c_proj`, `ln_2`, `mlp.c_fc`, `mlp.c_proj`
- `ln_f.weight`, `ln_f.bias` → the final LayerNorm

The total comes back as 124,439,808.

(Note: HuggingFace's `GPT2Model` doesn't include the LM head as a separate parameter precisely because of weight tying — `lm_head` shares its weight with `wte`. If you load `GPT2LMHeadModel` instead, you'll see the LM head listed but its weight is still tied to `wte`, so the parameter count doesn't change.)

The companion Colab notebook ([open in Colab](https://colab.research.google.com/github/chaitanya-19/llm-architecture-notebooks/blob/main/01_counting_gpt2_parameters.ipynb)) walks through this verification step by step, prints the full per-tensor breakdown, and reproduces every running total in the post. It also includes a hand-built attention computation for block 0 head 0, compared against the model's own attention output to within 1e-5. Open it in another tab while you re-read — every number in this post should match a number in that notebook.

## What's next

You now have a complete model of GPT-2 — every matrix, every LayerNorm, every parameter, every shape. This is the baseline. From here, every post in the rest of the series is going to swap out one piece of GPT-2 for what modern models do instead, and we'll quantify exactly what changes.

The next post is **Grouped-Query Attention**: why Llama 3 has 32 query heads but only 8 key and value heads. The headline result — Llama 3 cuts its KV cache by 75% versus standard multi-head attention — is going to be much more concrete now that you've seen exactly which matrices the K and V heads come from, and which projections shrink when you reduce them. We'll re-count attention with GQA, work out the KV-cache memory math at inference time, and see why the attention design that was perfectly fine at GPT-2 scale becomes a serving problem at frontier scale.

See you in post 2.
