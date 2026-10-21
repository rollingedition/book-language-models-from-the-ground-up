---
chapter_id: the-transformer-block
---

# The transformer block

By the end of this chapter you can assemble a transformer block from its parts, joined by residual connections: attention, normalization, a gated feed-forward layer and position information. You can then stack blocks into a complete language model. You'll check each part on the store's trained model, check the whole model against PyTorch's own implementations, measure what each block and each detail contributes, and count where the parameters go.

[Chapter 6](#attention) built attention: each position looks back at earlier ones and takes a weighted average of what it finds. Attention alone isn't a language model. It moves information between positions but transforms little of it, and it has almost no sense of word order. Layers of it stacked directly also degrade: without the other parts of a block, the output of pure attention tends towards the same vector at every position, faster with every layer ([Dong et al., 2021](https://arxiv.org/abs/2103.03404)). The **transformer block** adds the parts that fix those problems around attention ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

This chapter builds the block in the form the LLaMA models popularized: normalization of each part's input with RMSNorm, a gated feed-forward layer and rotary position embeddings in every layer ([Touvron et al., 2023](https://arxiv.org/abs/2302.13971), section 2.2). SmolLM2, whose tokenizer [chapter 2](#read-an-open-tokenizer) read, is built the same way. Its 360-million-parameter model has 960 dimensions, 32 layers and 15 heads, and differs from the store's model in two details covered below: its heads share keys and values, and its rotary positions turn more slowly ([configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/f8027fd0eaeea54caa13c31d31b9fdc459c38b49/config.json)).

The store's model, whose heads [chapter 6](#read-trained-heads) read, is {{out:transformer-parameters:layers}} of these blocks between an embedding table and an output layer; this book calls each block a layer of the model. At every position, each block does two things in order:

1. normalize the vector, run attention, and add the result back to the vector;
2. normalize again, run a feed-forward layer, and add that result back too.

The sections take those parts one at a time, starting with the additions that hold them together.

## Add each part's output to a residual stream {#add-to-the-residual-stream}

In [chapter 5](#a-neural-language-model)'s [window model](#predict-from-a-window), each layer replaced its input: the hidden layer's output was all the output layer saw. A transformer block adds to its input instead. Each part computes something from the current vector and adds its output to it: x ← x + attention(x), then x ← x + feed-forward(x). That's a **residual connection**. It comes from image models, where very deep stacks of layers trained well only once each layer added to its input instead of replacing it ([He et al., 2016](https://arxiv.org/abs/1512.03385)).

Two consequences follow. With the additions, the signal can pass "directly from one block to any other block", forwards and, as [gradients](#compute-gradients), backwards, so a deep stack trains far more reliably than it would without them ([He et al., 2016b](https://arxiv.org/abs/1603.05027)). And every position's vector becomes a shared workspace that each part reads from and writes to: the token's embedding goes in at the bottom, each part adds what it computes, and the output layer reads the sum at the top. Elhage and colleagues call this running sum the **residual stream** ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)).

For the store, run its trained model on one validation product. At every layer, measure the length of what attention and the feed-forward layer add, and of the vectors after both additions, averaged over positions. A vector's length here is the square root of the sum of its squared numbers:

[Example: residual-stream](../examples/residual-stream.py)

The vectors enter the first layer with an average length of {{out:residual-stream:length of the vectors entering layer 1}}, and leave the last with {{out:residual-stream:length of the vectors leaving the last layer}}. Every layer adds to them, and what attention adds is comparable in size to what the feed-forward layer adds, though the feed-forward layer adds more at every layer, and by more in the later ones: both parts do real work at every layer. The vectors grow as they go, because each addition piles onto the last. A layer reading the long vectors near the top and one reading the short ones at the bottom would need very differently scaled weights, unless something rescales what each layer reads; the [next section](#normalize-before-each-part) does.

**Working rule:** Write every block as additions to the residual stream, never as replacements, and keep the stream's width the same through the whole model so the additions line up. When you debug a deep model, measure what each part adds, as here: a part that adds almost nothing is doing almost nothing.

## Normalize before each part {#normalize-before-each-part}

The residual stream grows through the layers, as the [previous section](#add-to-the-residual-stream) measured, so each part should read a rescaled copy of it. **Normalization** rescales a vector to a standard size before a part reads it, so the part's weights work the same whether the stream is short or long. The original transformer normalized after each addition, with **layer normalization**, which subtracts each vector's mean and divides by its standard deviation ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)). Xiong and colleagues showed that normalizing before each part instead trains more stably, because only the copy each part reads is rescaled, never the stream itself ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)). And **RMSNorm** drops the mean: it divides each vector by its root mean square, the square root of the average of its squared numbers, and multiplies each dimension by a learned gain ([Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)):

RMSNorm(x) = x / √(mean(x²) + ε) · g

where ε is a tiny number, 10⁻⁵ here, that keeps the division safe, and g is the gain, which starts at 1. Zhang and Sennrich found RMSNorm about as accurate as layer normalization, and faster. The store's model uses it before attention, before the feed-forward layer, and once more before the output layer.

For the store, take the residual stream of the same validation product at both ends of the model, entering the first block and leaving the last. Normalize every position's vector by hand, with the gain left at 1, and compare with `lib.transformer.rms_norm`:

[Example: rms-norm-by-hand](../examples/rms-norm-by-hand.py)

Entering the first block, the vectors' root mean square is {{out:rms-norm-by-hand:root mean square entering block 1, before}} on average; leaving the last, it's {{out:rms-norm-by-hand:root mean square leaving the last block, before}}. After normalization, both are {{out:rms-norm-by-hand:root mean square entering block 1, after}}, and the hand calculation matches the library's at both ends: {{out:rms-norm-by-hand:same as lib.transformer.rms_norm, entering block 1}} and {{out:rms-norm-by-hand:same as lib.transformer.rms_norm, leaving the last block}}. RMSNorm rescales but doesn't centre: a position's numbers can still average away from zero afterwards, by up to {{out:rms-norm-by-hand:largest mean of a position's numbers after, leaving the last block}} here. That's the step it saves over layer normalization.

So though the stream grew many times longer across the layers, every attention and feed-forward layer read vectors of the same scale, times its learned gains.

**Working rule:** Normalize before each part of the block, not after, and normalize the stream once more before the output layer. Use RMSNorm unless you're matching a model that uses layer normalization. When you load another model's weights, check which normalization it uses, where, and with what ε, before anything else: the [mismatched-details example](#measure-mismatched-details) shows what a small mismatch costs.

## Transform each position with a gated feed-forward layer {#transform-each-position}

Attention moves information between positions; the **feed-forward layer** transforms it, at each position separately. It's the same idea as [chapter 5](#a-neural-language-model)'s [hidden layer](#predict-from-a-window): widen the vector, apply a function that isn't a straight line, and narrow the result back to the stream's width. The store's model widens {{out:transformer-parameters:width}} numbers to {{out:transformer-parameters:feed-forward width}}.

The form the LLaMA models use is **gated**. It computes two widened vectors from the input, passes one through a function called SiLU, and multiplies the two number by number, so the first decides how much of the second gets through ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)):

FFN(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down

where ⊙ multiplies number by number, and SiLU(z) = z · σ(z), with σ the logistic function from [chapter 4](#learn-by-predicting-neighbours). SiLU passes large positive numbers almost unchanged and turns large negative ones to nearly zero, smoothly. Shazeer compared several gated forms with the plain one, at the same number of parameters, and found the gated ones predicted better; this form is called SwiGLU.

For the store, compute SiLU at three points. Then run the first block's real feed-forward layer on the same validation product, and count, at each position, how many of its hidden units the gate nearly shuts:

[Example: gated-feed-forward](../examples/gated-feed-forward.py)

SiLU of −3, 0 and 3 gives {{out:gated-feed-forward:silu(-3), silu(0), silu(3)}}: nearly zero for the negative number, nearly unchanged for the positive one. Of the first block's {{out:gated-feed-forward:hidden units}} hidden units, the gate nearly shuts {{out:gated-feed-forward:hidden units nearly shut at a position (gate under 0.05 in size), on average}} at a typical position, between {{out:gated-feed-forward:fewest and most nearly shut at a position}} depending on the position. Which units are open depends on the input, so each position uses its own mix of them. The output has one vector per position at the stream's width again, {{out:gated-feed-forward:output shape}}, ready to add.

The feed-forward layers hold most of a block's parameters: in the store's model, three matrices of {{out:transformer-parameters:width}} by {{out:transformer-parameters:feed-forward width}}, against attention's four of {{out:transformer-parameters:width}} by {{out:transformer-parameters:width}}. Geva and colleagues found that in trained models these layers work like stores of patterns: each hidden unit responds to particular patterns in the text, and pushes the prediction towards particular next tokens ([Geva et al., 2021](https://arxiv.org/abs/2012.14913)).

**Working rule:** Use a gated feed-forward layer with SiLU. Set its width so its three matrices hold as many parameters as a plain layer four times the stream's width, the original transformer's ratio, 2,048 for 512 ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), section 3.3): that's about 8/3 times the width. SmolLM2-360M uses exactly 8/3, 2,560 for 960. The store's model rounds 8/3 of its width up to a multiple of 32, a rule of thumb that keeps matrix sizes convenient for hardware. When you load another model, check whether its feed-forward layer is gated, since a gated layer has three matrices where a plain one has two.

## Give attention positions with rotations {#rotary-positions}

A weighted average ignores the order of what it averages, as [chapter 6](#look-back-with-a-weighted-average) showed, so attention gets at most a weak sense of position from the causal mask alone. The original transformer added a vector for each position to each token's embedding ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), section 3.5). The LLaMA models instead use **rotary position embeddings**, RoPE, which work inside attention, at each layer ([Su et al., 2021](https://arxiv.org/abs/2104.09864); [Touvron et al., 2023](https://arxiv.org/abs/2302.13971), section 2.2). Before scoring, each query and key is rotated by angles that grow with its position. The vector's d numbers are taken in pairs, number i with number i + d/2 in the store's model, and each pair (a, b) is treated as a point on a plane and turned around the origin. At position p, pair i turns by the angle p·θᵢ, where θᵢ = 10000^(−i/(d/2)):

(a, b) → (a cos pθᵢ − b sin pθᵢ, a sin pθᵢ + b cos pθᵢ)

So the first pairs turn fast, a full radian per position, and the last pairs slowly. The line `angles = ...` in `lib.transformer.rope` computes p·θᵢ for every position and pair, and the line after it applies the rotation.

Rotation has two properties that suit attention. It doesn't change a vector's length, so it adds position without distorting content. And if a query at position m is turned by mθᵢ and a key at position n by nθᵢ, their dot product depends only on the difference, (m − n)θᵢ: the score depends on how far apart they are, not on where they sit in the text. A head can learn to look 3 tokens back, and it works the same at the start of a name and deep in a description.

For the store's model's heads of {{out:trained-attention:dimensions per head}} numbers, take a random query and key, rotate them for positions 0 to 19, and compare scaled scores, as [chapter 6](#attention) defines them, at three pairs of positions:

[Example: rope-by-hand](../examples/rope-by-hand.py)

The rotation keeps both vectors' lengths: {{out:rope-by-hand:length kept by the rotation, for the query and the key}}. The score between the key at position 2 and the query at position 5 is {{out:rope-by-hand:score with the key at 2 and the query at 5}}, and between positions 12 and 15 it's {{out:rope-by-hand:score with the key at 12 and the query at 15}}, the same, because both pairs are 3 apart. Between positions 2 and 8, 6 apart, it's {{out:rope-by-hand:score with the key at 2 and the query at 8}}: the same query and key, scored differently because they're further apart.

RoPE has no learned parameters. The angles are fixed, and the model learns queries and keys that make use of them. The store's model applies it to the queries and keys of every head in every layer, which gives [chapter 6's](#attend-to-the-previous-token) previous-token heads a direct way to find the previous token, alongside the weak signal the mask provides.

**Working rule:** Use rotary position embeddings, applied to queries and keys, never values. When you load another model's weights, check how its RoPE pairs the numbers and what base it uses. Some implementations pair neighbouring numbers; others pair the first half with the second, as `lib.transformer.rope` does and as SmolLM2's configuration states with `"rope_interleaved": false`. SmolLM2 uses a base of 100,000 where the store's model uses 10,000.

## Tie the output layer to the embeddings {#tie-the-output}

With all the parts built, the whole model is short enough to read in one piece. `lib.transformer.block` is one block, exactly as the chapter has described it:

```python
def block(x: np.ndarray, layer_weights: dict[str, np.ndarray], heads: int, positions=rope):
    """One transformer block: attention, then the gated feed-forward layer, each added to the residual stream."""
    attended, weights = attention(rms_norm(x, layer_weights["norm1"]), layer_weights, heads, positions)
    x = x + attended
    normalized = rms_norm(x, layer_weights["norm2"])
    x = x + (silu(normalized @ layer_weights["gate"]) * (normalized @ layer_weights["up"])) @ layer_weights["down"]
    return x, weights
```

`lib.transformer.run` passes each token's embedding through the blocks in turn, and these lines of `lib.transformer.forward` turn the final stream into one logit per token in the vocabulary, the scores [chapter 5](#a-neural-language-model)'s softmax turns into probabilities:

```python
    streams, kept = run(ids, name, positions, skip)
    weights, _, _ = load(name)
    logits = rms_norm(streams[-1], weights["final_norm"]) @ weights["embedding"].T
```

The last line holds one more idea: the output layer reuses the embedding table, turned on its side, instead of learning a matrix of its own. A token's logit is the dot product of the final vector with that token's embedding. Press and Wolf found that this **weight tying** lowers perplexity as well as saving the output layer's parameters ([Press and Wolf, 2017](https://arxiv.org/abs/1608.05859)). SmolLM2 ties them too (`"tie_word_embeddings": true` in its configuration), since for a model of its size a separate output layer would be a large share of the whole. The [parameter count](#count-the-parameters) shows what tying saves in the store's model.

**Working rule:** Tie the output layer to the embeddings in a small model, and check whether a model you load ties them before you look for a separate output matrix. Keep your own forward pass short enough to read whole, as `lib.transformer` is, so that every detail the next sections test is visible.

## Check the model against PyTorch {#check-against-pytorch}

PyTorch trained the store's model once, in `recordings/transformer.py`, as it trained [chapter 5](#a-neural-language-model)'s (see [Train with PyTorch](#train-with-pytorch)), and the script saved PyTorch's own logits for every position of one validation product. The training script uses PyTorch's own implementations of RMSNorm, causal attention and SiLU, so comparing them with `lib.transformer` checks the NumPy code against a library's, not against a second copy of itself. RoPE is the exception: PyTorch has no built-in version, so both programs write it out, and the check confirms only that they agree. The [rope-by-hand example](#rotary-positions) checks its properties instead.

For the store, recompute PyTorch's logits with the NumPy model, and compare:

[Example: transformer-same-as-pytorch](../examples/transformer-same-as-pytorch.py)

The NumPy model reproduces all {{out:transformer-same-as-pytorch:logits compared}} of PyTorch's logits, for {{out:transformer-same-as-pytorch:positions compared}} positions, to four decimal places: {{out:transformer-same-as-pytorch:same as PyTorch to 4 decimal places}}, for PyTorch {{out:transformer-same-as-pytorch:PyTorch version}}. Because the check covers a whole text, it covers the parts that depend on position, the mask and the rotations, as well as the norms, the gates and the tied output.

**Working rule:** Check your own forward pass against the training library's outputs on a full text, not a single position. Where the library has its own implementation of a part, train with it, so the check compares two independent implementations; where it doesn't, as with RoPE here, test the part's properties by hand.

## Leave out a block to see what it does {#leave-out-a-block}

The model is checked; now measure what its blocks contribute. Because every block adds to the residual stream, a block can be left out and the model still runs: the stream passes through it unchanged, and the next block reads it as before. That allows a simple experiment, called an **ablation**: remove one part, and measure what the model loses. It shows how much each block contributes to the trained model's predictions, which reading its weights can't.

For the store, score the names and descriptions of the first 100 validation products, {{out:skip-a-block:tokens scored, names and descriptions}} predictions, with the full model and then with each block left out in turn. Before you run it, predict which block's removal hurts most:

[Example: skip-a-block](../examples/skip-a-block.py)

With all four blocks, the [perplexity](#score-with-perplexity) per token is {{out:skip-a-block:all blocks}}. This perplexity covers names and descriptions together, so it can't be set beside [chapter 3's baseline](#the-baseline) or [chapter 5](#a-neural-language-model)'s window model, which are scored on validation names alone; [chapter 8](#training-a-small-model) makes that comparison. Leaving out block 2, 3 or 4 more than doubles it, to {{out:skip-a-block:without block 2}}, {{out:skip-a-block:without block 3}} and {{out:skip-a-block:without block 4}}: each later block contributes a share of the model's skill, and none is dispensable. Leaving out block 1 is a different kind of failure. Perplexity goes to {{out:skip-a-block:without block 1}}, worse than guessing evenly among all {{out:transformer-parameters:vocabulary}} tokens, which would score {{out:transformer-parameters:vocabulary}}.

The difference is in what the later blocks expect. They were trained on a stream that block 1 had already written to, and without it they read vectors unlike any they saw in training. Their predictions become less confident, with the top token's average probability falling from {{out:skip-a-block:all blocks, average probability of the top token}} to {{out:skip-a-block:without block 1, average probability of the top token}}, and they put that probability on the wrong tokens. Block 1 turns raw embeddings into the form everything after it is built on, partly by gathering context from the text so far, as three of [chapter 6's](#read-trained-heads) first-layer heads did. Blocks in a trained model aren't independent modules: each depends on what the earlier ones write.

**Working rule:** To learn what a part of a model does, remove it and measure, as here, rather than inferring from its weights. Expect the first block to matter most, since everything later is built on it; beyond that, measure, because here the second and last blocks mattered about equally, and more than the third. Read a small loss from removing a part with care: the remaining parts were trained alongside it, so the measurement says what the trained model relies on, not what the part could do alone.

## Measure what a mismatched detail costs {#measure-mismatched-details}

Every detail of a block has to match the weights it runs with, and a mismatch doesn't announce itself: the model still produces probabilities. The details most often wrong when loading another model's weights are in RoPE, because implementations differ in how they pair numbers and in their base, as SmolLM2's differs from the store's model's.

For the store, score the same 100 validation products with the trained weights, first as trained, then with RoPE pairing neighbouring numbers instead of halves, with RoPE's base at 100,000, SmolLM2's, instead of 10,000, and with no RoPE at all:

[Example: mismatched-details](../examples/mismatched-details.py)

As trained, the perplexity is {{out:mismatched-details:perplexity, as trained}}. Pairing neighbouring numbers instead of halves raises it to {{out:mismatched-details:perplexity, RoPE pairing neighbouring numbers}}, and leaving RoPE out raises it to {{out:mismatched-details:perplexity, no RoPE}}. Both would be caught by the first sample of generated text. The base is the dangerous one: at 100,000 the perplexity is {{out:mismatched-details:perplexity, RoPE with base 100,000}}, slightly worse, with no error and fluent-looking output, a mistake that could ship.

The other difference from SmolLM2 is **grouped-query attention**: its 15 query heads share 5 sets of keys and values, so its key and value matrices are a third of the size of its query matrix ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)). Loading such weights into a model that expects a key and value for every head fails with a shape error, which is the kind of mismatch you want.

**Working rule:** When you load another model's weights, compare its configuration with your code line by line: normalization and its ε, RoPE's pairing and base, the feed-forward layer's gating, tied outputs, and how many heads share keys and values. Then score text you know, as here, against the original implementation's outputs; a small loss with no error is the case to fear.

## Count where the parameters go {#count-the-parameters}

With every part in place, you can count what the model learns, and where. A model's parameters are what it can learn, and where they sit says what it spends that capacity on. For the store, count the trained model's parameters by part:

[Example: transformer-parameters](../examples/transformer-parameters.py)

The model has {{out:transformer-parameters:total}} parameters, slightly fewer than the {{out:window-model-by-hand:parameters}} of [chapter 5](#a-neural-language-model)'s window model. The embedding table, shared with the output layer, holds {{out:transformer-parameters:embedding (shared with the output)}} of them, {{out:transformer-parameters:embedding's share of the total}} of the total, because it has a row of {{out:transformer-parameters:width}} numbers for each of the {{out:transformer-parameters:vocabulary}} tokens. Those are the store tokenizer's vocabulary plus four markers: start, separator, end, and padding, which fills out short texts in a training batch. The feed-forward layers hold {{out:transformer-parameters:feed-forward}}, attention {{out:transformer-parameters:attention}}, and the normalization gains {{out:transformer-parameters:normalization}}.

The two models are about the same size and spend it differently. The window model put about three quarters of its parameters in its output layer, and could see four tokens back. The transformer puts most of its parameters in its blocks, shares its embeddings with its output, and can see every earlier token in a text of up to {{out:transformer-parameters:longest text, in tokens}} tokens. [Chapter 8](#training-a-small-model) explains how the model was trained, and measures it against the baseline.

**Working rule:** Count parameters by part whenever you design or load a model, as here. At small sizes, the embedding table is a large share of the total, so tie it to the output layer, and choose the vocabulary size with that cost in mind, as [chapter 2's](#choose-how-many-merges) choice of merges did.

## What goes wrong in production {#transformer-in-production}

A transformer's parts have to fit together exactly, and most of what goes wrong is a part that almost fits:

- **Mismatched details can fail quietly.** Where the norms go, how RoPE pairs numbers and its base, whether the feed-forward layer is gated and whether the output is tied all have to match the trained weights. Some mismatches give nonsense at once; others, like the RoPE base [measured here](#measure-mismatched-details), cost a little accuracy with no error. Check against reference outputs.
- **Lengths beyond training are untested.** RoPE gives an angle for any position, but a model trained on texts of up to {{out:transformer-parameters:longest text, in tokens}} tokens has never learned to use the angles beyond. Treat the training length as the model's limit until you've measured past it.
- **Large values can overflow at reduced precision.** Formats with fewer bits keep the same relative precision at every ordinary magnitude, but float16 can't hold values above 65,504, and in 8-bit formats a few large outliers ruin the precision of every other value ([Dettmers et al., 2022](https://arxiv.org/abs/2208.07339)). The store's model's stream peaks at {{out:residual-stream:largest number in the stream}} for the product measured here, far from float16's limit; check the largest values in your model's stream before serving it at lower precision.
- **Results depend on the machine.** As with every trained model in this book, keep the weights, the reference outputs and the exact training script together, and rerun the check after any change to the code or libraries.
- **Every part runs at every position.** The feed-forward layers cost the same at every position, and attention's cost grows with the square of the length, as [chapter 6](#attention-in-production) showed. Measure which dominates for your texts before you optimize either.

## What this chapter decided {#transformer-decisions}

- **The store's transformer uses the LLaMA-style block,** at a toy size: RMSNorm before each part, rotary positions in attention, a gated SiLU feed-forward layer, and an output layer tied to the embeddings. It has {{out:transformer-parameters:layers}} layers, a width of {{out:transformer-parameters:width}}, {{out:transformer-parameters:heads}} heads and a feed-forward width of {{out:transformer-parameters:feed-forward width}}. SmolLM2-360M has the same kinds of parts, larger, with grouped-query attention and a RoPE base of 100,000.
- **The NumPy forward pass in `lib.transformer` is the book's reference,** checked against PyTorch's own RMSNorm, attention and SiLU on every position of a validation product, to four decimal places. Every later example that uses the model runs it.
- **Every block adds to the residual stream, and every part reads a normalized copy of it,** so the stream can grow through the layers while each part sees vectors of a fixed scale.
- **Parts are judged by measuring the model without them, or with them changed.** Leaving out any one block makes the store's model much worse, and leaving out the first makes it useless, so the book treats a trained model's blocks as one system. A wrong RoPE base costs a little and raises no error, so loaded weights are always checked against reference outputs.
- **The model is about the size of [chapter 5](#a-neural-language-model)'s window model, with its parameters spent on the blocks instead of the output layer.** The embedding table, shared with the output, is still {{out:transformer-parameters:embedding's share of the total}} of it, so vocabulary size remains a cost the book watches.
