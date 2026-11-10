---
description: "Use when building, training, evaluating or choosing a language model, or debugging one: next-word prediction and autocomplete, perplexity on held-out text, BPE tokenizers, n-gram baselines with Kneser-Ney smoothing, word embeddings (PPMI, SVD, skip-gram), a small neural model, attention and causal masks, transformer blocks (RMSNorm, RoPE, gated feed-forward), training with AdamW and a learning-rate schedule, decoding (greedy, beam search, temperature, top-k, top-p, repetition penalty), reading an open model's parameters, memory and attention heads, and checking an instruction-tuned model for made-up answers, chat templates, long contexts and LoRA. Small models in NumPy and PyTorch that run on a CPU."
---

## Build and score a counted next-word model

**When:** a project needs a quick next-word model, such as search-box autocomplete, or a way to tell which of two language models predicts its text better.

1. Split the text once, by document, into training, validation and test parts before any model sees it; keep the test part for the final report. [Load the store's text](#load-the-store-text)
2. Pad each text with start and end markers, count adjacent pairs in the training split only, and keep the raw counts. [Count which word follows which](#count-word-pairs)
3. Divide each pair's count by its first word's count, and check each word's probabilities sum to one. [Turn counts into probabilities](#counts-to-probabilities)
4. For autocomplete, offer the top few next words, dropping punctuation. [Suggest the next word as a shopper types](#suggest-next-words)
5. Read samples generated with a fixed seed, but never choose models by them. [Generate names one word at a time](#generate-one-word-at-a-time)
6. Score by perplexity on validation text, with natural logarithms and the end marker counted, only between models with the same tokens and text. [Score a model with perplexity](#score-with-perplexity)
7. Check for infinite perplexity, and count the unseen pairs behind it. [Unseen word pairs break the model](#unseen-pairs-break-the-model)
8. Smooth before shipping, and not with add-one. [Give unseen pairs a little probability](#smooth-unseen-pairs)

**Check:** run [the example](examples/perplexity-on-new-names.py): unsmoothed validation perplexity is {{out:perplexity-on-new-names:validation perplexity}}; [the add-one example](examples/add-one-smoothing.py) gives {{out:add-one-smoothing:validation perplexity, add-one}}.

**Pitfalls:** [What goes wrong in production](#next-word-in-production): counts go stale as the catalog changes; suggestions need an allowed-token filter; one word of context ignores the rest of the query.

## Choose and check a tokenizer

**When:** a project must decide how to cut its text into tokens, or relies on an open model's tokenizer.

1. Measure how many held-out words are unseen; above a percent or two, don't use word tokens. [Words leave gaps](#words-leave-gaps)
2. Count characters per word: that factor multiplies every sequence length. [Characters cover everything](#characters-cover-everything)
3. Learn byte-pair encoding merges from the training split only, keep their order, and break ties by a fixed rule. [Merge the most frequent pair, again and again](#merge-frequent-pairs)
4. Train the tokenizer on the text the model will learn from, and use it for every model you compare. [Train the store's tokenizer](#train-the-store-tokenizer)
5. Choose the merge count where tokens per held-out word stops falling much and the share of rare token types starts climbing. [Choose how many merges](#choose-how-many-merges)
6. With an open model, use its tokenizer as shipped, and budget in its tokens. [Read an open model's tokenizer](#read-an-open-tokenizer)
7. Apply its merges by hand on your text and compare with the library after any upgrade. [Apply SmolLM2's merges by hand](#apply-merges-by-hand)

**Check:** run [the example](examples/vocabulary-size.py): at 4,000 merges a validation word takes {{out:vocabulary-size:tokens per word at 4,000 merges}} tokens, and {{out:vocabulary-size:rare token types at 4,000 merges}} of token types are rare. [The merges example](examples/merges-by-hand.py) must match the library for every name.

**Pitfalls:** [What goes wrong in production](#tokens-in-production): a mismatched tokenizer fails silently; numbers tokenize badly; special-token strings in untrusted text can become control tokens.

## Build an n-gram baseline with Kneser-Ney

**When:** a project needs a strong, simple yardstick before training a learned language model.

1. Never judge an order by perplexity on its own training text. [Longer contexts fit the training text better](#longer-contexts-fit-training)
2. Measure the share of held-out n-grams unseen in training at each order. [Longer contexts run out of evidence](#longer-contexts-run-out)
3. Combine each order with lower ones; choose any fixed weights on held-out text. [Mix short and long contexts](#mix-short-and-long-contexts)
4. When backing off, weight tokens by how many contexts they follow, not how often they occur. [Count the contexts a token follows](#count-contexts-a-token-follows)
5. Use interpolated Kneser-Ney with a discount of 0.75, check one probability by hand, and raise the order only while validation perplexity falls noticeably. [Smooth with Kneser-Ney](#smooth-with-kneser-ney)
6. Test suggestions for rare prefixes apart from common ones. [Autocomplete with three tokens of context](#autocomplete-with-more-context)
7. Record the best counted model's validation perplexity as the baseline for every later model, same text and tokenizer. [The baseline every later model is measured against](#the-baseline)

**Check:** run [the example](examples/kneser-ney-by-hand.py): {{out:kneser-ney-by-hand:by hand}} by hand against {{out:kneser-ney-by-hand:lib.ngrams}} from the module, summing to {{out:kneser-ney-by-hand:probabilities after the context, summed}}. [The order example](examples/kneser-ney-by-order.py) prints the 4-gram baseline, {{out:kneser-ney-by-order:4-gram validation perplexity}}.

**Pitfalls:** [What goes wrong in production](#ngrams-in-production): tables grow, so prune and remeasure; rare contexts get generic answers; hand-written backoff often loses probability, so check sums.

## Build and check word embeddings

**When:** a project needs related tokens or short texts to sit close together as vectors.

1. Use one-hot vectors only as an index for learned ones. [One-hot vectors know nothing](#one-hot-vectors-know-nothing)
2. Count co-occurrence within two to five tokens, never across documents. [Describe a token by the tokens around it](#describe-by-neighbours)
3. Reweight with positive PMI, keeping frequent tokens only. [Weigh co-occurrence by surprise](#weigh-by-surprise)
4. Compress with SVD to 50 to 300 dimensions, then read some nearest neighbours. [Compress counts into dense vectors](#compress-into-dense-vectors)
5. For skip-gram, draw noise tokens by frequency to the power 0.75, and check one update by hand. [Learn vectors by predicting neighbours](#learn-by-predicting-neighbours)
6. Never read high similarity as synonymy without checking for opposites. [Compare learned vectors with counted ones](#compare-learned-and-counted)
7. Judge embeddings by their task, not analogies. [Analogies, and where they fail](#analogies-and-where-they-fail)
8. Average token vectors for a quick view of short texts; expect loose ranking. [Find similar names with vectors](#find-similar-names)

**Check:** run [the example](examples/svd-neighbours.py): {{out:svd-neighbours:cosine(patio, outdoor)}}, against {{out:one-hot-vectors:cosine(patio, outdoor)}} in [the one-hot example](examples/one-hot-vectors.py).

**Pitfalls:** [What goes wrong in production](#vectors-in-production): similar use isn't similar meaning; rare tokens get poor vectors or none; embeddings go stale.

## Train a neural language model and compare it with the baseline

**When:** a project trains a small neural language model and must show whether it helps.

1. Write the forward pass by hand once, printing every shape. [Predict from a window of embeddings](#predict-from-a-window)
2. Check hand-written gradients against nudged estimates. [Compute gradients by backpropagation](#compute-gradients)
3. Check the loss falls on a handful of examples; try a few step sizes. [Take gradient descent steps](#take-gradient-descent-steps)
4. Train with a library, keeping your own forward pass to check it. [Train with PyTorch](#train-with-pytorch)
5. Keep the best epoch's weights by validation loss, never the last. [Stop when validation loss turns up](#stop-at-the-best-epoch)
6. Compare with the counted baseline, split by seen and unseen contexts. [Compare with the baseline](#compare-with-the-baseline)
7. When close, mix the two, choosing the share on held-out text. [Mix the network with the counts](#mix-with-the-counts)
8. Add training text only if it resembles the target; measure on the target. [More text helps only if it's the same kind](#same-kind-of-text)
9. Judge predictions by measurements over many contexts, not examples. [Suggest from similar contexts](#suggest-from-similar-contexts)

**Check:** run [the example](examples/gradients-by-hand.py) ({{out:gradients-by-hand:backprop agrees with nudging to 6 decimal places}}), then [the comparison](examples/neural-vs-baseline.py): {{out:neural-vs-baseline:window model validation perplexity}} against the baseline's {{out:neural-vs-baseline:4-gram baseline validation perplexity}}.

**Pitfalls:** [What goes wrong in production](#neural-in-production): overfitting is silent; mismatched training text hurts; averages hide rare contexts.

## Implement and inspect attention

**When:** a project writes or debugs attention, or asks what a trained model's heads do.

1. Write attention's four lines once and print the weight matrix. [Look back with a weighted average](#look-back-with-a-weighted-average)
2. Mask scores before softmax; test that later positions get zero weight and that changing a later token leaves earlier outputs alone. [Hide the future with a causal mask](#hide-the-future)
3. Divide scores by the square root of the query's size; check what a loaded model does. [Scale the scores](#scale-the-scores)
4. Keep heads at least 32 numbers wide; check the weights are heads by positions by positions. [Several heads look for different things](#several-heads)
5. Read weights on texts you understand, as hypotheses. [Read what trained heads attend to](#read-trained-heads)
6. Test each hypothesis by changing the input, with a control change. [Test what the model copies](#test-what-the-model-copies)
7. Describe heads by averages over a few hundred texts, beside an even spread. [Some heads attend to the previous token](#attend-to-the-previous-token)
8. Measure how far back it attends, report marker weight apart, and keep the first tokens when cutting context. [Reach past the window](#reach-past-the-window)

**Check:** run [the example](examples/causal-mask.py): weight on later tokens is {{out:causal-mask:weight on later tokens}}, and with the mask, whether an earlier output changes with the last token reads {{out:causal-mask:output at coffee changes with the last token, with the mask}}.

**Pitfalls:** [What goes wrong in production](#attention-in-production): cost grows with the square of the length; a wrong mask fails silently; attention weights aren't explanations; a fully masked row becomes NaN.

## Build a transformer, or load another model's weights

**When:** a project implements a transformer block, or runs someone else's weights in its own code.

1. Write blocks as additions to a fixed-width residual stream; measure what each part adds. [Add each part's output to a residual stream](#add-to-the-residual-stream)
2. Normalize before each part and before the output layer, with RMSNorm. [Normalize before each part](#normalize-before-each-part)
3. Use a gated SiLU feed-forward layer, about 8/3 the stream's width. [Transform each position with a gated feed-forward layer](#transform-each-position)
4. Rotate queries and keys, never values; check pairing and base. [Give attention positions with rotations](#rotary-positions)
5. Tie the output layer to the embeddings in a small model. [Tie the output layer to the embeddings](#tie-the-output)
6. Check your forward pass against the library on a full text. [Check the model against PyTorch](#check-against-pytorch)
7. Remove a block and measure, rather than read its weights. [Leave out a block to see what it does](#leave-out-a-block)
8. Before loading weights, compare the configuration line by line: normalization and ε, RoPE pairing and base, gating, tied outputs, shared keys and values. Then score known text against reference outputs. [Measure what a mismatched detail costs](#measure-mismatched-details)
9. Count parameters by part. [Count where the parameters go](#count-the-parameters)

**Check:** run [the example](examples/transformer-same-as-pytorch.py) ({{out:transformer-same-as-pytorch:same as PyTorch to 4 decimal places}}), then [the mismatch example](examples/mismatched-details.py): {{out:mismatched-details:perplexity, as trained}} as trained, {{out:mismatched-details:perplexity, RoPE with base 100,000}} with the wrong RoPE base, and no error.

**Pitfalls:** [What goes wrong in production](#transformer-in-production): mismatched details fail quietly; lengths beyond training are untested; large values overflow at reduced precision.

## Train a small transformer

**When:** a project trains a small transformer on its own text and must pick the run to keep.

1. Build inputs and shifted targets, mask padding out of the loss, shuffle with a fixed seed, and measure padding. [Turn products into batches](#turn-products-into-batches)
2. Use AdamW with modest weight decay, and clip gradients by combined length. [Update with AdamW and clipped gradients](#update-with-adamw)
3. Warm up for a few hundred steps, then decay; check saved step sizes against the schedule. [Warm up, then decay the step size](#warm-up-then-decay)
4. Keep the best epoch by validation loss on each kind of text; never compare perplexities across kinds. [Read the training curve](#read-the-training-curve)
5. Compare with the baseline on exactly the same predictions. [Compare with the baseline on names](#compare-on-names)
6. Measure each kind of text served, split by seen and unseen contexts, and try a mix. [Compare on descriptions](#compare-on-descriptions)
7. Prefer new text to more epochs, and change one thing per run. [What width, data and steps buy](#what-width-data-and-steps-buy)
8. Test autocomplete on rare and common prefixes, with controls. [Suggest the next word in the search box](#autocomplete-in-the-search-box)

**Check:** run [the example](examples/learning-rate-schedule.py) ({{out:learning-rate-schedule:same as the recorded step sizes}}), then [the comparison](examples/transformer-on-descriptions.py): {{out:transformer-on-descriptions:transformer, validation perplexity on descriptions}} against the 4-gram's {{out:transformer-on-descriptions:4-gram baseline, validation perplexity on descriptions}}.

**Pitfalls:** [What goes wrong in production](#training-in-production): a schedule bug looks like a model problem; choices made on validation data flatter the result; repeated text gets memorized.

## Choose decoding settings

**When:** a project generates text and must choose how each next token is picked.

1. Use greedy decoding for short, repeatable output; expect loops on longer text. [Choose the likeliest token](#choose-the-likeliest-token)
2. Use beam search for short completions; rank longer text per token. [Search for the likeliest text](#search-for-the-likeliest-text)
3. Keep temperature at or a little below 1. [Sample, with a temperature](#sample-with-a-temperature)
4. For varied text, use top-p around 0.9 or top-k around 20, with a fixed seed. [Sample, and cut the tail](#sample-and-cut-the-tail)
5. Compare settings on measures against real text, then read samples and check every factual claim. [Measure what each setting writes](#measure-each-setting)
6. With greedy or low-temperature decoding, add a repetition penalty of 1.2 to 1.3 and check length. [Penalize repetition](#penalize-repetition)
7. Set a minimum length if the training text has empty examples. [Forbid an empty description](#forbid-an-empty-description)

**Check:** run [the example](examples/measure-decoding.py): repeated 4-token phrases are {{out:measure-decoding:greedy, repeated 4-token phrases}} for greedy against {{out:measure-decoding:real descriptions, repeated 4-token phrases}} for real descriptions; [the minimum-length example](examples/minimum-length.py) prints the empty description's probability, {{out:minimum-length:P(the description is empty)}}.

**Pitfalls:** [What goes wrong in production](#decoding-in-production): fluent text states made-up facts; gaps in the training text become outputs; settings drift with each retrained model.

## Size up an open model before using it

**When:** a project considers an open-weights model and needs its cost, design and fit to its own text.

1. Count parameters by part from the weights, reading shapes for tying, grouped keys and values, and gating. [Count the parameters at scale](#parameters-at-scale)
2. Count from the configuration too; a mismatch means a misread detail. [Compare a family's sizes](#grow-the-same-design)
3. Budget memory as weights plus a key and value cache per text, and cap text length. [Count what it costs to run](#cost-to-run)
4. Score your own text per word, never per token, against a model trained on it. [Score the store's text with a general model](#score-the-store-text)
5. Describe heads by averages beside an even spread; keep the first token. [Find the same heads at scale](#same-heads-at-scale)
6. Run the same input-change tests on every model. [Test copying at scale](#copying-at-scale)
7. Measure its suggestions on your own prefixes. [Suggest the next word at scale](#suggestions-at-scale)
8. Compare candidates side by side, and choose a model per job. [Set the two models side by side](#side-by-side)

**Check:** run [the example](examples/open-model-on-the-store.py): per-word perplexity on names is {{out:open-model-on-the-store:names, per-word perplexity, SmolLM2-360M}} for the open model against {{out:open-model-on-the-store:names, per-word perplexity, the store's model}} for the small model trained on the text.

**Pitfalls:** [What goes wrong in production](#open-models-in-production): its text and tokenizer aren't yours; its details must match the running code; its revision changes, so pin it; check its licence.

## Check an instruction-tuned model's answers

**When:** a project uses a tuned model to answer or write about its own data, and must know what it gets wrong.

1. When tuning, mask the request out of the loss; for preferences, keep the reference model's log-probabilities. [Build the two training losses by hand](#build-the-losses)
2. Use a tuned model to follow requests and a base model to continue text. [Ask the base model and the tuned model](#ask-base-and-tuned)
3. Apply the model's own chat template through the library, check its default system message, and record the full prompt. [Format the request as the model was tuned](#format-the-request)
4. Ask questions the input can't answer, count made-up answers, and check claims against the data. [Measure made-up product details](#measure-made-up-details)
5. Measure how confidence relates to being right before using a threshold. [Check whether confidence warns you](#check-confidence)
6. Put key details at the start or end of the context, and test each position. [Find what a longer context loses](#find-what-context-loses)
7. Fine-tune on your text, with LoRA when memory is tight, and measure on held-out text. [Adapt a model to your text](#adapt-a-model)
8. Keep one fixed yardstick, then test the job itself before serving. [Look back at the yardstick](#look-back-at-the-yardstick)

**Check:** run [the example](examples/made-up-materials.py): where the data names no material, the tuned model made one up {{out:made-up-materials:tuned model, data names none, made up}} times and declined {{out:made-up-materials:tuned model, data names none, declined}}; in [the confidence example](examples/answer-confidence.py), {{out:answer-confidence:tuned model, made up answers above 0.5 per token}} made-up answers pass a 0.5 threshold.

**Pitfalls:** [What goes wrong in production](#instruction-models-in-production): it answers when it should say it doesn't know; fluency hides gaps; its behaviour depends on its revision and template.
