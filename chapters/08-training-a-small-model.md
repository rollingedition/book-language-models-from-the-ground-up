---
chapter_id: training-a-small-model
---

# Training a small model

By the end of this chapter you can train a small transformer from start to finish and judge the result. You'll turn texts into batches, update the weights with AdamW, set the step size's schedule, read the training curve, measure the model fairly against the counted baseline, and find out what more width, more text and more training buy at this scale. You'll also see where the model beats counting, and where it needs counting beside it.

[Chapter 7](#the-transformer-block) assembled the store's model and used its trained weights to check every part. This chapter is about where those weights came from: one training run, recorded in `recordings/transformer.py`. The store is the online furniture and home-goods store this book follows, and its text is the names and descriptions of its products. The store's model learned from that text to predict each next token, and this chapter measures how well it does on the two jobs it's for: suggesting the next word of a product name in the search box, and predicting the store's descriptions, where text runs long enough for distant context to matter.

As in [chapter 5](#train-with-pytorch), training ran once, in PyTorch, because its results depend on the machine. The examples read what it saved: the weights, the curves and the settings, recorded in the run's manifest. To train it yourself, install `recordings/requirements.txt` in its own environment, which adds PyTorch's CPU build, and run `python recordings/transformer.py` from the book's folder: it writes the same files to `examples/data/recorded/transformer/`, and runs on a laptop's processor.

## Turn products into batches {#turn-products-into-batches}

Training starts by turning the store's catalog into numbers the model can learn from. Each product becomes one text: the start marker, the name, the separator `<sep>`, the description and the end marker, tokenized by the [store's tokenizer](#train-the-store-tokenizer). A text longer than the model's limit is cut. Each training step takes a **batch** of texts at once, so the arithmetic runs on whole matrices, and texts of different lengths are filled out to the longest in the batch with a padding token.

Each text then gives the model one prediction per position. The **inputs** are its tokens without the last, and the **targets** are its tokens without the first, so position i is trained to predict token i + 1. Thanks to the [causal mask](#hide-the-future), one forward pass makes every one of those predictions at once. The loss is averaged over the targets that aren't padding: the padding costs computation but teaches nothing.

For the store, tokenize every training product, shuffle the texts, and build one epoch's batches the way the training script does, measuring how much of each batch is padding:

[Example: training-batches](../examples/training-batches.py)

The training split holds {{out:training-batches:training texts}} texts and {{out:training-batches:tokens in them}} tokens, counting each text's start, separator and end markers, which [chapter 4's](#describe-by-neighbours) count of {{out:cooccurrence-counts:training tokens}} leaves out, less what the length limit cuts, and only {{out:training-batches:texts cut at the length limit, 160 tokens}} of the texts are cut at the length limit. A batch of {{out:training-batches:texts per batch}} texts gives inputs and targets of the shapes {{out:training-batches:one batch's inputs and targets}}: one row per text, one column per position. Over one shuffled epoch, {{out:training-batches:share of targets that are padding, over one shuffled epoch}} of the targets are padding, between {{out:training-batches:least and most padding in a batch}} depending on the batch, because one long description sets the width for every shorter text beside it. One pass over the training texts, an **epoch**, takes {{out:training-batches:steps per epoch}} steps, and the recorded run took {{out:training-batches:steps in the recorded run}}, {{out:training-batches:epochs in the recorded run}} epochs.

Padding on that scale is waste: nearly half of each forward pass computes predictions the loss then ignores. It doesn't change what the model learns, since padding never counts, but it nearly doubles the cost of each step. Krell and colleagues found padding up to half of all tokens in common datasets, and proposed **packing**, several texts end to end in each row, with the attention kept from crossing between them ([Krell et al., 2021](https://arxiv.org/abs/2107.02027)). Putting texts of similar length in the same batch reduces padding too.

**What to do:** build batches as inputs and shifted targets, mask padding out of the loss, and shuffle the texts each epoch with a fixed seed, so the run can be repeated. Measure the share of padding in your batches, as here; if it's large, pack texts or group them by length before you buy a bigger machine.

## Update with AdamW and clipped gradients {#update-with-adamw}

With batches in hand, each step is the loop of [chapter 5](#take-gradient-descent-steps): forward pass, loss, [gradients](#compute-gradients), update. These lines of `recordings/transformer.py` are the whole of it, run once per batch:

```python
        ids = pad([train[i] for i in order[start:start + SETTINGS["batch"]]])
        logits = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], ignore_index=index[PAD])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

The first three lines are the [previous section](#turn-products-into-batches): a padded batch, inputs without the last token, and a loss over the shifted targets that ignores padding. Two lines differ from [chapter 5](#a-neural-language-model).

The optimizer is **AdamW**, the [Adam](#train-with-pytorch) of [chapter 5](#a-neural-language-model) with **weight decay** applied separately: at every step, each weight is also reduced by its value times the step size times a decay factor, here {{out:learning-rate-schedule:weight decay}} ([PyTorch's AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)). Loshchilov and Hutter showed that applying the decay this way, apart from the gradient, works better with Adam than adding it to the loss ([Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)). Decay is a form of [regularization](#stop-at-the-best-epoch), which chapter 5's window model lacked: it pulls every weight towards zero, so a weight stays large only while the training text keeps pushing it there.

The script also **clips** the gradients: when their combined length, measured as if every gradient were one long vector, exceeds 1.0, it scales them all down to that length ([PyTorch's clip_grad_norm_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)). One unusual batch then can't throw the weights far off. Pascanu and colleagues proposed clipping for exactly that: rare, very large gradients that would otherwise undo much of what training had learned ([Pascanu et al., 2013](https://arxiv.org/abs/1211.5063)).

**What to do:** use AdamW with a modest weight decay, and clip gradients by their combined length. Keep the training step short enough to read whole, as here, so that every choice in it is visible: what the loss ignores, what the optimizer decays and what gets clipped.

## Warm up, then decay the step size {#warm-up-then-decay}

The last line of the loop, `scheduler.step()`, changes the step size after every batch, following a **schedule**. For the first {{out:learning-rate-schedule:warm-up steps}} steps, it rises in a straight line from nearly zero to its peak, {{out:learning-rate-schedule:peak step size}}: a **warm-up**, as the transformer paper's schedule had ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), section 5.3). In Adam's first steps, its running average of squared gradients rests on only a few batches, so its step sizes vary widely; Liu and colleagues showed that warm-up works by reducing that variance ([Liu et al., 2020](https://arxiv.org/abs/1908.03265)). After the warm-up, the step size falls along half a cosine curve, to a tenth of the peak by the last step ([Loshchilov and Hutter, 2017](https://arxiv.org/abs/1608.03983)):

rate = peak × (0.1 + 0.9 × ½ (1 + cos(π × progress)))

where progress runs from 0 at the end of the warm-up to 1 at the last step. Large steps early cover ground; small steps late let the weights settle into a low point instead of bouncing around it.

For the store, compute the schedule by hand from the recorded settings, and compare it with the step sizes the training script saved as it ran, every tenth step:

[Example: learning-rate-schedule](../examples/learning-rate-schedule.py)

The step size starts at {{out:learning-rate-schedule:step size at step 0}}, reaches {{out:learning-rate-schedule:step size at step 100}} halfway through the warm-up and its peak, {{out:learning-rate-schedule:step size at step 200}}, at the warm-up's end. Halfway through training it's {{out:learning-rate-schedule:step size at step 8,568}}, and at the last step {{out:learning-rate-schedule:step size at step 17,135}}. The hand calculation matches every saved value: {{out:learning-rate-schedule:same as the recorded step sizes}}.

The decay matters, and the recordings show how much. The comparison run trained for four epochs at the same width, with the same seed and the same batches, so its schedule finished its decay at step {{out:width-data-and-steps:width 128, 4 epochs, steps}}. It scored {{out:width-data-and-steps:width 128, 4 epochs, perplexity on names}} on the validation names at that step. The recorded run, at the same step, was still near its peak step size, and scored {{out:transformer-training-curve:validation perplexity on names, epoch 4}}. Same data, same steps: the difference is the decay.

**What to do:** warm the step size up over the first few hundred steps, then decay it, and match the schedule's length to the run's. Save the step size as training runs, as the training script does, and check it against the schedule you meant: a schedule bug looks like a training problem, and is far easier to find this way.

## Read the training curve {#read-the-training-curve}

With its batches, updates and schedule set, the store's model trained for {{out:training-batches:epochs in the recorded run}} epochs, and the script measured it on the validation split after each one, on names and descriptions separately. It kept the weights from the epoch with the lowest validation loss on names and descriptions added together, as [chapter 5](#stop-at-the-best-epoch) kept its best epoch.

For the store, read the recorded curves as [perplexity](#score-with-perplexity) per token:

[Example: transformer-training-curve](../examples/transformer-training-curve.py)

Validation perplexity falls until epoch {{out:transformer-training-curve:epoch kept, the lowest name and description losses added together}}, the epoch kept, to {{out:transformer-training-curve:validation perplexity on names, kept epoch}} on names and {{out:transformer-training-curve:validation perplexity on descriptions, kept epoch}} on descriptions, and then levels off: the last epoch is no better. As in [chapter 5](#a-neural-language-model), epoch 1's training perplexity is above validation, because it averages over the epoch's first batches, when the model knew nothing. The training column mixes names and descriptions, as the training text does, so it shows the trend, not a number to set beside either validation column.

[Chapter 5](#a-neural-language-model)'s window model stopped improving after epoch {{out:training-curve:epoch kept}}; this model kept improving for far longer. The two runs differ in several ways that could explain it. This model learned from names and descriptions, {{out:training-batches:tokens in them}} tokens, against the window model's {{out:more-text-of-another-kind:training tokens, names}} from names alone; its step size decays; it has weight decay; and attention applies the same weights at every position. The run doesn't separate those reasons. What it does show is the gap between training and validation, which grows every epoch: by the end, the model predicts its training texts far better than new ones, so it's starting to memorize.

Descriptions score lower than names, but that doesn't mean the model is better at descriptions. Each kind of text has its own predictability: descriptions repeat phrases from their templates and their names, and names pack a new product into a few words. Perplexities compare models on the same text, never texts with each other.

**What to do:** record validation loss per epoch on each kind of text you care about, and keep the best epoch's weights. Read the gap between training and validation as the warning, not the training loss alone. Never compare perplexities across different kinds of text.

## Compare with the baseline on names {#compare-on-names}

With the training read, the result can be measured. The book's yardstick is the validation names: every model since [chapter 3](#counting-further-back) has been scored on them, and the [4-gram baseline](#the-baseline), trained on the training names, scores {{out:transformer-on-names:validation perplexity on names, transformer share 0.0}}. That baseline is the strongest counted one for names: the same 4-gram trained on whole texts, names and descriptions, scores {{out:transformer-on-descriptions:this 4-gram, trained on whole texts, validation perplexity on names}} on them, because descriptions dilute its counts of how names run.

The comparison has to count the same predictions. The 4-gram predicts each name's tokens and then its end marker; the transformer predicts each name's tokens and then `<sep>`, which plays the same part, since it marks where the name ends. A name needs nothing after it, so the transformer reads the name alone.

Before you run it, predict whether the transformer, alone, beats the counted baseline that [chapter 5](#a-neural-language-model)'s window model couldn't. Then score both models, and mixtures of the two, as [chapter 5](#mix-with-the-counts) did:

[Example: transformer-on-names](../examples/transformer-on-names.py)

Both models score {{out:transformer-on-names:validation tokens scored, each name's end included}} predictions. Alone, the transformer scores {{out:transformer-on-names:validation perplexity on names, transformer share 1.0}}, slightly worse than the baseline's {{out:transformer-on-names:validation perplexity on names, transformer share 0.0}}, though far better than [chapter 5](#a-neural-language-model)'s window model, {{out:neural-vs-baseline:window model validation perplexity}}. Mixed half and half, the two score {{out:transformer-on-names:validation perplexity on names, transformer share 0.5}}, the best of any model in this book on names; chapter 5's best mix scored {{out:mix-neural-and-counts:validation perplexity, network share 0.2}}.

So on names, counting is still hard to beat alone, for the reason [chapter 5](#a-neural-language-model) gave: names are short and formulaic, and three tokens of context see most of what a name holds. But the transformer is wrong in different places from the counts, and the mix gains from both. The share was chosen on the validation names, so the gain is slightly optimistic; the test names stay untouched for the book's final comparison.

**What to do:** compare a new model with the strongest baseline you can build on exactly the same predictions, including how each marks the end of a text. When a learned model and a counted one are close, mix them, and choose the share on held-out text.

## Compare on descriptions {#compare-on-descriptions}

The [names comparison](#compare-on-names) favours counting, because names are short. Descriptions run many times longer, and [chapter 6](#test-what-the-model-copies) showed the model copying from the name many tokens back, so this is where its longer reach should pay. A fair baseline here is a 4-gram trained on the same texts the transformer learned from, the name, `<sep>` and description joined, and scored on the same tokens: each description's tokens and its end, with the same text before them.

For the store, fit that 4-gram on every training text, and score both models, and an even mix, on the descriptions of the first 500 validation products, leaving out the texts cut at the length limit and the products with no description. Then split the tokens, as [chapter 5](#compare-with-the-baseline) did, by whether their three-token context occurs in the training texts:

[Example: transformer-on-descriptions](../examples/transformer-on-descriptions.py)

On {{out:transformer-on-descriptions:description tokens scored}} tokens from {{out:transformer-on-descriptions:products scored}} products, the 4-gram scores {{out:transformer-on-descriptions:4-gram baseline, validation perplexity on descriptions}} and the transformer {{out:transformer-on-descriptions:transformer, validation perplexity on descriptions}}: on descriptions, the transformer beats counting alone, the first learned model in this book to beat a counted model of the same text on its own.

The split shows where. {{out:transformer-on-descriptions:description tokens whose 3-token context is in the training texts}} of the tokens have a context the training texts contain, and on those the two models tie: {{out:transformer-on-descriptions:4-gram, seen contexts}} for the 4-gram, {{out:transformer-on-descriptions:transformer, seen contexts}} for the transformer. On the rest, the transformer scores {{out:transformer-on-descriptions:transformer, unseen contexts}} where the 4-gram scores {{out:transformer-on-descriptions:4-gram, unseen contexts}}. Descriptions are formulaic enough that counts do as well as the transformer wherever they have seen the context; where they haven't, the transformer reads further back and relates what it sees to what it learned, and the counts can only back off.

That's also why the mix works so well. Half and half, the two score {{out:transformer-on-descriptions:half and half, validation perplexity on descriptions}}, far below either alone: on any one token, at least one of the two is often confident and right, and the even mix keeps half of that probability, enough to avoid the worst surprises of each.

**What to do:** measure a model on the kind of text it will predict, against a baseline that sees the same text, and expect the ranking to change from one kind of text to another. Split the comparison by seen and unseen contexts to see where each model wins. Try a mix before concluding that either model is enough: here, the mix gained more than any change to either model that this chapter measured.

## What width, data and steps buy {#what-width-data-and-steps-buy}

The recorded model is one choice among many: {{out:transformer-parameters:width}} numbers wide, trained on all the training text for {{out:training-batches:epochs in the recorded run}} epochs. Studies of much larger models found that loss falls smoothly as models get bigger, see more text and train longer, and that bigger models learn more from each token they see ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)). For a fixed budget of computation, Hoffmann and colleagues found that model size and training text should grow together ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). And Muennighoff and colleagues found that repeating the same text for up to about four epochs is nearly as good as new text, with less gained from each further repetition ([Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)). Whether any of that holds at this toy scale is something to measure, not assume.

So the training script recorded seven cheaper runs beside the main one, each changing one thing:

- three widths, 64, 128 and 256, each trained for four epochs, with every head kept {{out:trained-attention:dimensions per head}} numbers wide;
- random quarters and halves of the training products, each trained for four epochs;
- the same quarter and half trained for as many steps as all the data's four epochs, by repeating them more often.

The numbers are the validation perplexities the script recorded, at each run's best epoch and at its last. For the store, read all eight runs:

[Example: width-data-and-steps](../examples/width-data-and-steps.py)

**Width** helps a lot, at a price. With the same steps, the names' perplexity falls from {{out:width-data-and-steps:width 64, 4 epochs, perplexity on names}} at width 64 to {{out:width-data-and-steps:width 128, 4 epochs, perplexity on names}} at 128 and {{out:width-data-and-steps:width 256, 4 epochs, perplexity on names}} at 256. The widest model, trained for four epochs, even beats the recorded model trained for sixteen, {{out:width-data-and-steps:width 128, 16 epochs, perplexity on names}} on names. But it has {{out:width-data-and-steps:width 256, 4 epochs, parameters}} parameters against {{out:width-data-and-steps:width 128, 4 epochs, parameters}}, and each step costs about that much more.

**New text** helps at no extra cost per step. With the same steps, a random quarter of the products scores {{out:width-data-and-steps:a quarter of the data, 16 epochs, perplexity on names}} on names at its best, half {{out:width-data-and-steps:half the data, 8 epochs, perplexity on names}}, and all of them {{out:width-data-and-steps:width 128, 4 epochs, perplexity on names}}. Doubling the text from half to all gained less than doubling the width from 128 to 256, but it cost no extra computation, and the first doubling, from a quarter to half, gained more than either.

**Repetition** helps for a while, then hurts. A quarter of the text seen four times scores {{out:width-data-and-steps:a quarter of the data, 4 epochs, perplexity on names}}; seen more often, it improves until epoch {{out:width-data-and-steps:a quarter of the data, 16 epochs, best epoch}}, then gets worse, ending at {{out:width-data-and-steps:a quarter of the data, 16 epochs, perplexity on names, last epoch}} after sixteen passes: the model memorizes what it has seen too often. Repeated text never caught up with new text at the same number of steps.

**More steps** help least. The recorded model's sixteen epochs improve on four, {{out:width-data-and-steps:width 128, 16 epochs, perplexity on names}} against {{out:width-data-and-steps:width 128, 4 epochs, perplexity on names}} on names, for four times the steps.

These runs also show the limits of this book's scale. The store's training text is fixed at what its catalog holds, {{out:training-batches:tokens in them}} tokens. SmolLM2-360M, whose design [chapter 7](#the-transformer-block) compared with the store's model, was trained on 4 trillion tokens ([model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/f8027fd0eaeea54caa13c31d31b9fdc459c38b49/README.md)), more than a million times as much, which lets a model 960 numbers wide and 32 layers deep keep learning from new text instead of memorizing old.

**What to do:** get more text of the kind you'll predict before repeating what you have more than a few times; it costs nothing per step. With computation to spare, try a wider model for fewer epochs before training a narrow one longer, and measure both. Change one thing per run and hold the number of steps fixed where it matters, or the comparison measures two things at once.

## Suggest the next word in the search box {#autocomplete-in-the-search-box}

Averages say how good a model is; the search box shows what that means to a shopper. [Chapter 3's](#autocomplete-with-more-context) 4-gram suggested `&` and `with` after `outdoor bar`, a context no training name contains, and [chapter 5's](#suggest-from-similar-contexts) window model suggested `stool`. A few training descriptions do contain `outdoor bar`, and the transformer learned from them as well as from the names, while the 4-gram learned from names alone.

For the store, ask the transformer, and the half-and-half mix the [names comparison](#compare-on-names) favoured, for their three likeliest next tokens after the same five prefixes, each treated as the start of a name. A suggestion of `<sep>` means the model expects the name to end there:

[Example: transformer-autocomplete](../examples/transformer-autocomplete.py)

After `outdoor bar`, the transformer gives `stool` a probability of {{out:transformer-autocomplete:P(stool | outdoor bar), transformer}}, against the 4-gram's {{out:transformer-autocomplete:P(stool | outdoor bar), 4-gram}}, and the mix {{out:transformer-autocomplete:P(stool | outdoor bar), half and half}}, still its top suggestion. The control prefixes show the models aren't suggesting `stool` after every `bar`: after `kitchen bar` and `wine bar` they suggest other continuations, and after `wine bar` both expect, with good reason, that the name may already be complete. After `patio bar`, a context the names contain often, the mix is more confident in `stool` than the transformer alone.

One prefix is an example, not a measure. The measure is the names comparison, and there the mix won. For the search box, that's the model to serve.

**What to do:** serve the mix for autocomplete, and test it on rare prefixes as well as common ones, with control prefixes beside each, as here. Treat a single striking suggestion as an illustration of a measured gain, never as the evidence for it.

## What goes wrong in production {#training-in-production}

Training goes wrong in ways that look fine while they happen:

- **Padding wastes computation.** Nearly half of each step's predictions were padding here. Pack texts, or group them by length, before you buy more computation.
- **A schedule bug looks like a model problem.** As a rule of thumb, a warm-up that never ends, or a decay that starts too early, shows up as a loss that falls too slowly; the decay comparison above shows how much a schedule can move the result. Save the step size, and check it against the schedule you meant.
- **Choices made on validation data flatter the result.** The kept epoch and the mixing share were both chosen on the validation split, so its numbers are slightly optimistic. Keep a test split untouched until the end.
- **Repeated text gets memorized.** A small dataset seen many times looks fine on the training loss and poor on new text. Count how many times training sees each text, and prefer new text to more epochs.
- **One model is measured on one kind of text.** The transformer ranked second to the counts on names and first on descriptions. Score every model on each kind of text it will serve before choosing which to deploy.
- **Results depend on the machine.** PyTorch doesn't guarantee identical results across releases or platforms ([Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)). Record the weights, the curves and the exact script, as `recordings/transformer.py` does with the script's hash, and keep them together.

## What this chapter decided {#training-decisions}

- **The store's model is trained in one recorded run:** batches of {{out:training-batches:texts per batch}} product texts, AdamW with weight decay {{out:learning-rate-schedule:weight decay}}, gradients clipped, a warm-up of {{out:learning-rate-schedule:warm-up steps}} steps to a peak step size of {{out:learning-rate-schedule:peak step size}}, cosine decay to a tenth, and the weights kept from epoch {{out:transformer-training-curve:epoch kept, the lowest name and description losses added together}} of {{out:training-batches:epochs in the recorded run}}.
- **Every comparison counts the same predictions.** On names, both models predict each name's tokens and its end, against [chapter 3](#counting-further-back)'s names-only baseline, the stronger one for names. On descriptions, a 4-gram trained on the same texts scores the same tokens, with the same text before them.
- **On the validation names, the transformer alone roughly ties the 4-gram baseline, and mixed half and half with it scores {{out:transformer-on-names:validation perplexity on names, transformer share 0.5}},** the best on names in this book so far. The mix is the model the book uses for autocomplete, replacing chapter 5's mix of the window model and the counts.
- **From this chapter on, models are also measured on the validation descriptions.** There the transformer beats the baseline alone, {{out:transformer-on-descriptions:transformer, validation perplexity on descriptions}} against {{out:transformer-on-descriptions:4-gram baseline, validation perplexity on descriptions}}, by winning where the counts haven't seen the context, and the mix scores {{out:transformer-on-descriptions:half and half, validation perplexity on descriptions}}.
- **At this scale, new text is the cheapest gain, width the largest at a price, and repetition helps only for a few passes.** The recorded model stays at width {{out:transformer-parameters:width}}, because every example from [chapter 6](#attention) on reads it and the examples have to run in seconds on a laptop; a wider one would score better.
