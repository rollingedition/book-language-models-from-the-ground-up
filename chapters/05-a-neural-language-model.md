---
chapter_id: a-neural-language-model
---

# A neural language model

By the end of this chapter you can build a neural language model that predicts the next token from a window of earlier tokens, compute its gradients by hand and check them, train it with PyTorch, and measure it fairly against the counted baseline. You'll also see that a learned model doesn't win by being learned: this one loses to counting on its own, and helps when the two are mixed.

[Chapter 3](#counting-further-back) built the book's baseline, a 4-gram Kneser-Ney model that predicts each token of the store's product names from the three tokens before it, by counting. Its limit was that it can't relate one context to another: it has counts for `patio bar` and none for `outdoor bar`, and nothing links the two. [Chapter 4](#words-as-vectors) showed that vectors can relate tokens, since `patio` and `outdoor` get close vectors when they're learned from the store's text. This chapter puts vectors inside a model. A **neural language model** reads the previous tokens as vectors, combines them with learned arithmetic, and produces a probability for every token in the vocabulary. It learns its vectors and its arithmetic together, by being trained to predict the next token. The design is the one Bengio and colleagues introduced ([Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)).

The store, the online furniture and home-goods store this book follows, supplies the text, and the test is the one every model since [chapter 3](#counting-further-back) faces: [perplexity](#score-with-perplexity) per token on the store's validation names, through the store's tokenizer. Lower is better, and the baseline scores {{out:neural-vs-baseline:4-gram baseline validation perplexity}}.

## Predict from a window of embeddings {#predict-from-a-window}

The model looks at a fixed **window** of the previous tokens, four here, the same context a 5-gram counts. Each token is looked up in an **embedding table**, a matrix with one row of numbers per token in the vocabulary, as in [chapter 4](#compress-into-dense-vectors), except that this table is learned for predicting the next token rather than computed from counts. The four rows are joined side by side into one long vector, x. A **hidden layer** computes h = tanh(xW + b): it multiplies x by a matrix of weights W, adds a vector of biases b, and passes each result through tanh, a function that squashes any number into the range from −1 to 1, so the layer can combine its inputs in ways sums alone can't. An **output layer** then computes z = hU + c, one score per token in the vocabulary, called the **logits**.

The last step turns scores into probabilities. **Softmax** exponentiates every logit and divides by the sum, so the results are positive and sum to one, and a higher score means a higher probability:

P(token i) = exp(zᵢ) / Σⱼ exp(zⱼ)

For the store, take the model the book trains in this chapter, load its weights, and run one prediction by hand: the next token after a name that starts `patio bar`. Each line of the example is one of the formulas above:

[Example: window-model-by-hand](../examples/window-model-by-hand.py)

The window is padded with start markers, because the name has only two tokens so far. Each of the four tokens becomes a vector of 64 numbers; joined, they make one vector of 256; the hidden layer turns that into 256 more; and the output layer gives one score for each of the {{out:window-model-by-hand:vocabulary size}} tokens the model knows, the store tokenizer's vocabulary plus the start and end markers. The model has {{out:window-model-by-hand:parameters}} parameters, the numbers it learns. About three quarters of them are in the output layer's weights, {{out:window-model-by-hand:parameters in output_w}}, because it needs a column for every token, and most of the rest are in the embedding table. Its prediction is sensible: `stool` with probability {{out:window-model-by-hand:P(stool</w> | patio bar)}}, then `with` and `&`, which continue names such as `patio bar with stools`.

**Working rule:** Build the forward pass of any model you train by hand at least once, and print the shape of every intermediate array, as this example does: a wrong shape is a common bug in neural code, and the shapes say exactly what the model computes. Expect the output layer to hold most of the parameters whenever the vocabulary is large.

## Compute gradients by backpropagation {#compute-gradients}

A model with random weights predicts badly; training adjusts the weights to predict well. To measure how badly, use the **loss**: L = −log P(next token), the negative log probability the model gave to the token that actually came next, the same surprise [chapter 1](#score-with-perplexity) averaged into perplexity. Training lowers the average loss, and lower loss means lower perplexity. To lower it, each weight needs a **gradient**: how much the loss would change if that weight changed a little, written ∂L/∂w.

**Backpropagation** computes all the gradients at once, by applying the chain rule of calculus backwards through the forward pass: from the loss to the logits, from the logits through the output layer to the hidden layer, through tanh, and back to the embeddings that were used ([Rumelhart et al., 1986](https://doi.org/10.1038/323533a0)). Two of its steps are worth knowing by heart. The gradient of the loss with respect to the logits is the softmax probabilities minus 1 at the true token. And the gradient through tanh is multiplied by 1 − h², which is small where h is near ±1. The book's shared module `lib.tinynet` writes the whole of it out, line by line, for a tiny version of the store's model, with 5 tokens, embeddings of 2 numbers, a window of 2 and 3 hidden units.

Backpropagation code often goes subtly wrong, so check it against a slower method that can't: nudge one weight up and down by a tiny amount, measure how the loss moves, and compare. For the store's model that would mean more than a million nudges, so check every parameter of the tiny one:

[Example: gradients-by-hand](../examples/gradients-by-hand.py)

All {{out:gradients-by-hand:parameters checked}} gradients from backpropagation agree with the nudged estimates to six decimal places: the last line reads {{out:gradients-by-hand:backprop agrees with nudging to 6 decimal places}}. The nudging method costs two forward passes per parameter, which is why no one trains with it, but on a tiny model it's the reference that tells you backpropagation is right.

**Working rule:** Check hand-written gradients against nudged estimates on a tiny model before trusting them, as here. When you use a library that computes gradients for you, as the [PyTorch section](#train-with-pytorch) does, check the library's forward pass against your own instead.

## Take gradient descent steps {#take-gradient-descent-steps}

With gradients in hand, training is a loop. **Gradient descent** moves every weight a small step against its gradient, w ← w − η · ∂L/∂w, where η, the **step size**, also called the **learning rate**, sets how far each step goes. Each step makes the loss a little smaller for the examples it was computed on, and repeated over many examples, the weights settle where the model predicts well.

For the store's tiny model from the [previous section](#compute-gradients), take three (window, next token) examples, compute the average loss, and take twenty steps of gradient descent with a step size of 0.5, using `lib.tinynet`'s gradients:

[Example: gradient-descent-by-hand](../examples/gradient-descent-by-hand.py)

The average loss falls from {{out:gradient-descent-by-hand:average loss before}} before training to {{out:gradient-descent-by-hand:average loss after 1 steps}} after one step, {{out:gradient-descent-by-hand:average loss after 5 steps}} after five, and {{out:gradient-descent-by-hand:average loss after 20 steps}} after twenty. With only three examples to learn, the tiny model can drive its loss close to zero, which is memorizing; the store's model learns from hundreds of thousands of examples, where the same loop has to find patterns that hold across them.

Real training changes two things about this loop. It computes each step on a **batch** of examples, a few hundred at a time, instead of all of them, so each step is cheap and the steps vary a little. And it uses a refined update instead of a fixed step size, as the next section describes.

**Working rule:** Start any new model by running a few steps on a handful of examples and checking that the loss falls, as here: a loss that won't fall on three examples means a bug, not a hard problem. Choose the step size by trying a few: too small and the loss crawls, too large and it jumps around or grows.

## Train with PyTorch {#train-with-pytorch}

Writing backpropagation by hand for every model would be slow and error-prone. PyTorch, a widely used library for training neural networks, computes gradients automatically from the forward pass, and runs the arithmetic fast. The book trains its model with it, in `recordings/neural_lm.py`, which builds the same window model as the [first section](#predict-from-a-window): an embedding table, one tanh layer and an output layer, trained on the store's training names. The heart of the script is the loop of the [previous section](#take-gradient-descent-steps), on batches of 512 predictions:

```python
loss = loss_of(model(train_x[pick]), train_y[pick])   # forward pass and average loss on one batch
optimizer.zero_grad()                                  # clear the previous batch's gradients
loss.backward()                                        # backpropagation: a gradient for every weight
optimizer.step()                                       # move every weight against its gradient
```

The optimizer is **Adam**, a refinement of gradient descent ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980)). It keeps, for every weight, a running average of its gradients and a running average of their squares, and steps by the first divided by the square root of the second. A weight whose gradients are consistently large gets steps of about the same size as one whose gradients are small, so one step size works for all of them.

Training depends on the machine's floating-point arithmetic, which varies slightly between processors, so it runs once and the book's examples read the weights it saved, as they read every result that depends on the machine. The examples never import PyTorch: `lib.neural` recomputes the model's forward pass in NumPy, from those weights. That only works if the NumPy version computes what PyTorch computed, so the training script saved PyTorch's own logits for 64 validation contexts, and the next example compares:

[Example: same-as-pytorch](../examples/same-as-pytorch.py)

The NumPy forward pass reproduces all {{out:same-as-pytorch:logits compared}} of PyTorch's logits to four decimal places, for the model trained with PyTorch {{out:same-as-pytorch:PyTorch version}}. Every number this chapter reports about the trained model comes from that NumPy code, except the training curve, which the script recorded as it trained.

**Working rule:** Train with a library that computes gradients for you, and keep a short, independent forward pass of your own to check its outputs, as `lib.neural` does here. Record the library's outputs for a few inputs whenever you save weights, so the check can run anywhere, long after training.

## Stop when validation loss turns up {#stop-at-the-best-epoch}

A model with 1.4 million parameters, trained on a few hundred thousand tokens, can memorize its training text, as [chapter 3's 5-gram](#longer-contexts-fit-training) did. Training perplexity keeps falling as it does. Validation perplexity falls at first, while the model learns patterns that hold for new names, then rises, once it starts fitting quirks of the training names. That turn is **overfitting**, and the simplest defence is **early stopping**: measure validation loss after every **epoch**, one full pass over the training text, and keep the weights from the best epoch.

For the store, read the training script's record of both perplexities after each of its twelve epochs:

[Example: training-curve](../examples/training-curve.py)

In epoch 1, training perplexity is higher than validation, because the training figure averages over the whole epoch, including its first batches, when the model was still untrained, while validation is measured at the epoch's end. After that, training perplexity falls at every epoch, while validation perplexity bottoms out at {{out:training-curve:validation perplexity at the kept epoch}} in epoch {{out:training-curve:epoch kept}} and climbs from there, to {{out:training-curve:validation perplexity at the last epoch}} by the last epoch. The training script kept the weights from the best epoch, and every result in the rest of this chapter uses them.

The gap between the two curves is the model memorizing. It grows every epoch, because the training names repeat their product lines and phrasing, and this model has no **regularization**, nothing in its training that discourages memorizing, so early stopping is its only defence.

**Working rule:** Always record validation loss per epoch and keep the best epoch's weights, never the last. When the curves part early, as they do here after a few epochs, the model has more capacity than its training text supports: regularize it, use a smaller model, or get more text of the same kind.

## Compare with the baseline {#compare-with-the-baseline}

The real test is the one every model in this book takes: perplexity per token on the validation names, against the [4-gram baseline](#the-baseline). Before you run the comparison, predict which model wins. The neural model can relate similar contexts, which the 4-gram can't; the 4-gram has exact counts for every context it has seen.

[Example: neural-vs-baseline](../examples/neural-vs-baseline.py)

The window model's validation perplexity is {{out:neural-vs-baseline:window model validation perplexity}}, against the baseline's {{out:neural-vs-baseline:4-gram baseline validation perplexity}}. The counted model wins, by a wide margin.

To see where, split the validation tokens by whether their three-token context occurs in the training names, and score both models on each part:

[Example: seen-and-unseen-contexts](../examples/seen-and-unseen-contexts.py)

{{out:seen-and-unseen-contexts:validation tokens with a context seen in training}} of the validation tokens have a context the training names contain, and on those the 4-gram scores {{out:seen-and-unseen-contexts:4-gram perplexity, seen contexts}} and the network {{out:seen-and-unseen-contexts:network perplexity, seen contexts}}. On the unseen contexts, where the network's ability to relate contexts should help most, the 4-gram scores {{out:seen-and-unseen-contexts:4-gram perplexity, unseen contexts}} and the network {{out:seen-and-unseen-contexts:network perplexity, unseen contexts}}. The network is worse on both, and by more on the unseen ones.

So this network doesn't lose only where counts are exact; it hasn't learned to generalize well either. Two features of this model and this text are the likeliest reasons. The network has no regularization and overfits after a few epochs, as the [training curve](#stop-at-the-best-epoch) showed. And the store's names are short and formulaic: three tokens of context cover most of what a name holds, which is the setting where Kneser-Ney's counts are strongest. Bengio and colleagues found the opposite on larger, more varied text: on the Brown corpus, about 800,000 training words, their best network alone had a test perplexity of 268, and mixed with a trigram model 252, against the best n-gram model's 312 ([Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), section 4.2 and table 1). A learned model isn't better because it's learned; it has to be measured.

**Working rule:** Always compare a learned model with a strong counted baseline on the same text and tokens before claiming it helps, and split the comparison by seen and unseen contexts to see where each model wins. Don't conclude from one small dataset that neural models don't work, or that they always do.

## Mix the network with the counts {#mix-with-the-counts}

A model that's worse on average can still be useful, if its mistakes differ from the better model's. When two models make different mistakes, mixing their probabilities can beat both, which is what Bengio and colleagues did with their network and a smoothed trigram model ([Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), section 4). The mix is a weighted average of the two models' probabilities for each token, as in [chapter 3's interpolation](#mix-short-and-long-contexts).

For the store, score the validation names with the network's share of the mix at 0, 0.2, 0.3, 0.5 and 1, where 0 is the counted model alone and 1 is the network alone:

[Example: mix-neural-and-counts](../examples/mix-neural-and-counts.py)

The mix beats both models. With a network share of 0.2, the validation perplexity is {{out:mix-neural-and-counts:validation perplexity, network share 0.2}}, better than the baseline's {{out:mix-neural-and-counts:validation perplexity, network share 0.0}} and far better than the network's {{out:mix-neural-and-counts:validation perplexity, network share 1.0}} alone. The network is worse on average, but where the counts give a token very little probability, the network's share keeps it from being tiny, and that lowers the average surprise.

The kept epoch and the share were both chosen by looking at the validation names, so the gain is slightly optimistic; the book's final comparison, on the test names, will check it. And the mix only helps because the two models' errors differ: if they made the same mistakes, averaging them would gain nothing.

**Working rule:** When a learned model and a counted one are close, try mixing their probabilities before choosing one, and choose the share on held-out text. Keep the share small when one model is much stronger, as here, and confirm the gain on a separate test set before relying on it.

## More text helps only if it's the same kind {#same-kind-of-text}

If the network overfits because it learned from too little text, the obvious fix is more text. The store has plenty: its product descriptions. The training script can train the same model on names and descriptions together, and the book recorded that version too, scored on the same validation names.

For the store, compare the two recorded models on the validation names:

[Example: more-text-of-another-kind](../examples/more-text-of-another-kind.py)

The training names hold {{out:more-text-of-another-kind:training tokens, names}} tokens, and names and descriptions together {{out:more-text-of-another-kind:training tokens, names and descriptions}}, about eight times as many. The model that learned from names alone scores {{out:more-text-of-another-kind:learned from names}}, and the one that learned from names and descriptions scores {{out:more-text-of-another-kind:learned from all}}: worse, with eight times the text. That second model trained for four epochs, and its validation perplexity was still falling when it stopped, by smaller steps each epoch: {{out:more-text-of-another-kind:all-text model, validation perplexity by epoch}}. Our judgement is that more epochs wouldn't bring it near the names model.

Descriptions are a different kind of text. They're written in full sentences, with words such as `your`, `this` and `perfect` that names rarely contain, and a model that learns mostly from them predicts names as if they were descriptions. The added text taught the model the wrong distribution for the task. A model learns the text it's given, and its perplexity measures how well it predicts text like that: data of another kind can make it worse, and needs to be weighted or kept apart.

**Working rule:** Before adding training text, check that it resembles the text you'll predict; train on both and measure on the target text, as here, rather than assuming more is better. When the extra text differs a lot, train on the target text alone, or lay out each text so the model can tell its kinds apart, as [chapter 8](#turn-products-into-batches) does with a separator between a product's name and its description.

## Suggest from similar contexts {#suggest-from-similar-contexts}

The network lost on average, including on contexts the training names lack, but averages hide single cases. [Chapter 3's autocomplete](#autocomplete-with-more-context) showed the 4-gram's worst case, `outdoor bar`, a context no training name contains, where it fell back to generic continuations.

For the store, ask the network for its three likeliest next tokens after five prefixes, each treated as the start of a name, and compare a few of its embeddings:

[Example: neural-autocomplete](../examples/neural-autocomplete.py)

After `outdoor bar`, the network suggests `stool` first, with probability {{out:neural-autocomplete:P(stool | outdoor bar)}}, where the 4-gram put `&` and `with` ahead of it. The control prefixes show that `stool` isn't automatic after `bar`: after `bar` alone, `kitchen bar` and `wine bar`, it isn't among the network's three suggestions. And in the network's own embedding table, `patio` is a little closer to `outdoor`, with a cosine similarity of {{out:neural-autocomplete:cosine(patio, outdoor) in the model's embeddings}}, than to `kitchen` or `wine`. That's consistent with the network having carried something from `patio bar` over to `outdoor bar`, though the similarity is weak, and one prefix proves little.

This is one case, not an average: the [seen and unseen split](#compare-with-the-baseline) showed the network worse on unseen contexts overall. It shows what a learned model can do that counting can't, and that this one does it only sometimes.

**Working rule:** Look at a learned model's predictions on contexts the training text lacks, with control prefixes beside them, but judge it by measurements over many contexts, not by examples. For autocomplete in production, use the mix, and test it on rare prefixes as well as common ones.

## What goes wrong in production {#neural-in-production}

A neural language model brings new failure modes with its new strengths:

- **Overfitting is silent.** Training loss always looks good. Without a validation curve, a model trained too long ships worse than one trained less. Keep the best epoch, and watch the gap.
- **Results depend on the machine.** Retraining on another processor gives slightly different weights. Save the weights you ship, the code and settings that made them, and the library's outputs for a few inputs, as the training script here does.
- **The output layer dominates.** Scoring every token in the vocabulary at every step is most of the computation and most of the parameters. A larger vocabulary costs here first.
- **Mismatched training text hurts.** Adding text unlike the target, as descriptions were for names, makes predictions worse while the training loss looks fine. Measure on the target text after every change to the training data.
- **Averages hide rare contexts.** A model can score well overall and still fail on the unusual prefixes that shoppers type when they look for something new. Score seen and unseen contexts separately, as this chapter did, and watch the unseen ones.
- **A mix needs both models at serving time.** Mixing a network with counts doubles what has to be loaded and kept current. Rebuild the counts whenever you retrain the network, so the two describe the same text.

## What this chapter decided {#neural-decisions}

- **The book's neural models are trained in PyTorch, once, in `recordings/`,** and the examples recompute them in NumPy from the saved weights. The NumPy forward pass reproduces PyTorch's logits to four decimal places, so the examples need no PyTorch.
- **Models keep the weights from their best validation epoch.** The window model's validation perplexity was lowest after epoch {{out:training-curve:epoch kept}}, and rose every epoch after, while training perplexity kept falling: the model was memorizing the training names.
- **On the store's names, the window model loses to the 4-gram baseline alone, on seen and unseen contexts alike, and beats it mixed.** A network share of 0.2 gives {{out:mix-neural-and-counts:validation perplexity, network share 0.2}}, the best of the book's models so far.
- **Learned models are compared with the baseline on seen and unseen contexts separately,** not only on average, because the two parts can tell different stories. Here the network lost on both, and by more on the unseen ones.
- **Training text must resemble the target text.** Adding the descriptions made the model worse at names, though names and descriptions together held about eight times as many tokens as the names alone, so every model's training text is chosen for the text it will be scored on.
- **Names are short, so from [chapter 8](#training-a-small-model) the book also measures models on the store's descriptions.**
