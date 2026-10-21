---
chapter_id: words-as-vectors
---

# Words as vectors

By the end of this chapter you can turn every token of the store's text into a short vector of numbers, so that tokens used in similar ways get similar vectors, by two methods: compressing counts of which tokens appear together, and training a small model to predict each token's neighbours. You'll measure what those vectors know, where they fail, and how they let a model relate text it has never seen to text it has.

[Chapter 3](#counting-further-back) ended on a limit that counting can't get past. Its 4-gram model had seen `patio bar stool` many times, and suggested `stool` after `patio bar`. After `outdoor bar`, which no training name contains, it fell back to generic suggestions, because a count of one context tells it nothing about another. A shopper, or any reader, knows that `outdoor` and `patio` mean nearly the same thing to a furniture store, and a model needs some way to know it too. This chapter gives every token a vector, a list of numbers, and arranges the vectors so that tokens used alike sit close together.

The store, the online furniture and home-goods store this book follows, has the text to learn this from: the names and descriptions of its catalog. Its training split, the 80% of products that models learn from, holds {{out:cooccurrence-counts:training tokens}} tokens of it. Every vector in this chapter comes from that text alone, through the store's tokenizer from [chapter 2](#tokens).

## One-hot vectors know nothing {#one-hot-vectors-know-nothing}

Every model so far has treated tokens as names: `patio` is one entry in a table and `outdoor` another, with nothing linking them. Written as vectors, that's a **one-hot** encoding: each token gets a vector as long as the vocabulary, with a 1 in its own position and 0 everywhere else. To measure how alike two vectors are, this book uses **cosine similarity**: the cosine of the angle between them, 1 when they point the same way, 0 when they're at right angles. It ignores how long the vectors are and compares only their directions.

For the store, give the 2,000 most frequent tokens of its training text one-hot vectors, and compare three pairs. Keeping only the most frequent tokens, about half of the store tokenizer's vocabulary, holds every matrix in this chapter small enough to compute in seconds:

[Example: one-hot-vectors](../examples/one-hot-vectors.py)

Each vector has {{out:one-hot-vectors:numbers in each vector}} numbers. The cosine similarity of `patio` and `outdoor` is {{out:one-hot-vectors:cosine(patio, outdoor)}}, and so is that of `patio` and `sofa`: a one-hot encoding says the two outdoor words are exactly as unrelated as a patio and a sofa. Only a token compared with itself scores {{out:one-hot-vectors:cosine(patio, patio)}}. The 2,000 tokens that get vectors make up {{out:one-hot-vectors:share of training tokens with a vector}} of the training text; the rest are rarer, and this chapter leaves them out. Every pair of different tokens sits at right angles to every other, so nothing learned about one can carry over to another.

That's the same flaw as [chapter 3](#counting-further-back)'s counts, in another form. A model that reads one-hot vectors has to learn about every token separately, from the contexts that token itself appears in. Rare tokens get little, and unseen combinations get nothing.

**Working rule:** Don't feed one-hot vectors to a model that needs to relate tokens to each other. Use them only as an index, as the next sections do, to look up a learned vector for each token.

## Describe a token by the tokens around it {#describe-by-neighbours}

The way out is to describe a word by the company it keeps. Two words that appear among the same neighbours, such as `patio` and `outdoor`, which both turn up near `furniture`, `chair` and `umbrella`, are likely to mean similar things, even if they rarely appear together. So describe each token by its neighbours: count, for every pair of tokens, how often one appears within a small **window** of the other, here two tokens either side, and never across the gap between two products' texts. Each token's row of counts is then a vector of what it appears near.

For the store, count co-occurrences in the training split's names and descriptions, and list what appears most often near `patio`:

[Example: cooccurrence-counts](../examples/cooccurrence-counts.py)

Within two places of `patio`, the training text holds {{out:cooccurrence-counts:tokens near patio, counting repeats}} tokens, counting a token again each time it appears there. The most frequent of them are disappointing: a comma, `or`, `with`, `your` and a full stop, before the first word with anything to do with patios, `furniture`. Those tokens appear near almost everything, because they're everywhere. A row of raw counts mostly says how common each neighbour is, not what's special about the token whose row it is.

So raw counts make poor vectors. Two tokens that both appear often will look alike because they share the same frequent neighbours, commas and `with`, whatever they mean. What matters is which neighbours a token has more often than you'd expect from how common those neighbours are.

**Working rule:** Count co-occurrence within a small window, two to five tokens, in text the model will learn from, and never across document boundaries. Don't use the raw counts as vectors: reweight them, as the next section does, so frequent neighbours stop dominating.

## Weigh co-occurrence by surprise {#weigh-by-surprise}

The fix compares each count with what chance would predict. If `patio` and `furniture` appeared independently of each other, they'd co-occur about as often as their separate frequencies imply. **Pointwise mutual information**, or PMI, measures how far the real count departs from that: the log of the ratio between how often two tokens co-occur and how often they would by chance ([Church and Hanks, 1990](https://aclanthology.org/J90-1003/)). In symbols, for tokens a and b:

PMI(a, b) = log [ P(a, b) / (P(a) · P(b)) ]

where P(a, b) is the share of all co-occurrence counts that are of a and b together, and P(a) and P(b) are each token's share. PMI is positive when two tokens appear together more than chance predicts, and negative when less. Negative values are unreliable, since "less than chance" usually means "rarely seen at all", so **positive PMI**, or PPMI, replaces them with zero. The book's shared module `lib.vectors` computes it from the co-occurrence counts of the [previous section](#describe-by-neighbours), in three lines: the count chance would predict for each pair, the log of the real count over it, and zero wherever that's negative or the pair never occurs:

```python
expected = counts.sum(1, keepdims=True) * counts.sum(0, keepdims=True) / total
pmi = np.log(counts / expected)
return np.where(counts > 0, np.maximum(pmi, 0), 0.0)
```

For the store, reweight the counts with PPMI and list `patio`'s strongest neighbours again:

[Example: ppmi-neighbours](../examples/ppmi-neighbours.py)

The commas are gone. `patio`'s strongest neighbours by PPMI are `porch`, `deck`, `backyard`, `sectional`, `furniture` and `garden`: words a person would associate with a patio, ranked by how much more often than chance they appear near it. And {{out:ppmi-neighbours:share of PPMI entries that are zero}} of the PPMI matrix is zero, because most pairs of tokens appear together no more often than chance.

PPMI vectors are already useful: two tokens with similar PPMI rows have similar distinctive neighbours. But each vector is still 2,000 numbers long, mostly zeros, and two tokens with related but not identical neighbours, say `deck` and `porch`, can share few nonzero entries.

**Working rule:** Reweight co-occurrence counts with positive PMI before using them as vectors. Expect PMI to overrate pairs of rare tokens, whose few co-occurrences look far above chance; that's one reason to keep only frequent tokens, as this chapter does, or to require a minimum count.

## Compress counts into dense vectors {#compress-into-dense-vectors}

A [PPMI vector](#weigh-by-surprise) of 2,000 numbers, mostly zeros, has a lot of room and little information. Compressing it into a short, dense vector keeps the strongest patterns of co-occurrence and drops the noise, and in the process puts tokens with related neighbours close together even when they share few exact neighbours. The tool is the **singular value decomposition**, SVD, which finds the 64 directions that together reconstruct the PPMI rows best, and weights each by how much of the rows it explains. A token's 64 weighted coordinates along those directions are its new vector: an **embedding**, a dense vector that stands for a token. The method `lib.vectors.svd_vectors` computes them, scales each to unit length, and `lib.vectors.nearest` finds the tokens with the highest cosine similarity to any vector.

For the store, compute the 64-number embeddings and list each of six tokens' nearest neighbours:

[Example: svd-neighbours](../examples/svd-neighbours.py)

Every token now has {{out:svd-neighbours:dimensions per token}} numbers, and the neighbours read like a thesaurus built for a furniture store. `patio` and `outdoor` are each other's nearest neighbour, with a cosine similarity of {{out:svd-neighbours:cosine(patio, outdoor)}}, where their one-hot vectors scored zero. `sofa` sits beside `sectional`, `chaise` and `loveseat`; `oak` beside `walnut` and `espresso`, a wood finish; `blue` beside other colours; and `48` beside other sizes in inches.

None of that was taught. The method never saw a definition of `sofa`, a list of colours or a hint that `48` is a number. It found that these tokens appear among the same kinds of neighbours in the store's text, and the SVD put them close together because of it. That's the property [chapter 3](#counting-further-back)'s counts lacked: whatever a model learns about `patio bar stool` can now carry over, through the embeddings, to `outdoor bar stool`.

**Working rule:** Compress PPMI with SVD to between 50 and 300 dimensions, and check the result the way this section does: list a handful of tokens' nearest neighbours and read them. If the neighbours are mostly frequent function words, the reweighting went wrong; if they're unrelated, the window or the vocabulary is too small.

## Learn vectors by predicting neighbours {#learn-by-predicting-neighbours}

The other way to get embeddings is to learn them by prediction, which is how every neural language model in this book gets its own. **Skip-gram with negative sampling**, the method behind word2vec, gives each token two vectors, one for when it's the centre token and one for when it's a neighbour, and trains them on one task: given a token, tell its real neighbours apart from random tokens ([Mikolov et al., 2013a](https://arxiv.org/abs/1301.3781); [Mikolov et al., 2013b](https://arxiv.org/abs/1310.4546)). For a token and one of its real neighbours, the model scores the pair by the dot product of their vectors, turns the score into a probability with the logistic function, and nudges the vectors so the probability rises. For a few random **noise** tokens drawn from the vocabulary, it nudges the probability down. Repeated over millions of pairs, tokens with the same neighbours end up with similar vectors. That's no coincidence: Levy and Goldberg showed that skip-gram with negative sampling, trained this way, in effect factorizes a PMI matrix, shifted down by the log of the number of noise tokens, much as the [SVD](#compress-into-dense-vectors) factorizes PPMI ([Levy and Goldberg, 2014](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html)).

Write v for the centre token's vector, u for the real neighbour's, and uₙ for each noise token's, and σ for the logistic function, σ(s) = 1 / (1 + e⁻ˢ), which turns a score into a probability. The loss for one pair is the surprise at the real neighbour plus the surprise at each noise token's not being one:

L = −log σ(v·u) − Σₙ log σ(−v·uₙ)

and `lib.skipgram.step` lowers it with three lines of gradient descent, where `real` is σ(v·u), `fake` holds σ(v·uₙ) for each noise token, and `rate` is the step size:

```python
np.add.at(inputs, centre, -rate * grad_v)                               # move v: towards u, away from each uₙ
np.add.at(outputs, context, -rate * (real - 1)[:, None] * v)            # move u towards v
np.add.at(outputs, noise, -rate * fake[:, :, None] * v[:, None, :])     # move each uₙ away from v
```

For the store, take one real pair, `patio` and `umbrella`, and two noise tokens, `sofa` and `48`. Start the centre vectors at small random values and the neighbour vectors at zero, as `lib.skipgram.train` does, and apply three updates:

[Example: skipgram-step](../examples/skipgram-step.py)

Before training, the model gives every pair a probability of {{out:skipgram-step:P(umbrella is a real neighbour of patio), before}} of being real: the neighbour vectors are zero, so every score is zero, and σ(0) is one half. The model knows nothing. After three updates, `umbrella` scores {{out:skipgram-step:P(umbrella is a real neighbour of patio), after 3 steps}} as a neighbour of `patio`, and `sofa` and `48` have dropped to {{out:skipgram-step:P(sofa, 48 are real neighbours), after 3 steps}}. That's all training does, millions of times over: pull real neighbours together and push random tokens apart.

**Working rule:** When you train embeddings, start the second set of vectors at zero and the first at small random values, as `lib.skipgram.train` does, and draw noise tokens in proportion to their frequency raised to the power 0.75, which Mikolov et al. found works better than plain frequency. Check one update by hand, as here, before running millions: a probability moving the wrong way means the gradient's sign is wrong.

## Compare learned vectors with counted ones {#compare-learned-and-counted}

Training skip-gram on all the store's training text runs millions of updates, and the exact result depends on the machine's floating-point arithmetic, which varies slightly between processors. So it runs once, in `recordings/skipgram.py`, with the book's own code and a fixed seed, and the examples read its saved vectors, as the book does with every result that depends on the machine. Its manifest records the settings: a window of two tokens, 64 dimensions, five noise tokens per pair and two passes over the text. One setting is fixed in `lib.skipgram` rather than recorded: pairs centred on very frequent tokens, such as commas, are mostly skipped, with a threshold of 10⁻⁴, which keeps them from dominating training ([Mikolov et al., 2013b](https://arxiv.org/abs/1310.4546), who suggest around 10⁻⁵ for larger text).

For the store, read the recorded vectors and list the same six tokens' neighbours as the [SVD's](#compress-into-dense-vectors):

[Example: skipgram-neighbours](../examples/skipgram-neighbours.py)

The training loss fell from {{out:skipgram-neighbours:training loss, first batch}} on the first batch to {{out:skipgram-neighbours:training loss, last 100 batches}} on the last hundred: the model learned to tell real neighbours from noise. Its neighbours agree with the SVD's on most tokens, `sofa` with `sectional` and `chaise`, `oak` with `walnut` and `espresso`, colours with colours, sizes with sizes, and differ in emphasis on others: skip-gram puts `outdoor` closest to `indoor`, because the two words appear in the same positions in descriptions such as `indoor or outdoor use`.

That difference is worth noticing. Vectors built from neighbours put together words used in the same places, and opposites often are: `indoor` and `outdoor` here, and, in the [next section](#analogies-and-where-they-fail), `small`, whose nearest neighbours include `larger` as well as `smaller`. Similar vectors mean similar use, not similar meaning.

**Working rule:** Treat counted embeddings (PPMI and SVD) and learned ones (skip-gram) as close alternatives: on a few million tokens they give similar neighbours, and counted ones are deterministic and fast. Prefer learned embeddings when they're part of a larger model you're training anyway, as in every later chapter of this book. Never read a high similarity as synonymy without checking for opposites.

## Analogies, and where they fail {#analogies-and-where-they-fail}

Embeddings famously support analogies by arithmetic: if the difference between `chair` and `chairs` is a direction that means "plural", then adding it to `table` should land near `tables`. In vectors, the answer to "a is to b as c is to what?" is the token nearest to b − a + c, leaving out the three tokens in the question ([Mikolov et al., 2013a](https://arxiv.org/abs/1301.3781)). It's a neat test of whether the vectors have learned consistent relationships, and a good place to see their limits.

For the store, try three analogies with the [skip-gram vectors](#compare-learned-and-counted), and list the nearest neighbours of three tokens they raise:

[Example: analogies](../examples/analogies.py)

The first works: `chair` is to `chairs` as `table` is to `tables`, the top answer. The second fails, and the reason is in the last lines of the output: `lamps` isn't among the 2,000 tokens that have vectors, so the right answer can't be found at all, and the arithmetic returns whatever is nearest. The third fails differently. `light` is in the vocabulary, but `black` is to `white` as `dark` is to `brown`, by the vectors' reckoning. The neighbour lists show why: `light` sits with `lights`, `bulb`, `dimmable` and `candle`, because in the store's text it's mostly a lamp, as in `ceiling light`, not a shade, while `dark` sits with `brown`, `walnut` and `espresso`, the finishes it describes. And `small` sits with `larger` as well as `smaller`: size words appear in the same places, whichever size they name.

Analogies are a demonstration, not a measure. They work for relationships the text expresses consistently and often, such as plurals of common words, and fail for rare words, for words with several senses, and for relationships the text rarely spells out. A model doesn't use analogies; it uses the vectors' positions, and those carry the same strengths and blind spots.

**Working rule:** Don't judge embeddings by analogies; judge them by how much they help the task they're for, as the next section and every later chapter do. When an analogy fails, check whether the answer is in the vocabulary and whether the words mean what you assume in your text.

## Find similar names with vectors {#find-similar-names}

Embeddings for tokens give, with one more step, embeddings for whole texts: average the vectors of a text's tokens, scale the result to unit length, and similar texts get similar vectors. That's a rough version of what search engines call semantic retrieval, finding texts by meaning rather than by matching words, which this book's companion, *A Primer on Search Relevance*, builds properly with a trained model ([semantic retrieval](https://rollingedition.com/primer-on-search-relevance/semantic-retrieval)). Here it tests the one thing this chapter set out to fix: [chapter 3's](#autocomplete-with-more-context) counted model can't relate `outdoor bar` to `patio bar`.

For the store, average the skip-gram vectors of every training name, and find the names closest to two queries, one of them that problem prefix, `outdoor bar`:

[Example: name-vectors](../examples/name-vectors.py)

The example prints a rare word, most often a product line's name, as `[name]`, as [chapter 1](#load-the-store-text) explained. For `outdoor bar`, the closest names include two `patio bar stool` names, neither of which contains the word `outdoor`. [Chapter 3](#counting-further-back)'s 4-gram model couldn't connect the two, because it had counts for `patio bar stool` and none for `outdoor bar`. The averaged vectors connect them through `outdoor` and `patio`, whose vectors are close. For `patio umbrella`, the closest names are patio and outdoor products, and the top one is an umbrella cover, but none of them is an umbrella itself: averaging lets `patio` outweigh `umbrella`.

Averaging is crude: it forgets word order, so `bar stool` and `stool bar` get the same vector, and it lets frequent tokens drown out the one that matters. But it shows the property every later model in this book is built on: a model that reads embeddings instead of one-hot tokens can carry what it learned about one phrase over to another.

**Working rule:** Average token embeddings when you need a quick, order-free representation of short texts, and expect it to find related items but rank them loosely. For real semantic retrieval, use a model trained for it, as the Primer does. For language modeling, give the model the embeddings themselves and let it learn how to combine them, which [chapter 5](#a-neural-language-model) starts doing.

## What goes wrong in production {#vectors-in-production}

Embeddings carry their training text's habits into everything that uses them:

- **Similar use isn't similar meaning.** `indoor` and `outdoor` sit close together. A system that treats close vectors as interchangeable will happily swap opposites; check the nearest neighbours of the words your product depends on.
- **Rare and new tokens get poor vectors, or none.** This chapter kept 2,000 tokens. A token outside that list has no vector, and a token seen a handful of times has a noisy one. The store's tokenizer has more than twice that many tokens, and a rare word that the tokenizer splits into pieces gets a vector only from those of its pieces that are among the 2,000; a model that learns its own embeddings, as every later chapter's does, covers the whole vocabulary.
- **Vectors reflect the text they came from.** The store's text describes `light` as a lamp, so its vector sits with lighting. Embeddings trained on one domain carry its senses, and its biases, into another.
- **Results depend on the machine.** Training embeddings by gradient descent gives slightly different vectors on different processors, which is why this book records them. Store the vectors you ship, with the code and settings that made them.
- **Embeddings go stale.** New product lines and new words appear in the catalog; retrain or extend embeddings when the vocabulary drifts, and check neighbour lists for tokens that matter after each retrain.

## What this chapter decided {#vectors-decisions}

- **Tokens are compared by the cosine similarity of their vectors.** It ignores length and compares direction, and it's what `lib.vectors.nearest` returns. Every vector is scaled to unit length first, so the cosine is a plain dot product and frequent tokens, whose raw vectors are longer, don't win by size.
- **Counted embeddings come from positive PMI compressed with SVD,** over a window of two tokens and the 2,000 most frequent tokens, in `lib.vectors`. On the store's text they put `patio` next to `outdoor`, which one-hot vectors couldn't.
- **Learned embeddings come from skip-gram with negative sampling,** trained once in `recordings/skipgram.py` with the book's own code and a fixed seed, because their exact values depend on the machine. They agree with the counted ones on most tokens.
- **Similar vectors mean similar use.** Opposites such as `indoor` and `outdoor` get close vectors, and analogies work only for relationships the text expresses often, so the book judges embeddings by how much they help a task, never by analogies.
- **Every later model learns its own embeddings,** as part of predicting the next token, starting with [chapter 5](#a-neural-language-model)'s neural language model. This chapter's vectors are the evidence that embeddings carry knowledge from one context to another; the models in the rest of the book put that to work.
