---
chapter_id: choosing-the-next-token
---

# Choosing the next token

By the end of this chapter you can turn a language model's probabilities into text, and choose how: always the likeliest token, the likeliest whole text by beam search, or a sample shaped by a temperature, a top-k or top-p cut, a repetition penalty and a minimum length. You'll measure what each choice does to the text, and learn which failures no choice of setting can fix.

A language model gives a probability for every possible next token; it doesn't write. To write, something has to choose one token, append it, ask for the next probabilities, and repeat until the model predicts the end marker, `</s>`. That choosing is called **decoding**, and it decides much of what the text reads like, with the model unchanged.

The store, the online furniture and home-goods store this book follows, has a reason to write. Of its {{out:greedy-description:validation products}} validation products, {{out:greedy-description:validation products with no description}} have no description at all, and so do {{out:greedy-description:training products with no description}} of its training products. The store's model, the transformer that [chapter 8](#training-a-small-model) trained and measured, learned descriptions from the store's own text, so it can draft one from a product's name: start the text with the name and `<sep>`, as every training text was laid out, and let the model continue. This chapter does that with greedy decoding, beam search and sampling, using the book's shared module `lib.decoding`, which builds each method in NumPy on top of `lib.transformer`. Then it measures what each method writes against the store's real descriptions, and fixes two failures that the measurements expose: repetition, and descriptions that come out empty.

## Choose the likeliest token {#choose-the-likeliest-token}

The simplest decoder is **greedy**: at every step, take the single likeliest token. It needs no randomness, so it always writes the same text for the same start, and each step costs one [forward pass](#tie-the-output) of the model. The text it writes is, token by token, the model's best guess.

For the store, take a product with no description, `front porch welcome sign wall décor`, and decode greedily for up to {{out:greedy-description:longest description, in tokens}} tokens. Then, for a second product, `5 light light bulb`, print the probability that the description ends before it starts:

[Example: greedy-description](../examples/greedy-description.py)

The greedy description reads like the store's text, because it is the store's text, recombined: `this welcome sign is a great way to add a touch of rustic charm to your home`. Then it starts to loop. It writes `features a distressed finish with a distressed finish`, and later `has a distressed finish` again, and it uses two four-token phrases twice, `this welcome sign is` and `to your home .`, which the example lists. Greedy decoding loops because the same likeliest continuation wins every time the text reaches a similar point: `distressed` is the model's favourite finish for a sign, so it comes back whenever a finish is due. Holtzman and colleagues found the same in large models: decoding that always maximizes probability leads to text that is "bland, incoherent, or gets stuck in repetitive loops" ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)).

The description also says the sign is `made of natural wood`, which nothing in the product's name says, and the bulb shows a different failure. The model gives an immediate end, an empty description, a probability of {{out:greedy-description:P(the description is empty), 5 light light bulb}}. In the training text, {{out:greedy-description:training products named as bulbs with no description}} of the products named as bulbs have no description, against {{out:greedy-description:training products with no description}} of all products: the model learned the catalog as it is, gaps included, and a [later section](#forbid-an-empty-description) deals with it.

**Working rule:** Use greedy decoding when you want the same output every time and short text, such as a search-box suggestion, where the likeliest next word is the point. For anything longer than a phrase, expect it to repeat itself, and choose one of the methods in the [sections that follow](#search-for-the-likeliest-text).

## Search for the likeliest text {#search-for-the-likeliest-text}

[Greedy decoding](#choose-the-likeliest-token) picks the likeliest token at each step, which isn't the same as the likeliest text: a slightly less likely first word can lead to much likelier words after it. **Beam search** looks further. It keeps the k likeliest partial texts, the beam, scored by their **total log-probability**, the sum of the logarithms of their tokens' probabilities, where higher means likelier. At each step it extends every text in the beam by its k likeliest next tokens, and keeps the best k of the results. A text that reaches the end marker is finished and stays in the running. The heart of `lib.decoding.beam` is the line that scores each extension:

```python
            candidates += [(score + log_p[token], written + [int(token)]) for token in np.argsort(-log_p, kind="stable")[:width]]
```

For the store, use a beam of {{out:beam-search:beam width}} twice. First, complete a search query, `outdoor bar`, by up to {{out:beam-search:longest completion, in tokens, its end included}} tokens, the end of the name included, the way a search box might offer whole phrases. Then write the welcome sign's description, and print the likeliest text overall, and the likeliest per token, its total log-probability divided by its length. Before you run it, predict the welcome sign's likeliest description:

[Example: beam-search](../examples/beam-search.py)

For the query, beam search works as intended. Its top completions of `outdoor bar` are `stool`, then `stool cushion`, each with its probability: short, likely phrases, which is what a search box should offer. This uses the transformer alone; the mix that [chapter 8](#training-a-small-model) chose for autocomplete would need the 4-gram's probability for every candidate at every step, which `lib.decoding` doesn't compute.

For the description, the likeliest text overall is empty. Ending at once has a probability of {{out:beam-search:its probability}}, and every longer text multiplies many probabilities below one, so none comes close. Stahlberg and Byrne found the same in translation models: for most sentences the most probable translation was empty, and beam search worked only because it searched badly ([Stahlberg and Byrne, 2019](https://arxiv.org/abs/1908.10090)). Ranking by probability per token removes that bias, and gives a sensible description. It still invents a material, though: this time the sign is `crafted of iron`.

**Working rule:** Use beam search for short completions where the likeliest phrase is the goal, such as search queries, ranked by total probability. For longer text, rank finished texts per token, since total probability favours the shortest, and even then prefer the sampling methods of the [next sections](#sample-with-a-temperature), which write text closer to what people write.

## Sample, with a temperature {#sample-with-a-temperature}

Greedy decoding and beam search use only the top of the distribution. **Sampling**, which [chapter 1](#predicting-the-next-word) used to generate names, uses all of it: draw the next token at random, in proportion to its probability, so a token with probability 0.2 is chosen one time in five. Before sampling, the distribution can be made sharper or flatter with a **temperature**, T: divide every logit by T before the softmax. In symbols, with zᵢ the logit of token i, as in [chapter 5](#predict-from-a-window):

P(token i) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)

A temperature below 1 sharpens the distribution, giving more of the probability to the likeliest tokens; above 1, it flattens it, giving more to unlikely ones. A temperature near 0 approaches greedy decoding, and a temperature of 1 leaves the model's probabilities as they are. Dividing by T keeps the order of the tokens; it only changes how far apart their probabilities are.

For the store, stop the greedy description after its first few words, and look at the next token's distribution at three temperatures: the three likeliest tokens with their probabilities, how many tokens it takes to cover 90% of the probability, and how much probability falls outside the tokens that cover 90% at temperature 1:

[Example: temperature-by-hand](../examples/temperature-by-hand.py)

After `{{out:temperature-by-hand:text so far}}`, the model's own distribution, at temperature 1, gives its three likeliest tokens {{out:temperature-by-hand:temperature 1.0, top 3}}, and needs {{out:temperature-by-hand:temperature 1.0, tokens needed to cover 90% of the probability}} tokens to cover 90% of its probability. The remaining tenth is spread over the rest of the vocabulary's {{out:temperature-by-hand:tokens in the vocabulary}} tokens: a long tail, each token in it unlikely. At temperature 0.5 the three likeliest take {{out:temperature-by-hand:temperature 0.5, top 3}}, and {{out:temperature-by-hand:temperature 0.5, tokens needed to cover 90% of the probability}} tokens cover 90%. At 1.5 they shrink to {{out:temperature-by-hand:temperature 1.5, top 3}}; covering 90% takes {{out:temperature-by-hand:temperature 1.5, tokens needed to cover 90% of the probability}} tokens, and the tokens outside temperature 1's 90% now carry {{out:temperature-by-hand:temperature 1.5, probability outside temperature 1.0's 90% cover}} of the probability.

That tail is the risk in sampling. Most tokens in it are wrong here, and at temperature 1 a sampler draws from it about once in every ten tokens, more often as the temperature rises.

**Working rule:** Treat temperature as the dial between predictable and varied text. Keep it at or a little below 1 for text that should stay close to what the training text usually says, such as product descriptions. A lower temperature makes text more predictable, not more accurate.

## Sample, and cut the tail {#sample-and-cut-the-tail}

Lowering the temperature shrinks the tail but never removes it. Two methods remove it directly, before sampling. **Top-k** keeps only the k likeliest tokens and renormalizes their probabilities to sum to one ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)). **Top-p**, also called nucleus sampling, keeps the smallest set of likeliest tokens whose probabilities reach at least p ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751), equation 2). Both are options of `lib.decoding.truncate`, whose core is three steps: sort the tokens, add up their probabilities in that order, and find where the sum reaches p:

```python
    order = np.argsort(-probabilities, kind="stable")
    ...
        keep = int(np.searchsorted(np.cumsum(probabilities[order]), top_p)) + 1
```

Top-p adapts to the model's confidence, where top-k doesn't. At the point in the [previous section's example](#sample-with-a-temperature) where the model was unsure of the next word, it took {{out:temperature-by-hand:temperature 1.0, tokens needed to cover 90% of the probability}} tokens to cover 90% of the probability, so top-p 0.9 would keep that many, while top-k 20 would keep only 20. After a word that the model predicts with near certainty, top-p keeps one or two tokens and top-k still keeps 20, most of them wrong. Holtzman and colleagues argued for top-p on exactly this ground: a fixed k is "sub-optimal across varying contexts".

For the store, write {{out:sample-descriptions:longest description, in tokens}} tokens of the welcome sign's description four ways, with the same random seed: sampling at temperature 1, sampling at 0.7, top-k with k = 20, and top-p with p = 0.9. Then replay the temperature-1 sample, and list the tokens it drew from outside each step's 90% nucleus, the tokens top-p 0.9 would have cut:

[Example: sample-descriptions](../examples/sample-descriptions.py)

Sampling at temperature 1 wanders: it writes `welcome welcome sign sign`, invents a word, `glywheel`, and claims the sign is `made in the usa`. Only some of that comes from the tail: {{out:sample-descriptions:temperature 1.0, tokens drawn from outside the 90% nucleus}} tokens did, {{out:sample-descriptions:temperature 1.0, those tokens}}. The doubled words and the made-up word were built from tokens inside the nucleus, which top-p would have kept. Cutting the tail removes the rarest mistakes, not all of them. At 0.7 the text holds together better, though it now says the sign is `made of high-quality metal`, where greedy decoding said wood. Top-k and top-p read more like the catalog than plain sampling, and still invent: top-p's sign is `cast iron`, with a `podishade` design.

Across this chapter, the welcome sign has been given a material six times, by greedy decoding, beam search, three of these samples and the [penalized greedy text](#penalize-repetition) below, and nothing in its name says what it's made of. The model writes what descriptions of welcome signs usually say, and they usually state a material. A decoding method changes which plausible material the model picks; none of them can make it pick the true one.

**Working rule:** For varied text, cut the tail with top-p around 0.9 or top-k around 20, with a temperature at or a little below 1, and fix the random seed when you need to reproduce a result. Don't expect any setting to make the text true: decoding chooses among what the model finds plausible, and checking facts is a separate job.

## Measure what each setting writes {#measure-each-setting}

Single samples show the flavour of each method; measurements show the trade-offs. Several properties of generated text can be measured without a person reading it:

- **repetition:** the share of four-token phrases that repeat an earlier one in the same description
- **the model's own [perplexity](#score-with-perplexity) of the text:** how surprising the model finds what it wrote, pooled over every token, where low means predictable
- **whether the description ends** within the limit, as real ones do
- **whether it uses a whole word of the product's name**, as a description of that product should
- **its average length** in tokens

For the store, draw {{out:measure-decoding:products, drawn at random from the validation products with a description}} validation products at random from those with a description, write a description for each with every method, and compare them with the products' real descriptions, cut at the same {{out:measure-decoding:longest description, in tokens}}-token limit:

[Example: measure-decoding](../examples/measure-decoding.py)

The real descriptions repeat nothing ({{out:measure-decoding:real descriptions, repeated 4-token phrases}}), and the model finds them fairly surprising: its perplexity of them is {{out:measure-decoding:real descriptions, the model's perplexity of the text}}, higher than [chapter 8](#training-a-small-model)'s figure for descriptions because these are a different sample and only their first {{out:measure-decoding:longest description, in tokens}} tokens. Greedy text is the opposite: {{out:measure-decoding:greedy, repeated 4-token phrases}} of its phrases repeat, and its perplexity is {{out:measure-decoding:greedy, the model's perplexity of the text}}, far more predictable than anything a person wrote. [Holtzman and colleagues](https://arxiv.org/abs/1904.09751) made the same observation at scale: human text isn't the text a model finds most probable.

No setting matches the real descriptions on every measure. Sampling at temperature 1 comes closest on perplexity, {{out:measure-decoding:sampling, temperature 1.0, the model's perplexity of the text}}, but only {{out:measure-decoding:sampling, temperature 1.0, uses a whole word of the name}} of its descriptions use a word of the product's name, against {{out:measure-decoding:real descriptions, uses a whole word of the name}} of the real ones: it drifts off the product. Top-p 0.9 ends as often as the real text, {{out:measure-decoding:top-p 0.9, ended within 60 tokens}} against {{out:measure-decoding:real descriptions, ended within 60 tokens}}, with a perplexity of {{out:measure-decoding:top-p 0.9, the model's perplexity of the text}}, but stays on the product in only {{out:measure-decoding:top-p 0.9, uses a whole word of the name}}. Top-k 20 stays on the product most among the samplers, {{out:measure-decoding:top-k 20, uses a whole word of the name}}, with no repetition, a perplexity of {{out:measure-decoding:top-k 20, the model's perplexity of the text}} and descriptions of {{out:measure-decoding:top-k 20, average length in tokens}} tokens on average. Twenty products is a small sample, so read these as directions, not exact rates.

What the measures can't say is whether a description is true. A description can be fluent, varied, on topic and wrong, and [the previous section's](#sample-and-cut-the-tail) were, about the sign's material.

**Working rule:** Compare decoding settings on measures like these, against real text of the same kind, before reading samples. Decide which measure matters most for the job: for product descriptions, staying on the product and not repeating come first. Then read a sample of the outputs anyway, for what no measure catches, and check every factual claim against the product's data.

## Penalize repetition {#penalize-repetition}

[Greedy decoding's](#choose-the-likeliest-token) loops can also be fixed without sampling. Keskar and colleagues proposed a **repetition penalty**: divide the logit of every token already generated by a factor θ above 1, so it becomes less likely ([Keskar et al., 2019](https://arxiv.org/abs/1909.05858), section 4.1). Dividing a negative logit would raise it instead, so implementations multiply negative logits by θ, as `lib.decoding.penalize` does, so the token becomes less likely either way. Keskar and colleagues found a factor around 1.2 balanced avoiding repetition against writing sensibly.

For the store, go back to the point where the [greedy description](#choose-the-likeliest-token) was about to write `distressed` a second time. Apply the penalty by hand to the next token's logits, check it against `lib.decoding.penalize`, and compare the probabilities before and after. Then write the whole description greedily with the penalty:

[Example: repetition-penalty-by-hand](../examples/repetition-penalty-by-hand.py)

With a factor of {{out:repetition-penalty-by-hand:penalty factor}}, the hand calculation matches the library's ({{out:repetition-penalty-by-hand:same as lib.decoding.penalize}}). Without the penalty, `distressed` has probability {{out:repetition-penalty-by-hand:P(distressed), without a penalty}} and wins; with it, {{out:repetition-penalty-by-hand:P(distressed), with the penalty}}, and another token wins. The penalized greedy description doesn't loop. It does still invent: this time the sign is made of `natural coir`, a fibre from coconut husks.

The penalty has a side effect the [measurements](#measure-each-setting) show. It penalizes every token already used, including common words such as `the` and `and` that a description needs many times, but never the end marker, which hasn't been used yet. So the end becomes relatively likelier, and descriptions end sooner: {{out:measure-decoding:greedy, repetition penalty 1.3, ended within 60 tokens}} of the penalized greedy descriptions ended within the limit, against {{out:measure-decoding:greedy, ended within 60 tokens}} without the penalty, with an average length of {{out:measure-decoding:greedy, repetition penalty 1.3, average length in tokens}} tokens against {{out:measure-decoding:greedy, average length in tokens}}. Its repetition fell to {{out:measure-decoding:greedy, repetition penalty 1.3, repeated 4-token phrases}}.

**Working rule:** Add a repetition penalty around 1.2 to 1.3 when you decode greedily or at a low temperature, and check what it does to length, since it shortens text. As a rule of thumb, if descriptions come out clipped, leave common function words and punctuation out of the penalty.

## Forbid an empty description {#forbid-an-empty-description}

The [greedy example](#choose-the-likeliest-token) found the model ready to write nothing for a bulb, and [beam search](#search-for-the-likeliest-text) found nothing to be the likeliest description for a sign. Both come from the catalog: many products have no description, so the model learned that ending at once is common. For a store that wants to fill exactly those gaps, the model has to be stopped from reproducing them. The simple fix is a **minimum length**: forbid the end marker, by setting its logit to minus infinity, until the description has some number of tokens. `lib.decoding.generate` takes it as `shortest`.

For the store, take the bulb, `5 light light bulb`, and decode greedily without a minimum and with one of {{out:minimum-length:minimum length, in tokens}} tokens:

[Example: minimum-length](../examples/minimum-length.py)

The model gives the empty description a probability of {{out:minimum-length:P(the description is empty)}}, and greedy decoding takes it: the description comes out empty. With the minimum, it writes one, and two failures from earlier in this chapter appear in it. It repeats itself, `a light , a light , and more light`, and it states things the name doesn't: that the bulb is `made of metal`, `dimmable`, and `compatible with most standard bulbs`. The minimum stopped the gap; it can't supply the facts.

**Working rule:** Set a minimum length whenever the training text includes empty or very short examples of what you're generating. Combine it with the other settings in this chapter, not instead of them, and expect the text it forces out to need the same checking as any other.

## What goes wrong in production {#decoding-in-production}

Decoding is where a model's text meets its readers, and most of its failures are quiet:

- **Fluent text states made-up facts.** Six descriptions in this chapter gave the welcome sign a material, and they disagreed. Never publish generated product text without checking each factual claim against the product's data; [chapter 11](#from-predicting-to-following) measures how often a model invents.
- **Gaps in the training text become outputs.** The model predicts an empty description for a bulb, because the catalog has many, and beam search finds the empty description likeliest of all. Forbid the end marker until a [minimum length](#forbid-an-empty-description), or the model will reproduce the gaps you're trying to fill.
- **Greedy text loops.** Use sampling with a cut tail, or a repetition penalty, for anything longer than a phrase, and measure repetition on a sample of outputs.
- **Sampled text isn't reproducible by default.** Fix and record the random seed with every generated text, so a complaint about one description can be traced and repeated.
- **Settings drift with the model.** A retrained model has a different distribution, so a temperature or top-k tuned on the last model can write differently with the next. Rerun the measurements of this chapter after every retrain, before the new model writes anything a shopper sees.
- **Each token costs a forward pass.** `lib.decoding` reruns the model over the whole text for every new token, which is simple and fine for {{out:measure-decoding:longest description, in tokens}} tokens. Serving systems keep every layer's keys and values from earlier positions, so each new token's matrix multiplications cover one position, though attention still reads the cached keys and values of every earlier one ([Pope et al., 2022](https://arxiv.org/abs/2211.05102)).

## What this chapter decided {#decoding-decisions}

- **Autocomplete uses the likeliest tokens.** Search-box suggestions show the top few tokens of the mix chosen in [chapter 8](#training-a-small-model), and beam search, ranked by total probability, can extend them to short phrases.
- **Descriptions use top-k 20 sampling,** the setting that stayed on the product most often without repeating, on the measures above, with a fixed, recorded seed and a minimum length so they can't come out empty. With only twenty products behind it, the choice is rechecked on a larger sample before any description reaches a shopper.
- **Beam search ranks longer texts per token,** because total probability favours the empty text. Short completions of search queries are ranked by total probability, where the likeliest phrase is what a shopper wants.
- **Decoding settings are chosen by measurement against real text:** repetition, the model's own perplexity, whether descriptions end, whether they use the name, and their length, over a random set of products, before samples are read, and again after every retrain.
- **Generated descriptions are drafts, never a source of product facts.** A person or a checking step corrects every claim that the name and the catalog can't support: every method here invented a material for a product whose name gives none.
