---
chapter_id: choosing-the-next-token
---

# Choosing the next token

By the end of this chapter you can turn a language model's probabilities into text, and choose how: always the likeliest token, the likeliest whole text by beam search, or a sample shaped by a temperature, a top-k or top-p cut, a repetition penalty and a minimum length. You'll measure what each choice does to the text, and learn which failures no choice of setting can fix.

A language model gives a probability for every possible next token; it doesn't write. To write, something has to choose one token, append it, ask for the next probabilities, and repeat until the model predicts the end marker, `</s>`. That choosing is called **decoding**, and it decides much of what the text reads like, with the model unchanged.

The store this book follows, which sells furniture and home goods online, has a reason to write: {{out:greedy-description:validation products with no description}} of its {{out:greedy-description:validation products}} validation products have no description, and so do {{out:greedy-description:training products with no description}} of its training products. The store's model, the transformer that [chapter 8](#training-a-small-model) trained and measured, learned descriptions from the store's own text, so it can draft one from a product's name: start the text with the name and `<sep>`, as every training text was laid out, and let the model continue. This chapter does that with greedy decoding, beam search and sampling, using the book's shared module `lib.decoding`, built in NumPy on `lib.transformer`. Then it measures what each method writes against the store's real descriptions, and fixes two failures that the measurements expose: repetition, and descriptions that come out empty.

## Choose the likeliest token {#choose-the-likeliest-token}

The simplest decoder is **greedy**: at every step, take the single likeliest token. It needs no randomness, so it always writes the same text for the same start, and each step costs one [forward pass](#tie-the-output) of the model. The text it writes is, token by token, the model's best guess.

A product with no description is the store's need, so start with one: `front porch welcome sign wall décor`, decoded greedily for up to {{out:greedy-description:longest description, in tokens}} tokens. Then take a second product, `5 light light bulb`, and print the probability that its description is empty:

[Example: greedy-description](../examples/greedy-description.py)

The greedy description reads like the store's own text, then loops. It reads that way because it is the store's text, recombined, as in `this welcome sign is a great way to add a touch of rustic charm to your home`. It writes `features a distressed finish with a distressed finish`, and later `has a distressed finish` again, and it uses two four-token phrases twice, `this welcome sign is` and `to your home .`, which the example lists. Greedy decoding loops because the same likeliest continuation wins whenever the text reaches a similar point: `distressed` is the model's favourite finish, so it comes back. Holtzman and colleagues found the same in large models: decoding that always maximizes probability leads to text that is "bland, incoherent, or gets stuck in repetitive loops" ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)).

The greedy description also says the sign is `made of natural wood`, which nothing in the product's name says. The bulb shows a different failure: for `5 light light bulb`, the model gives the immediate end, an empty description, with a probability of {{out:greedy-description:P(the description is empty), 5 light light bulb}}. The model learned the catalog as it is, gaps included. Of the training products named as bulbs, {{out:greedy-description:training products named as bulbs with no description}} have no description, against {{out:greedy-description:training products with no description}} of all products. A [later section](#forbid-an-empty-description) deals with the empty one.

**Working rule:** Use greedy decoding when you want the same output every time and short text, such as a search-box suggestion, where the likeliest next word is the point. For anything longer than a phrase, expect it to repeat itself, and choose one of the methods in the [sections that follow](#search-for-the-likeliest-text).

## Search for the likeliest text {#search-for-the-likeliest-text}

[Greedy decoding](#choose-the-likeliest-token) picks the likeliest token at each step, which isn't the same as the likeliest text: a slightly less likely first word can lead to much likelier words after it. **Beam search** looks further. It keeps the k likeliest partial texts, the beam, scored by their **total log-probability**, the sum of the logarithms of their tokens' probabilities, where higher means likelier. At each step it extends every text in the beam by its k likeliest next tokens, and keeps the best k of the results. A text that reaches the end marker is finished and stays in the running. The heart of `lib.decoding.beam` is the line that scores each extension:

```python
            candidates += [(score + log_p[token], written + [int(token)]) for token in np.argsort(-log_p, kind="stable")[:width]]
```

Which description do you expect beam search to rank likeliest for the welcome sign? The example runs it twice, with a beam of {{out:beam-search:beam width}}. The first run completes the search query `outdoor bar` by up to {{out:beam-search:longest completion, in tokens, its end included}} tokens, the end of the name included, the way a search box might offer whole phrases; the second writes the welcome sign's description and prints the likeliest text overall and per token, its total log-probability divided by its length.

[Example: beam-search](../examples/beam-search.py)

On the query, beam search gives what a search box should: short, likely phrases, with `stool` first and then `stool cushion`. This uses the transformer alone; the mix that [chapter 8](#training-a-small-model) chose for autocomplete would need the 4-gram's probability for every candidate at every step, which `lib.decoding` doesn't compute.

The likeliest description overall is empty. Ending at once has a probability of {{out:beam-search:its probability}}, and every longer text multiplies many probabilities below one, so none comes close. Stahlberg and Byrne found the same in translation models: for most sentences the most probable translation was empty, and beam search worked only because it searched badly ([Stahlberg and Byrne, 2019](https://arxiv.org/abs/1908.10090)). Ranking by probability per token removes that bias, and gives a sensible description. It still invents a material, though: this time the sign is `crafted of iron`.

**Working rule:** Use beam search for short completions where the likeliest phrase is the goal, such as search queries, ranked by total probability. For longer text, rank finished texts per token, since total probability favours the shortest, and even then prefer the sampling methods of the [next sections](#sample-with-a-temperature), which write text closer to what people write.

## Sample, with a temperature {#sample-with-a-temperature}

Greedy decoding and beam search use only the top of the distribution. **Sampling**, which [chapter 1](#predicting-the-next-word) used to generate names, uses all of it: draw the next token at random, in proportion to its probability, so a token with probability 0.2 is chosen one time in five. Before sampling, the distribution can be made sharper or flatter with a **temperature**, T: divide every logit by T before the softmax. In symbols, with zᵢ the logit of token i, as in [chapter 5](#predict-from-a-window):

P(token i) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)

A temperature below 1 sharpens the distribution, giving more of the probability to the likeliest tokens; above 1, it flattens it, giving more to unlikely ones. A temperature near 0 approaches greedy decoding, and a temperature of 1 leaves the model's probabilities as they are. Dividing by T keeps the order of the tokens; it only changes how far apart their probabilities are.

To see what temperature does to one real choice, stop the greedy description early and read the next token's distribution at three temperatures. For each one, the example prints the three likeliest tokens, how many cover 90% of the probability, and how much falls outside temperature 1's 90% cover.

[Example: temperature-by-hand](../examples/temperature-by-hand.py)

The temperature sets how far the probability spreads. After `{{out:temperature-by-hand:text so far}}`, at temperature 1 the model needs {{out:temperature-by-hand:temperature 1.0, tokens needed to cover 90% of the probability}} tokens to cover 90%, and the rest is a long tail, each token in it unlikely. At 0.5, {{out:temperature-by-hand:temperature 0.5, tokens needed to cover 90% of the probability}} tokens cover 90%. At 1.5, it takes {{out:temperature-by-hand:temperature 1.5, tokens needed to cover 90% of the probability}} tokens, and {{out:temperature-by-hand:temperature 1.5, probability outside temperature 1.0's 90% cover}} of the probability falls outside temperature 1's 90% cover.

| Temperature | Three likeliest tokens | Tokens to cover 90% | Outside T = 1's 90% cover |
| --- | --- | --- | --- |
| 0.5 | {{out:temperature-by-hand:temperature 0.5, top 3}} | {{out:temperature-by-hand:temperature 0.5, tokens needed to cover 90% of the probability}} | {{out:temperature-by-hand:temperature 0.5, probability outside temperature 1.0's 90% cover}} |
| 1 | {{out:temperature-by-hand:temperature 1.0, top 3}} | {{out:temperature-by-hand:temperature 1.0, tokens needed to cover 90% of the probability}} | {{out:temperature-by-hand:temperature 1.0, probability outside temperature 1.0's 90% cover}} |
| 1.5 | {{out:temperature-by-hand:temperature 1.5, top 3}} | {{out:temperature-by-hand:temperature 1.5, tokens needed to cover 90% of the probability}} | {{out:temperature-by-hand:temperature 1.5, probability outside temperature 1.0's 90% cover}} |

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

Four samplers, each from the same random seed, write the welcome sign's description up to {{out:sample-descriptions:longest description, in tokens}} tokens: sampling at temperature 1, sampling at 0.7, top-k with k = 20, and top-p with p = 0.9. Then the example replays the temperature-1 sample, listing the tokens drawn from outside each step's 90% nucleus, which top-p 0.9 would have cut.

[Example: sample-descriptions](../examples/sample-descriptions.py)

Cutting the tail removes the rarest mistakes, not all of them. Sampling at temperature 1 wanders: it writes `welcome welcome sign sign`, invents a word, `glywheel`, and claims the sign is `made in the usa`. Only some of that comes from the tail: {{out:sample-descriptions:temperature 1.0, tokens drawn from outside the 90% nucleus}} tokens did, {{out:sample-descriptions:temperature 1.0, those tokens}}. The doubled words and the made-up word were built from tokens inside the nucleus, which top-p would have kept. At 0.7 the text holds together better, though it now says the sign is `made of high-quality metal`, where greedy decoding said wood. Top-k and top-p read more like the catalog than plain sampling, and still invent: top-p's sign is `cast iron`, with a `podishade` design.

Across this chapter, the welcome sign has been given a material six times, by greedy decoding, beam search, three of these samples and the [penalized greedy text](#penalize-repetition) below, and nothing in its name says what it's made of. The model writes what descriptions of welcome signs usually say, and they usually state a material. A decoding method changes which plausible material the model picks; none of them can make it pick the true one.

**Working rule:** For varied text, cut the tail with top-p around 0.9 or top-k around 20, with a temperature at or a little below 1, and fix the random seed when you need to reproduce a result. Don't expect any setting to make the text true: decoding chooses among what the model finds plausible, and checking facts is a separate job.

## Measure what each setting writes {#measure-each-setting}

Several properties can be measured without a person reading the text:

- **repetition:** the share of four-token phrases that repeat an earlier one in the same description
- **the model's own [perplexity](#score-with-perplexity) of the text:** how surprising the model finds what it wrote, pooled over every token, where low means predictable
- **whether the description ends** within the limit, as real ones do
- **whether it uses a whole word of the product's name**, as a description of that product should
- **its average length** in tokens

How close does each setting come to real text? The example draws {{out:measure-decoding:products, drawn at random from the validation products with a description}} validation products at random from those with a description, writes a description for each with every method, and compares them with the products' real ones, cut at the same {{out:measure-decoding:longest description, in tokens}}-token limit:

[Example: measure-decoding](../examples/measure-decoding.py)

Greedy text is far more predictable than anything a person wrote, with a perplexity of {{out:measure-decoding:greedy, the model's perplexity of the text}}, and {{out:measure-decoding:greedy, repeated 4-token phrases}} of its four-token phrases repeat. The real descriptions repeat {{out:measure-decoding:real descriptions, repeated 4-token phrases}} of their four-token phrases, and the model finds them fairly surprising, at a perplexity of {{out:measure-decoding:real descriptions, the model's perplexity of the text}}. That figure is higher than [chapter 8](#training-a-small-model)'s for descriptions, because these are a different sample and only their first {{out:measure-decoding:longest description, in tokens}} tokens. [Holtzman and colleagues](https://arxiv.org/abs/1904.09751) made the same observation at scale: human text isn't the text a model finds most probable.

No setting matches the real descriptions on every measure. Sampling at temperature 1 comes closest on perplexity, at {{out:measure-decoding:sampling, temperature 1.0, the model's perplexity of the text}}, but only {{out:measure-decoding:sampling, temperature 1.0, uses a whole word of the name}} of its descriptions use a word of the product's name, against {{out:measure-decoding:real descriptions, uses a whole word of the name}} of the real ones, so it drifts off the product. Top-p 0.9 ends as often as the real text, {{out:measure-decoding:top-p 0.9, ended within 60 tokens}} against {{out:measure-decoding:real descriptions, ended within 60 tokens}}, but stays on the product in only {{out:measure-decoding:top-p 0.9, uses a whole word of the name}}. Top-k 20 stays on the product most among the samplers, at {{out:measure-decoding:top-k 20, uses a whole word of the name}}, with no repetition and descriptions of {{out:measure-decoding:top-k 20, average length in tokens}} tokens on average. Twenty products is a small sample, so read these as directions, not exact rates. On staying on the product, which comes first, top-k 20 does best of the samplers.

What the measures can't say is whether a description is true. A description can be fluent, varied, on topic and wrong, and [the previous section's](#sample-and-cut-the-tail) were, about the sign's material.

**Working rule:** Compare decoding settings on measures like these, against real text of the same kind, before reading samples. Decide which measure matters most for the job: for product descriptions, staying on the product and not repeating come first. Then read a sample of the outputs anyway, for what no measure catches, and check every factual claim against the product's data.

## Penalize repetition {#penalize-repetition}

[Greedy decoding's](#choose-the-likeliest-token) loops can also be fixed without sampling. Keskar and colleagues proposed a **repetition penalty**: divide the logit of every token already generated by a factor θ above 1, so it becomes less likely ([Keskar et al., 2019](https://arxiv.org/abs/1909.05858), section 4.1). Dividing a negative logit would raise it, so implementations multiply negative logits by θ, as `lib.decoding.penalize` does, so the token becomes less likely either way. Keskar and colleagues found a factor around 1.2 balanced avoiding repetition against writing sensibly.

The repeat starts at the [greedy description's](#choose-the-likeliest-token) second `distressed`. Apply the penalty by hand to the next token's logits, check the result against `lib.decoding.penalize`, and compare the probabilities before and after. Then the description is written greedily with the penalty, to see whether the loop goes.

[Example: repetition-penalty-by-hand](../examples/repetition-penalty-by-hand.py)

With a factor of {{out:repetition-penalty-by-hand:penalty factor}}, `distressed` falls from a probability of {{out:repetition-penalty-by-hand:P(distressed), without a penalty}} to {{out:repetition-penalty-by-hand:P(distressed), with the penalty}}, and another token wins where it won before. The hand calculation matches the library's, and the penalized greedy description doesn't loop. It does still invent: this time the sign is made of `natural coir`, a fibre from coconut husks.

The penalty has a side effect that the [measurements](#measure-each-setting) show: descriptions end sooner. It penalizes every token already used, including common words such as `the` and `and` that a description needs many times, but never the end marker, which hasn't been used yet. So the end becomes relatively likelier, and {{out:measure-decoding:greedy, repetition penalty 1.3, ended within 60 tokens}} of the penalized greedy descriptions ended within the limit, against {{out:measure-decoding:greedy, ended within 60 tokens}} without the penalty, with an average length of {{out:measure-decoding:greedy, repetition penalty 1.3, average length in tokens}} tokens against {{out:measure-decoding:greedy, average length in tokens}}. Its repetition fell from {{out:measure-decoding:greedy, repeated 4-token phrases}} to {{out:measure-decoding:greedy, repetition penalty 1.3, repeated 4-token phrases}}.

**Working rule:** Add a repetition penalty around 1.2 to 1.3 when you decode greedily or at a low temperature, and check what it does to length, since it shortens text. As a rule of thumb, if descriptions come out clipped, leave common function words and punctuation out of the penalty.

## Forbid an empty description {#forbid-an-empty-description}

The [greedy example](#choose-the-likeliest-token) found the model ready to write nothing for a bulb, and [beam search](#search-for-the-likeliest-text) found nothing to be the likeliest description for a sign. Both come from the catalog: many products have no description, so the model learned that ending at once is common. To fill exactly those gaps, the model has to be stopped from reproducing them. The simple fix is a **minimum length**: forbid the end marker, by setting its logit to minus infinity, until the description has some number of tokens. `lib.decoding.generate` takes it as `shortest`.

The bulb that greedy decoding left empty is the test case. It's decoded twice, with no minimum and then with a minimum of {{out:minimum-length:minimum length, in tokens}} tokens, and the output first gives the chance its description ends at once.

[Example: minimum-length](../examples/minimum-length.py)

The minimum fills the gap, but it can't supply the facts. Without it, greedy decoding takes the empty description, at a probability of {{out:minimum-length:P(the description is empty)}}. With the minimum, it writes one, and two failures from earlier in this chapter appear in it. It repeats itself, `a light , a light , and more light`, and it states things the name doesn't: that the bulb is `made of metal`, `dimmable`, and `compatible with most standard bulbs`.

**Working rule:** Set a minimum length whenever the training text includes empty or very short examples of what you're generating. Combine it with the other settings in this chapter, not instead of them, and expect the text it forces out to need the same checking as any other.

## What goes wrong in production {#decoding-in-production}

Most of decoding's failures are quiet:

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
