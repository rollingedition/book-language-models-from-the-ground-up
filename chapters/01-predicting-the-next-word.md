---
chapter_id: predicting-the-next-word
---

# Predicting the next word

By the end of this chapter you can build a complete language model from counts, use it to suggest the next word as a shopper types, generate text with it, and measure with one number how well it predicts text it has never seen. Every later chapter replaces one part of this model with a better one, and from [chapter 3](#counting-further-back) on, measures each change the same way.

A **language model** is a program that assigns a probability to every word that could come next, given the words so far. That one ability covers a lot. Suggesting how to finish a search query means picking a likely next word. Writing a sentence means picking a next word, adding it, and picking again. And a model that predicts well has learned something about the text it was trained on, so how well it predicts is how you measure it. The large language models behind chat assistants do the same thing at a vastly larger scale: given the text so far, they assign a probability to each possible next piece of text. This book builds that machinery one part at a time, small enough to run on a laptop, so you can see what each part does.

This book follows one case study throughout: an online furniture and home-goods store, which we call the store. The store's search box needs autocomplete, so that a shopper who types `coffee` sees `table` offered as the next word, and the store needs short product descriptions written and checked. Both are language-model problems, and the store has the text to learn them from: the names and descriptions of every product in its catalog. In this chapter you count which word follows which in the product names, turn the counts into probabilities, and put them to work. The model you build here has a serious flaw, and the chapter ends by measuring it.

## Load the store's text {#load-the-store-text}

Before a model can learn which words follow which, it needs text to learn from, and a way to tell whether what it learned carries over to text it hasn't seen. The second need matters as much as the first. A model can predict the text it learned from perfectly by memorizing it, which says nothing about how it will do on the next product the store adds. So the text is split once, before anything learns from it, into three parts: a **training split** that models learn from, a **validation split** for comparing models while you build them, and a **test split** kept for final reports. Each product belongs to exactly one split, so no name or description a model is scored on appears in what it learned from.

The store's text, which every example in this book learns from, is real. It comes from WANDS, a public product-search dataset that Wayfair published with a paper at ECIR 2022 ([Chen et al., 2022](https://easychair.org/publications/preprint/wm4R)), under the MIT licence ([licence](https://github.com/wayfair/WANDS/blob/main/LICENSE)). The script `prep/wands_text.py` prepares it, and its data card, `examples/data/wands-text/README.md`, says how: the text is already lowercased, with spaces around punctuation, descriptions are cut at a sentence end to keep the file small, and products whose text names the store itself are left out, so no model in this book learns to write the store's name. The examples read the text through `lib.store`, one of the book's shared modules: short files in `examples/lib/` that hold setup code many examples need, which you can open and read.

[Example: store-text](../examples/store-text.py)

The store's text covers {{out:store-text:products}} products: {{out:store-text:train products}} in the training split, {{out:store-text:validation products}} in the validation split and {{out:store-text:test products}} in the test split. This chapter learns from product names only, which are short and repetitive enough for counting to work. The training names hold {{out:store-text:words in training names}} words, of {{out:store-text:different words in training names}} different kinds. That second number is large for a catalog of furniture: many names start with a product line's name, a word that appears in a handful of products and nowhere else. This book doesn't print those names: where an example's output would show one, it prints `[name]`, which marks any word used in fewer than 30 of the training descriptions, most often a product line's name. Those rare words will matter by the end of the chapter.

**Working rule:** Split your text into training, validation and test parts before any model sees it, and split by document (here, by product), so near-duplicates of one item never straddle two parts. Compare models on validation text while you work, and touch the test text only for the final report. Learn from the shortest, most regular text you have first; this chapter starts with names, and descriptions come later in the book.

## Count which word follows which {#count-word-pairs}

With the training names loaded, a simple model of "what comes next" is a table of how often each word followed each other word. A **bigram** is a pair of adjacent words, such as `coffee table`. Count every bigram in the training names, and you know, for any word, which words came after it and how often. Claude Shannon used exactly this idea in the paper that founded information theory, approximating English by choosing each word according to how often it follows the one before ([Shannon, 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)).

Two details make the counts complete. A name's first word follows nothing, and its last word is followed by nothing, but both are worth predicting: the first word decides what kind of product a shopper is typing, and knowing where a name ends lets a model stop. So each name is padded with two markers, `<s>` at the start and `</s>` at the end, which count as words. The pair `<s> coffee` then records that a name started with `coffee`, and `table </s>` records that a name ended with `table`.

For the store, start with one word whose next word matters to the search box: `coffee`. The example counts every pair in the training names, then looks at what followed `coffee`:

[Example: count-next-words](../examples/count-next-words.py)

The training names hold {{out:count-next-words:word pairs}} word pairs, of {{out:count-next-words:different pairs}} different kinds, and `coffee` is followed by a word {{out:count-next-words:words after coffee}} times. One word dominates: almost every `coffee` in the store's names is a `coffee table`. A comma comes third, because the store's text treats punctuation as words of its own. That's a choice about tokens, the units a model counts, and [chapter 2](#tokens) shows how much it matters.

The table of counts is the whole model. It has no rules about grammar and no knowledge of furniture; it has seen `coffee table` many times and `coffee lamp` never, and that is all it knows. The shared module `lib.bigrams` holds this same counting code, as the class `Bigrams`, so later examples can build the model in one line.

**Working rule:** Count pairs in the training split only, and pad every text with start and end markers so the model learns how texts begin and end. Keep the raw counts, not only the probabilities you'll derive from them: the counts tell you how much evidence each probability rests on, and the next sections need them.

## Turn counts into probabilities {#counts-to-probabilities}

Counts say what happened; a language model needs probabilities, which say how likely each next word is. The step from one to the other is division. The probability of a word given the word before it is the number of times the pair occurred, divided by the number of times the first word occurred with anything after it. In symbols, writing `w` for the previous word and `v` for the next one:

P(v | w) = count(w v) / count(w)

Read it as "the probability of `v`, given `w`". Estimating probabilities this way, from the fraction of times something happened, is called **maximum likelihood estimation**: of all the probability tables you could choose, it's the one that makes the training text likeliest ([Jurafsky and Martin, chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf)).

For the store, apply the formula to the word `coffee` and four candidates for the next word:

[Example: next-word-probabilities](../examples/next-word-probabilities.py)

The probability of `table` after `coffee` is {{out:next-word-probabilities:P(table | coffee)}}, of `mug` {{out:next-word-probabilities:P(mug | coffee)}} and of `jar` {{out:next-word-probabilities:P(jar | coffee)}}. The probabilities of every word that followed `coffee` add up to {{out:next-word-probabilities:sum over every word after coffee}}, as a probability distribution must: whatever comes next, it's one of these words.

The fourth line is the one to remember. `coffee lamp` never occurs in the training names, so its probability is {{out:next-word-probabilities:P(lamp | coffee)}}. The model doesn't say that a coffee lamp is unlikely; it says a coffee lamp is impossible. For a furniture store that's a reasonable bet, but the same zero applies to every pair the model hasn't seen, including perfectly ordinary ones. That's the flaw this chapter ends on.

**Working rule:** Derive probabilities from counts by division, and check that each word's probabilities sum to one; a sum that doesn't is a bug in the counting. Treat a probability as only as good as the count under it: 0.9 from 1,000 pairs is evidence, while 1.0 from a single pair is an anecdote.

## Suggest the next word as a shopper types {#suggest-next-words}

A probability table answers the store's first question directly. Autocomplete offers the likeliest next words, so a shopper who has typed `coffee` sees `table` first. With the bigram model, the suggestions for any typed text are the words with the highest probability after its last word. The bigram model's class, `Bigrams`, from [count which word follows which](#count-word-pairs), has a method, `next_words`, that returns them, likeliest first.

For the store, type five prefixes a shopper might enter, and ask for three suggestions each:

[Example: autocomplete](../examples/autocomplete.py)

The suggestions after `coffee`, `bar` and `kitchen` are what a shopper would want: tables, stools, sinks and islands. The ones after `outdoor` are weaker. The top suggestion, `door`, is the start of a real product type: of the {{out:autocomplete:training names containing 'outdoor door'}} training names containing `outdoor door`, {{out:autocomplete:of them, followed by 'mat'}} continue with `mat`. But a one-word suggestion can't offer a two-word product type, and after `door` the shares are small, because `outdoor` is followed by many different words. A comma and an ampersand also show up as suggestions, which no search box should offer. Filtering punctuation out of suggestions is a reasonable patch here, and [chapter 2](#tokens) treats the cause.

The last line shows the model's real limit. A shopper who types `patio bar` gets exactly the suggestions for `bar`, because a bigram model looks at one word of context and nothing before it. `patio` is lost, so the model offers `pull` and `&`, which belong to cabinet hardware and to names such as `bar & counter stools`. Using more context is the subject of [chapter 3](#counting-further-back), and a model that weighs every earlier word is where the book is heading.

**Working rule:** For autocomplete, offer the top few next words by probability, and drop tokens that aren't words, such as punctuation. Expect a bigram model's suggestions to be good after words that are almost always followed by the same thing, such as `coffee`, and poor after words with many possible continuations, such as `outdoor`. When the earlier words matter, as they do in `patio bar`, a bigram model can't help, however much text it learns from.

## Generate names one word at a time {#generate-one-word-at-a-time}

The same probability table can write text as well as complete it. To **generate**, start at `<s>`, draw the next word at random in proportion to its probability, append it, and repeat from the word you drew, until the model draws `</s>`. Drawing at random, rather than always taking the likeliest word, is what lets the model produce varied names instead of the same one every time. The large language models this book builds toward write text the same way: one token at a time, each chosen from probabilities computed from the text so far.

For the store, generate six product names. The random generator uses a fixed seed, so the example prints the same names on every run, and the names in this book match what the code produces:

[Example: generate-names](../examples/generate-names.py)

Read the names closely and two things stand out. Each adjacent pair of words is plausible, because each pair occurred in a real name: `genuine leather`, `kitchen cart`, `lounge chair`. And the names as a whole often aren't, because nothing keeps a name on one product. One name runs from a bistro set to a dimmable light to a smart dryer, joined by commas, since after a comma almost anything can follow. The model has no memory beyond the last word, so it can't know it was describing a bistro set. Others attach a product line's name, printed as `[name]`, to a product of a different kind, such as a leather lounge chair, because names that start with a product line are common in the training text. Nothing is invented: a bigram model can only write words it counted, and the example confirms it, with {{out:generate-names:generated words never seen in training}} generated words that the training names don't contain.

Generated names make a model's limits visible in a way a number can't, which makes them worth reading. They don't measure anything, though: six names are too few to compare two models, and a reader's sense of "plausible" changes from day to day. The next section gives the number.

**Working rule:** Generate samples whenever you change a model, and read them, because they show failures that a score hides, such as text drifting from one product to another. Use a fixed seed so the samples are reproducible. Never compare models by their samples; compare them by perplexity on validation text.

## Score a model with perplexity {#score-with-perplexity}

A language model is good when it gives high probability to the text that actually occurs. That suggests a score: run the model over real text, and see how much probability it gave to each word that came next. The standard way to summarize this is **perplexity**, and this book uses it to compare every model it builds.

Start from one pair. If the model gave the actual next word probability p, its **surprise** at that word is −log p: zero when it was certain (p = 1) and larger the less it expected the word. The **cross-entropy** of a text is the average surprise over every word in it, written with N as the number of predicted words:

cross-entropy = −(1/N) Σ log P(next word | previous word)

**Perplexity** is that average turned back into a count: perplexity = exp(cross-entropy) ([Jurafsky and Martin, chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf)). It reads as the number of equally likely words the model was choosing between, on average, at each step. A perplexity of 10 means the model was, on average, as uncertain as if it had to pick among 10 equally likely words. Lower is better, and a model that knew every next word would score 1. This book uses natural logarithms throughout, which changes cross-entropy's units but not perplexity.

For the store, score one training name, `sled coffee table`, pair by pair, and then every training name:

[Example: perplexity-by-hand](../examples/perplexity-by-hand.py)

The surprise is almost all at the start. Few names begin with `sled`, so the model was surprised by it, and after that each word was close to expected: `coffee` follows `sled` in most training names that contain `sled`, and `table` follows `coffee`. The name's average surprise is {{out:perplexity-by-hand:average -log P}}, and its perplexity {{out:perplexity-by-hand:perplexity of the name}}. Over every training name, the model's perplexity is {{out:perplexity-by-hand:perplexity of every training name}}: on the text it learned from, it was choosing among about that many likely words at each step.

That number flatters the model, because it's measured on the text the counts came from. Every pair in the training names has a count, so no pair surprises the model infinitely. The real test is text it hasn't seen.

**Working rule:** Score every language model by its perplexity on validation text, computed exactly as above, with natural logarithms and every predicted word counted, the end marker included. Compare perplexities only between models that use the same tokens and the same validation text, since splitting text differently changes N and what counts as a word. Report training perplexity only beside validation perplexity, to show how much the model memorized.

## Unseen word pairs break the model {#unseen-pairs-break-the-model}

A model trained by counting knows only the pairs it has counted, and the validation names were kept apart from the training names for exactly this reason: they show what happens when the model meets text it never saw. Before running the next example, predict its first line. The model scored {{out:perplexity-by-hand:perplexity of every training name}} on the training names, in [score a model with perplexity](#score-with-perplexity). What will its perplexity be on the {{out:perplexity-on-new-names:validation names}} validation names?

[Example: perplexity-on-new-names](../examples/perplexity-on-new-names.py)

The validation perplexity is infinite: the example prints {{out:perplexity-on-new-names:validation perplexity}}. One pair the model gave probability zero is enough, because its surprise, −log 0, is infinite, and so is any average that includes it. And there isn't only one. Of the {{out:perplexity-on-new-names:validation word pairs}} word pairs in the validation names, {{out:perplexity-on-new-names:pairs never seen in training}} never occur in the training names, {{out:perplexity-on-new-names:share never seen in training}} of them, and {{out:perplexity-on-new-names:names with an unseen pair}} of the {{out:perplexity-on-new-names:validation names}} validation names contain at least one. The first few show why: a validation name that starts with a product line the training names never mention, printed as `[name]`, makes both the first pair and the second unseen.

This is the flaw the [probabilities section](#counts-to-probabilities) warned about. Maximum likelihood gives every unseen pair probability zero, and in real text most new documents contain something unseen: a new product line, an unusual size, a familiar word in an unfamiliar place, such as `half-circle` after `wood`. More training text helps a little, but it can't solve the problem. Language keeps producing new combinations, so a model that calls every unseen combination impossible will always fail on new text.

The fix has to give unseen pairs some probability, which means taking a little away from the pairs the model has seen. That's called **smoothing**.

**Working rule:** Never trust a language model that gives zero probability to anything; check for infinite or undefined perplexity on validation text before you compare models. When you see one, count the unseen pairs, as above, to know how much of the text the model can't handle.

## Give unseen pairs a little probability {#smooth-unseen-pairs}

The model in [unseen word pairs break the model](#unseen-pairs-break-the-model) fails because it gives unseen pairs probability zero, and smoothing gives them some. The simplest smoothing method, **add-one smoothing**, pretends that every possible pair occurred once more than it did. Every count goes up by one, including the zeros, and the totals grow to match:

P(v | w) = (count(w v) + 1) / (count(w) + V)

where V is the size of the **vocabulary**, every word the model can predict. A word the training text doesn't contain at all is counted as one more vocabulary entry, `<unk>`, for unknown, so the probabilities still sum to one. No pair now has probability zero, so perplexity on new text is finite. The question is what it costs.

For the store, compare the counted and add-one models on the probability of `table` after `coffee`, and on perplexity:

[Example: add-one-smoothing](../examples/add-one-smoothing.py)

The vocabulary holds {{out:add-one-smoothing:vocabulary}} words, the end marker and `<unk>` included, and that's the problem. Add-one smoothing adds one imagined occurrence for every one of them after every word, so after `coffee` it imagines tens of thousands of extra pairs against about a thousand real ones. The probability of `table` after `coffee` falls from {{out:add-one-smoothing:P(table | coffee) counted}} to {{out:add-one-smoothing:P(table | coffee) add-one}}. The model now believes far less in what it has seen, and the perplexities show it: on the training names, perplexity rises from {{out:add-one-smoothing:training perplexity, counted}} to {{out:add-one-smoothing:training perplexity, add-one}}, and on the validation names it's {{out:add-one-smoothing:validation perplexity, add-one}}. The validation perplexity is finite, which is progress, but the model gives so much probability to word pairs that never happen that its predictions are nearly useless.

Add-one smoothing fixes the infinity and breaks almost everything else, because it adds one imagined pair for every word in the vocabulary, however much evidence the real counts hold, and spreads them evenly. Better smoothing methods take far less probability away from seen pairs, and give it to unseen pairs in proportion to how often their words occur elsewhere. They are the subject of [chapter 3](#counting-further-back), and the counted model of this chapter, looking further back and with a better smoothing method, becomes the baseline that every later model in the book is measured against.

**Working rule:** Never ship a counted model without smoothing, and never ship one with add-one smoothing: measure validation perplexity and expect add-one to lose to every method in [chapter 3](#counting-further-back), which compares them on the same tokens. Use add-one only as this chapter does, to show that zero probabilities are fixable and that the fix needs care.

## What goes wrong in production {#next-word-in-production}

A bigram autocomplete is cheap and fast, and it's still a fair baseline for a search box. These are the problems that come with it:

- **The catalog moves on.** A model counted from last month's names doesn't know this month's product lines. A shopper who types a new product line gets no suggestions at all, because the model has no counts after it, until the counts are rebuilt, so rebuild them on a schedule and whenever the catalog changes.
- **Suggestions can embarrass.** The model suggests whatever followed a word in its training text, punctuation included. Filter suggestions through a list of allowed tokens, and review the most frequent suggestions by hand before they go live.
- **Popular products crowd out the rest.** A word followed by one product type most of the time, such as `coffee`, will always suggest that type. Shoppers looking for a coffee mug have to type more. Offer several suggestions, not one.
- **One word of context is too little.** As `patio bar` showed, the model ignores everything but the last word. If shoppers type long queries, measure how often the suggestions change once you condition on more of them, which [chapter 3](#counting-further-back) does.
- **The table grows with the text.** A bigram table has an entry for every pair ever seen; with longer contexts, the number of entries explodes. Count only what you need, and prune pairs seen once if memory is tight.

## What this chapter decided {#next-word-decisions}

- **The store's text is split once, by product, into training, validation and test splits, and models learn from the training split only.** A score on text a model learned from measures memory, not prediction, and splitting by product keeps near-duplicate names of one product from landing on both sides.
- **Every text is padded with `<s>` and `</s>`.** Models then learn how texts start and end, which autocomplete needs for a shopper's first word and generation needs to stop.
- **Models are compared by perplexity on validation text,** with natural logarithms and every predicted word counted, the end marker included. Generated samples are read for what they show, but never used to choose between models.
- **The bigram model of the store's product names, built from counts, is the book's first model.** It suggests good next words after words with one common continuation, such as `coffee`, and poor ones after words with many, such as `outdoor`. It sees one word of context, so `patio bar` and `bar` get the same suggestions.
- **Unsmoothed, its validation perplexity is infinite; with add-one smoothing it's finite but poor.** [Chapter 3](#counting-further-back) gives it a better smoothing method, makes it the baseline, and keeps the same splits and the same measure, so every later model is compared with it on equal terms.
