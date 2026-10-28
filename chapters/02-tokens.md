---
chapter_id: tokens
---

# Tokens

By the end of this chapter you can build a byte-pair encoding tokenizer from scratch, choose its vocabulary size by measuring what each size costs, and read the tokenizer of an open language model well enough to apply its merges by hand. Every model in the rest of the book reads the store's text through the tokenizer you build here.

A **token** is the unit a language model reads and predicts: whatever the text is cut into before the model sees it. [Chapter 1](#predicting-the-next-word) cut the store's product names into words, because the store's text, which this book's examples learn from, already has spaces between words and around punctuation. That choice was quick and it fell short in two ways. The bigram model couldn't give any probability to a word it never saw in training, so a word it hadn't met in training, such as `barstools`, broke it. And it treated `chair` and `chairs`, or `48` and `49`, as unrelated words, with nothing learned about one carried over to the other. The first problem comes from the choice of tokens, so this chapter fixes the tokens. The second needs more than tokens can give, and [chapter 4](#words-as-vectors) takes it up.

The store is the online furniture and home-goods store this book follows throughout, and its text is the product names and descriptions of its catalog. This chapter measures three ways to cut that text into tokens: words, characters, and the middle ground that most open generative models use, byte-pair encoding. Then it opens one of those models' tokenizers and takes it apart.

## Words leave gaps {#words-leave-gaps}

Cutting text into words gives a model tokens that carry meaning, which is why [chapter 1](#predicting-the-next-word) started there. The cost is the vocabulary. A model can only predict tokens in its **vocabulary**, the set of tokens it learned from training text, so any word that first appears after training is out of reach: an **out-of-vocabulary** word. In text with names, numbers and new products, there are always some.

For the store, count how many words in the validation names never appear in the training names. The validation names are the 10% of the store's products that no model learns from, kept aside to show how a model handles text it hasn't seen:

[Example: unseen-words](../examples/unseen-words.py)

The training names hold {{out:unseen-words:different words in training names}} different words, and {{out:unseen-words:words seen once in training}} of them occur only once, which is common for product lines and model numbers. Of the {{out:unseen-words:validation words}} words in the validation names, {{out:unseen-words:validation words never seen in training}} never occur in training, {{out:unseen-words:share never seen in training}} of all validation words. Most of them, {{out:unseen-words:share of them that are rare words, most often product lines' names}}, are rare words, most often product lines' names, which [chapter 1](#load-the-store-text) prints as `[name]`; the rest are ordinary words that happen to miss the training names, such as `neutral` and `noise-free`. A word-level model can't predict any of them, can't score text that contains them without special handling, and can't relate `noise-free` to `noise` and `free`, which it knows well.

A larger training set would shrink the gap but never close it, because the catalog keeps adding product lines, and a word seen once gives a model almost nothing to learn from anyway.

**Working rule:** Measure the out-of-vocabulary rate of your text on held-out data before choosing word tokens. As a rule of thumb, if more than a percent or two of held-out words are unseen, as in the store's names, don't use words as tokens. Word tokens remain reasonable for text with a closed vocabulary, such as a fixed set of commands.

## Characters cover everything {#characters-cover-everything}

Going to the other extreme fixes the gap. If every character is a token, the vocabulary is tiny and almost nothing is out of it: a new product line is spelled with the same letters as an old one. The price is length. Each word becomes several tokens, so the model has more steps to predict, and the context it needs, such as the start of a product's name, is many more tokens back.

For the store, compare the training names' characters with the validation names', and count tokens both ways:

[Example: character-tokens](../examples/character-tokens.py)

The training names use {{out:character-tokens:different characters in training names}} different characters, and only {{out:character-tokens:validation characters never seen in training}} characters in the validation names never appear in them. As words, the validation names are {{out:character-tokens:validation tokens as words}} tokens; as characters, {{out:character-tokens:validation tokens as characters}}, or {{out:character-tokens:characters per word}} characters for each word, counting the spaces. A model reading characters has to predict `c`, `o`, `f`, `f`, `e`, `e` before it gets to the point where a word model starts: knowing that `table` is likely next.

Characters also leave a last, small gap. The {{out:character-tokens:validation characters never seen in training}} unseen characters are a soft hyphen and a `t` with a cedilla, and a character tokenizer can't spell a word that contains one. Open models close that gap by working with **bytes** instead: in UTF-8, the usual text encoding, every character in every language is stored as one to four bytes, and there are only 256 possible byte values, so a vocabulary that starts from all 256 can represent any text at all ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), section 2.2). The tokenizer read later in this chapter does exactly that.

Longer sequences cost in two ways. Every step a model takes is computation, so {{out:character-tokens:characters per word}} times the steps is at least {{out:character-tokens:characters per word}} times the work, and more for the models of later chapters, whose cost grows faster than the length. And every model in this book looks back over a limited window of tokens; in characters, that window covers that many times fewer words.

**Working rule:** Use characters as tokens only for small experiments, or as a fallback inside a larger scheme, as the next sections do. Check how many characters per word your text has, since that factor multiplies every sequence length and every training step.

## Merge the most frequent pair, again and again {#merge-frequent-pairs}

Words are too coarse and characters too fine, so the method most open generative models use, SmolLM2 among them, sits between them. **Byte-pair encoding**, or BPE, starts from characters and builds bigger tokens out of the pairs that occur most often. It began as a data-compression method, which Sennrich, Haddow and Birch adapted to cut words into pieces for translation models ([Sennrich et al., 2016](https://aclanthology.org/P16-1162/)). The algorithm is short:

1. Spell every word in the training text as characters, and mark its end with `</w>`, so a merge never joins one word to the next.
2. Count every adjacent pair of symbols, weighting each word by how often it occurs.
3. Merge the most frequent pair into one new symbol everywhere, and add it to the vocabulary.
4. Repeat from step 2, as many times as the vocabulary size you want.

The list of merges, in the order they were learned, is the tokenizer. To encode a word, spell it as characters and apply the merges in the same order.

For the store, run the algorithm by hand on the training names for eight merges, written out in full so every step is visible:

[Example: merge-pairs-by-hand](../examples/merge-pairs-by-hand.py)

The first merges join common letters to the end marker, such as `e` with `</w>`, because so many of the store's words end in `e`, `r` or `d`. Then common pieces inside words appear, such as `in`, `an` and `st`. After eight merges, `table` is spelled `t a b l e</w>`: still mostly characters. With thousands of merges, common words such as `table` become single tokens, while rare words stay split into pieces that other words share.

That's the idea that solves [chapter 1](#predicting-the-next-word)'s first problem. Frequent words get their own tokens, so sequences stay short. Rare and new words are spelled from pieces the model has seen, so any word spelled with characters the training text contains is in reach. BPE doesn't solve the second problem: as the [next section](#train-the-store-tokenizer) shows, a frequent word and its plural can each become a whole token with nothing in common.

**Working rule:** Learn merges from the training split only, as you would any other model, and keep the order of the merges, since encoding depends on it. Break ties between equally frequent pairs by a fixed rule, as the example does, so the same text always gives the same tokenizer.

## Train the store's tokenizer {#train-the-store-tokenizer}

The algorithm in the [previous section](#merge-frequent-pairs) recounts every pair after every merge, which is fine for eight merges and too slow for thousands. The book's shared module `lib.bpe` runs the same algorithm, but counts pairs once and, after each merge, updates only the words the merge changed, keeping the counts in a heap so the most frequent pair is always on top. It learns thousands of merges in seconds, and gives the same merges the plain version does: the hand-written example's last line checks its eight merges against `lib.bpe.train`.

For the store, learn 4,000 merges from the training split's names and descriptions together, since the book's models will learn from both, and encode two names and three pairs of words:

[Example: store-tokenizer](../examples/store-tokenizer.py)

The training text holds {{out:store-tokenizer:different words in training text}} different words, and the tokenizer's vocabulary has {{out:store-tokenizer:tokens in the vocabulary}} tokens: the {{out:store-tokenizer:characters and the end marker}} characters of the training text and the end marker, one new token for each of the 4,000 merges, and `<unk>`, which stands for any character the training text never contained. Common words come out whole: `sled coffee table` is three tokens, one per word. The word `barstools`, which [chapter 1's](#unseen-pairs-break-the-model) bigram model couldn't handle because the training names never contain it, comes out as two pieces the tokenizer knows well, `bar` and `stools</w>`. The last three lines show what BPE doesn't do. `chair` and `chairs` are both common, so each became a whole token, and the two tokens share nothing. `48` is one token and `49` two, `4` and `9</w>`. And `noise-free` splits into pieces that share nothing with `noise`. Tokens give every word a spelling, not a meaning; relating them is the job of [chapter 4](#words-as-vectors).

A model reading these tokens can spell any word made of characters the training text contains. It still has to learn what a new product line is, but it can at least predict one, and give it a probability.

**Working rule:** Train the tokenizer on the same text your model will learn from, here names and descriptions together, so frequent words in that text become single tokens. Build it once and use it for every model you compare; the book's models all read the store's text through `lib.bpe.store_tokenizer()`.

## Choose how many merges {#choose-how-many-merges}

The number of merges sets the vocabulary size, and it's the one real choice in a BPE tokenizer. More merges make sequences shorter, which saves computation. But each new token covers rarer text, and a token the training text shows only a few times gives a model too little to learn from: the same problem as rare words, in a new place.

For the store, learn 16,000 merges once, and use the first 0, 1,000, 2,000, 4,000, 8,000 and 16,000 of them as six tokenizers; the first k merges of a longer run are exactly the merges a k-merge run would learn. For each, count how many different tokens the training text uses, how many tokens each validation word becomes, and what share of the tokens appear fewer than 20 times in the training text:

[Example: vocabulary-size](../examples/vocabulary-size.py)

Before you read the table, consider what each end of it should show. With no merges, every token is a character, so tokens are frequent but words are long. With many merges, words are short, but many tokens are rare. The table bears that out. The last column is the share of the different tokens the training text uses, its token types, that it shows fewer than 20 times. With characters only, a word takes more than five tokens, and {{out:vocabulary-size:rare token types with characters only}} of the token types are rare, because the descriptions contain unusual characters. At 4,000 merges, a validation word takes {{out:vocabulary-size:tokens per word at 4,000 merges}} tokens on average, and {{out:vocabulary-size:rare token types at 4,000 merges}} of the token types are rare, about as few as at 2,000. From 8,000 merges the rare share climbs, and at 16,000 it's {{out:vocabulary-size:rare token types at 16,000 merges}}: most of the extra merges cover words the training text barely contains.

The vocabulary size also sets part of the model's size. A neural language model, from [chapter 5](#a-neural-language-model) on, stores a vector of numbers for every token in its vocabulary, and scores every token in the vocabulary at every step it predicts. A vocabulary four times as large means four times as many of those vectors to learn, and four times as many scores to compute at every step, whatever the text. For the small models this book trains on a laptop, that cost matters as much as sequence length.

So the store's tokenizer uses 4,000 merges: past that, each merge buys little length and adds a token that the book's small models can't learn well. Large open models use vocabularies of tens of thousands of tokens or more, SmolLM2's has {{out:smollm2-tokens:SmolLM2 vocabulary}}, because they learn from far more text, so even their rarer tokens occur often enough.

**Working rule:** Choose the number of merges by measuring, on your training text, the share of rare token types, and on held-out text, the tokens per word; stop where tokens per word stops falling much and rare types start climbing. Scale the vocabulary with the training text; as a rule of thumb, a small model on a few million words wants thousands of merges, not tens of thousands.

## Read an open model's tokenizer {#read-an-open-tokenizer}

Open language models ship their tokenizer as a file beside their weights, and reading one shows the same ideas at a larger scale, plus a few choices the store's tokenizer didn't have to make. This book uses SmolLM2, an open language model by Hugging Face whose weights and tokenizer anyone can download, under the Apache 2.0 licence ([model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)). The book keeps a copy of its tokenizer file in `examples/data/smollm2-tokenizer/`, fetched at a pinned revision by `prep/smollm2_tokenizer.py`, so the examples run offline. The examples read it with the tokenizers library, {{version:tokenizers}}, the library the file was made for.

For the store, encode three product names with SmolLM2's tokenizer and with the store's, and count tokens per validation name each way:

[Example: smollm2-tokens](../examples/smollm2-tokens.py)

SmolLM2's vocabulary has {{out:smollm2-tokens:SmolLM2 vocabulary}} tokens, against the store tokenizer's {{out:store-tokenizer:tokens in the vocabulary}}, because it was trained on far more text than furniture names. Three differences in its output are worth naming:

- **Spaces are part of tokens.** `Ġ` stands for a space: `Ġtable` is "table" with the space before it. Instead of marking where a word ends, as the store's `</w>` does, this tokenizer keeps each word's leading space, so the text can be rebuilt exactly, spaces included.
- **Every digit is its own token.** `48` becomes `4` and `8`, and `6.5` becomes `6`, `.` and `5`. Splitting digits stops the vocabulary from filling with numbers, each seen a few times, and treats all numbers the same way.
- **A word at the start of a text has no space before it, so it's a different token.** SmolLM2's vocabulary has `Ġsled` as one token, but not `sled` without the space, so a name that starts with `sled` gets `s` and `led`, as the last lines of the output show. And `barstools`, which SmolLM2's vocabulary lacks as a whole word, splits into `bar`, `st` and `ools`, as the first lines show.

On the store's validation names, SmolLM2 uses {{out:smollm2-tokens:tokens per validation name, SmolLM2}} tokens per name and the store's tokenizer {{out:smollm2-tokens:tokens per validation name, store}}, against {{out:smollm2-tokens:tokens per validation name, words}} words. A tokenizer trained on your own text is slightly more efficient on it, partly because it doesn't split numbers into digits, and partly because SmolLM2 spends a separate `Ġ` token before every number and splits each name's first word, which has no space before it.

**Working rule:** When you use an open model, use its tokenizer exactly as shipped; a model only understands the tokens it was trained on. Count tokens with that tokenizer, not words, whenever you budget for context length or cost. When you train your own model on narrow text, train your own tokenizer on that text.

## Apply SmolLM2's merges by hand {#apply-merges-by-hand}

SmolLM2's tokenizer file looks like a black box until you see that it holds the same thing the store's tokenizer does: a list of merges, in order. The file, `tokenizer.json`, has three parts that matter here. A pre-tokenizer splits text into pieces before any merging: its steps, which the example prints, split digits apart one at a time, then split words with their leading spaces and map every byte to a printable character, which is where `Ġ` comes from. That mapping is GPT-2's `bytes_to_unicode` ([GPT-2's encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py)). Then a list of merges, ranked by when they were learned. And the vocabulary, which gives each token its number.

For the store, take the pre-tokenizer's pieces from the library, apply the merges by hand, earliest-learned first, as in the [store's tokenizer](#train-the-store-tokenizer), and compare the result with the library's:

[Example: merges-by-hand](../examples/merges-by-hand.py)

The tokens built by hand match the library's for every name: the line ends with `same as the library: True` each time. That's all SmolLM2's tokenizer does to these names; it also defines special tokens, matched before any of these steps. It differs from the store's in its pre-tokenizer and its training text, but it encodes text by repeatedly merging the earliest-learned pair, as the store's does.

The match is worth checking because subtle differences in how text is split matter downstream. A model trained on one tokenizer and fed another's tokens reads nonsense without raising an error, since every token number is valid. The same thing happens, and is harder to spot, when one version of a tokenizer differs from another in how it handles spaces or digits.

**Working rule:** Treat a tokenizer as part of a model: version it with the model, load it from the model's own files, and test it on your text, as this section does, whenever you upgrade either one. When two tokenizers disagree on even one token for the same text, find out why before you trust anything a model built on them says.

## What goes wrong in production {#tokens-in-production}

Tokens decide what a model can see, and most tokenizer problems show up far from the tokenizer:

- **A mismatched tokenizer fails silently.** A model given another tokenizer's token numbers reads valid numbers that mean the wrong things, and produces confident nonsense. Ship the tokenizer file with the model, and test them together.
- **Costs are counted in tokens.** Context limits and per-token prices apply to tokens, not words. The store's names take {{out:smollm2-tokens:SmolLM2 tokens per word}} tokens per word in SmolLM2's tokenizer; text full of numbers, codes or another language takes more. Measure your own text.
- **Numbers are hard.** A tokenizer that splits digits apart makes a model reason about `48` as two tokens; one that doesn't gives `48` and `49` unrelated tokens. Either way, expect models to handle sizes and quantities worse than words, and check them separately.
- **Special tokens can be typed.** Open models mark the start or end of a text, or a turn in a conversation, with special tokens written as strings such as `<|endoftext|>`. If text a user typed contains one of those strings and the tokenizer turns it into the special token, the model reads a control signal the user shouldn't be able to send. Check how your tokenizer handles special-token strings in untrusted text.
- **New text drifts from the training text.** A tokenizer trained on last year's catalog splits this year's product lines into more pieces. Measure tokens per word on new text from time to time; a steady rise means the tokenizer is falling behind its text.

## What this chapter decided {#tokens-decisions}

- **The book's models read the store's text as byte-pair encoding tokens.** Words left {{out:unseen-words:share never seen in training}} of validation words out of the vocabulary, and characters made sequences {{out:character-tokens:characters per word}} times as long as words. BPE avoids both: any word spelled with characters the training text contains can be encoded, and common words are single tokens. It doesn't relate words to each other, such as `chair` and `chairs`; that's [chapter 4](#words-as-vectors)'s job.
- **The store's tokenizer learns 4,000 merges from the training split's names and descriptions,** in `lib.bpe.store_tokenizer()`. At that size, a word averages {{out:vocabulary-size:tokens per word at 4,000 merges}} tokens, and few token types are too rare to learn. More merges shorten sequences a little and add many rare tokens.
- **Each word ends with `</w>`, so merges never cross words.** Open models such as SmolLM2 attach the space to the next word instead; both keep word boundaries visible to the model.
- **Open models are always used with their own tokenizers.** A model only understands the token numbers it was trained on, so [chapter 10](#inside-an-open-model) reads SmolLM2's model with the tokenizer this chapter took apart, and every recording of an open model in this book names the tokenizer it used.
- **Perplexities from different tokenizers aren't comparable.** A model that predicts more, smaller tokens takes more steps and has a different perplexity on the same text, so every comparison of models in this book uses the store's tokenizer, and [chapter 3](#counting-further-back) starts that comparison over on its tokens.
