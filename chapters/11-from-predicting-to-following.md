---
chapter_id: from-predicting-to-following
---

# From predicting to following instructions

By the end of this chapter you can explain how a model that predicts the next token becomes one that follows requests, build the two training losses that do it, and measure what that change does and doesn't fix. You'll compare a base model with its instruction-tuned version on the same requests, measure how often each makes up a product detail, check whether its confidence warns you when it does, and find where in a longer context a model stops finding what it was given.

This book follows one case throughout: an online furniture and home-goods store, which it calls the store, and the text of its product names and descriptions. Every model in the book so far has done one thing: predict the next token of text like its training text. [Chapter 10](#inside-an-open-model) read **SmolLM2-360M**, an open model trained that way on general text, and found it scored the store's names far worse, per word, than the store's own small model, and its descriptions worse. People don't want a continuation of their text; they want an answer to their request. Models are taught that in two further steps of training.

**Instruction tuning**, or supervised fine-tuning, continues training the model on examples of requests paired with good responses, so that predicting the next token of such a pair means answering the request. Wei and colleagues found that tuning on many tasks phrased as instructions made a model better at tasks it had never seen ([Wei et al., 2022](https://arxiv.org/abs/2109.01652)). **Preference tuning** then trains the model to prefer the better of two responses to the same request. Ouyang and colleagues had people rank the responses, and trained with reinforcement learning from a reward model learned from those rankings ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)); **Direct Preference Optimization**, DPO, gets the same effect with a simple loss on the preference pairs themselves ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).

**SmolLM2-360M-Instruct** is the base model of [chapter 10](#inside-an-open-model) after both steps: supervised fine-tuning on public and curated datasets, then DPO on UltraFeedback ([model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/README.md)). UltraFeedback's preference pairs were scored by a large language model, not by people ([Cui et al., 2023](https://arxiv.org/abs/2310.01377)). This chapter calls SmolLM2-360M the base model and SmolLM2-360M-Instruct the tuned model. The script `recordings/instruction_tuning.py` ran both once, at pinned revisions, decoding greedily, so each answer is the model's likeliest and the run can be repeated exactly, and saved what they wrote and how likely they found each token it wrote. The examples measure everything from those files.

## Build the two training losses by hand {#build-the-losses}

Both steps of tuning are training, with the loop of [chapter 8](#update-with-adamw); what changes is the loss. Instruction tuning uses the ordinary next-token loss with one difference: only the response's tokens count. The request is read, as context, but the model isn't trained to predict it, since the point is to answer requests, not to write them.

For the store, lay a product out as a request and a response, its name as the request and its description as the response, as the store's model already reads it. Compute the store's model's loss over every token, and over the response's tokens only:

[Example: instruction-loss-by-hand](../examples/instruction-loss-by-hand.py)

Of the {{out:instruction-loss-by-hand:tokens predicted in the text}} tokens the model predicts, the instruction loss counts {{out:instruction-loss-by-hand:tokens the instruction loss counts, the response's}}: the description's tokens and its end. The average loss over every token is {{out:instruction-loss-by-hand:average loss over every token}}, and over the response only {{out:instruction-loss-by-hand:average loss over the response only}}. Training on the second number, over many requests and responses, teaches a model to produce responses.

DPO's loss works on a pair of responses to one request, one preferred and one rejected. It compares how much more likely the model being tuned makes each response than the **reference model** it started from, the model after instruction tuning, and rewards it for raising the preferred one more. With log-probabilities log π of each response under the tuned model and log π_ref under the reference, and a factor β that sets how far the model may move:

margin = β × [(log π(preferred) − log π_ref(preferred)) − (log π(rejected) − log π_ref(rejected))]

loss = −log σ(margin)

with σ the logistic function from [chapter 4](#learn-by-predicting-neighbours) ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290), equation 7). For the store, follow the arithmetic on made-up log-probabilities, with β = {{out:dpo-loss-by-hand:beta}}:

[Example: dpo-loss-by-hand](../examples/dpo-loss-by-hand.py)

When the tuned model favours the preferred response more than the reference does, the margin is {{out:dpo-loss-by-hand:tuned model favours the preferred answer more than the reference does, margin}} and the loss {{out:dpo-loss-by-hand:tuned model favours the preferred answer more than the reference does, loss}}; when it favours it less, the margin is {{out:dpo-loss-by-hand:tuned model favours it less, margin}} and the loss rises to {{out:dpo-loss-by-hand:tuned model favours it less, loss}}. Lowering the loss pushes the preferred response's probability up and the rejected one's down, measured against where the model started.

**What to do:** mask the request out of the loss when you tune a model on requests and responses, and keep the reference model's log-probabilities beside the tuned model's when you train on preferences. Both are small changes to a loop you already have.

## Ask the base model and the tuned model {#ask-base-and-tuned}

The store needs descriptions for products that have none, the job [chapter 9](#choosing-the-next-token) gave its own small model. The base model, SmolLM2-360M, can't be asked, only continued, so it gets the start of a product listing: `Product:`, the name, and `Description:`. The tuned model, SmolLM2-360M-Instruct, gets a request: `Write a short product description for:` and the name.

For the store, read what each model wrote for [chapter 9](#choosing-the-next-token)'s product, `front porch welcome sign wall décor`, and measure all {{out:base-and-tuned:products described, none of which has a description to draw on}} products the recording described, each without a description to draw on:

[Example: base-and-tuned](../examples/base-and-tuned.py)

For the welcome sign, the base model writes about front porches: `The front porch is a common feature of many homes`, then about relaxing and meeting friends. It continues the text in a plausible direction, which isn't the one the store needs. The tuned model writes something shaped like a product description, `a statement piece that showcases your personality and style`, but calls the sign `a space` and `This wall`: fluent about a product it doesn't understand. Across the products, the base model's texts repeat {{out:base-and-tuned:base model, repeated four-word phrases}} of their four-word phrases, and the tuned model's {{out:base-and-tuned:tuned model, repeated four-word phrases}}. The recording can't say which step of tuning stopped the loops.

Neither model knows anything about the products beyond their names. The base model named a material the name doesn't give in {{out:base-and-tuned:base model, descriptions that name a material the name doesn't}} descriptions, and the tuned model in {{out:base-and-tuned:tuned model, descriptions that name a material the name doesn't}}, and each of those came from the model, not from the store's data, as [chapter 9](#choosing-the-next-token)'s did.

**What to do:** use a tuned model when you need a request followed, and a base model when you need text continued in the style of what came before. Don't read fluency as knowledge: a tuned model writes a better-shaped description from the same lack of facts.

## Format the request as the model was tuned {#format-the-request}

The tuned model answered in the [previous section](#ask-base-and-tuned) because its request came in the layout it was tuned on, its **chat template**: special tokens that mark where a **system message**, instructions placed before the user's request, the request itself and the model's reply begin and end. The model learned to answer requests only in that layout, so a request in any other form is only text to continue. Libraries that load the model apply the template from the model's tokenizer files ([tokenizer configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/tokenizer_config.json)).

For the store, read exactly what the tuned model was given with its template, and compare its reply with the reply to the same request given as plain text:

[Example: chat-template](../examples/chat-template.py)

With the template, the request sits between `<|im_start|>user` and `<|im_end|>`, after a system message the template adds by default, `You are a helpful AI assistant named SmolLM, trained by Hugging Face`, and before `<|im_start|>assistant`, where the model's reply begins. The model replies with the description of the previous section. Without the template, given the same words as plain text, the tuned model writes `{{out:chat-template:tuned model, without it}}` and stops. Tuning didn't teach the model to follow requests in general; it taught it to follow requests in that layout.

The system message matters too: it's part of every request the model reads, so a different one, such as a description of the store and its tone, can change any reply. That makes it a setting to choose and record, like a decoding setting in [chapter 9](#choosing-the-next-token).

**What to do:** always format requests to a tuned model with its own chat template, applied by the library from the model's tokenizer files, never assembled by hand. Check the template's default system message, since the model reads it on every request, and record the full formatted prompt with every answer you keep.

## Measure made-up product details {#measure-made-up-details}

A model that writes about a product with no facts has to make them up. The fairer test gives it the facts. Both SmolLM2 models are shown a product's name and real description, and asked: `What material is this product made of? Answer in a few words, using only the description.` The tuned model gets it as a request in its template; the base model gets it as a listing to continue, ending `Question:`, the question, and `Answer:`. If the product's data names a material, a good answer names it; if not, a good answer says the description doesn't say.

Checking the answers takes a list of material words, from `wood` and `metal` to `polyester` and `leather`, in the book's shared module `lib.claims`. An answer is **supported** if every material it names is in the product's name or description, **made up** if it names one that isn't, **declined** if it says the data doesn't say, and **none** if it names no material and doesn't decline. The check has limits, and makes judgement calls: it counts engineered wood such as `mdf` and `rubberwood` as wood, it counts `brushed nickel finish` as naming nickel, and it misses a material the list doesn't include. Every answer it classes is recorded for anyone to read. Making up a detail the input doesn't support is one kind of what's called **hallucination** ([Ji et al., 2023](https://arxiv.org/abs/2202.03629)), and this test measures exactly that kind.

For the store, class each model's answers for {{out:made-up-materials:products asked, each with its description given}} products drawn at random, split by whether the product's data names a material. Before you run it, predict whether the tuned model makes up fewer materials than the base model where the data names none:

[Example: made-up-materials](../examples/made-up-materials.py)

| Answer | Base, data names a material | Tuned, data names a material | Base, data names none | Tuned, data names none |
| --- | --- | --- | --- | --- |
| Supported | {{out:made-up-materials:base model, data names a material, supported}} | {{out:made-up-materials:tuned model, data names a material, supported}} | {{out:made-up-materials:base model, data names none, supported}} | {{out:made-up-materials:tuned model, data names none, supported}} |
| Made up | {{out:made-up-materials:base model, data names a material, made up}} | {{out:made-up-materials:tuned model, data names a material, made up}} | {{out:made-up-materials:base model, data names none, made up}} | {{out:made-up-materials:tuned model, data names none, made up}} |
| None | {{out:made-up-materials:base model, data names a material, none}} | {{out:made-up-materials:tuned model, data names a material, none}} | {{out:made-up-materials:base model, data names none, none}} | {{out:made-up-materials:tuned model, data names none, none}} |
| Declined | {{out:made-up-materials:base model, data names a material, declined}} | {{out:made-up-materials:tuned model, data names a material, declined}} | {{out:made-up-materials:base model, data names none, declined}} | {{out:made-up-materials:tuned model, data names none, declined}} |

Of the products, {{out:made-up-materials:products whose name or description names a material}} have data that names a material, and on those both models mostly answer from it. On the {{out:made-up-materials:products whose name and description name none}} that don't, the right answer is that the description doesn't say, and neither model gave it once. Both mostly made a material up instead, and the rest of their answers named none without saying why.

Instruction tuning didn't fix this, and on these products the base model, given a listing to continue, did no worse. The request said to use only the description; when the description had no answer, the models answered anyway, with a material that products like this one often have. Following the form of an instruction isn't the same as knowing when you don't know.

**What to do:** test a model on questions its input can't answer, not only on ones it can, and count made-up answers separately. Check every factual claim a model makes about a product against the product's data, automatically where you can, as `lib.claims` does for materials.

## Check whether confidence warns you {#check-confidence}

The [previous section](#measure-made-up-details) classed the tuned model's answers to the material question as supported or made up. If its made-up answers came with low confidence, low confidence could flag them. A model whose confidence matches how often it's right is **calibrated**, and Guo and colleagues found that the deep neural networks they studied often aren't: they were more confident than their accuracy warranted ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)). The recording saved the probability the tuned model gave each token it wrote, its end token included, so a confidence per token can be computed for each answer, as a geometric mean:

confidence per token = exp((log p₁ + … + log pₙ) / n)

the exponential of the average log-probability, the same average that perplexity inverts, so a short answer and a long one are measured on the same scale.

For the store, compare the tuned model's confidence in its supported and made-up answers, and count how many of each a threshold of 0.5 per token would keep:

[Example: answer-confidence](../examples/answer-confidence.py)

The made-up answers are less confident on average, {{out:answer-confidence:tuned model, made up answers, average probability per token}} per token against {{out:answer-confidence:tuned model, supported answers, average probability per token}} for supported ones. But a threshold of 0.5 per token would keep {{out:answer-confidence:tuned model, supported answers above 0.5 per token}} supported answers and still let through {{out:answer-confidence:tuned model, made up answers above 0.5 per token}} made-up ones. Confidence is a signal, not a filter: it shifts the odds, and it lets confident errors through. With so few answers, these are directions, not rates to rely on.

**What to do:** don't use a model's confidence alone to decide whether to trust an answer. Measure how its confidence relates to being right on your own questions, as here, and if you use a threshold, choose it from those measurements and count what it lets through.

## Find what a longer context loses {#find-what-context-loses}

The [material question](#measure-made-up-details) gave the model one product at a time. A store's assistant often has several products' data in front of it at once, such as search results, a category page or a comparison, and has to find the right product's details among the others. Liu and colleagues found that models use information at the start and the end of a long context better than information in the middle ([Liu et al., 2024](https://arxiv.org/abs/2307.03172)).

For the store, give the tuned model, SmolLM2-360M-Instruct, {{out:answer-position:product listings in each context}} product listings, the asked product among them, first, in the middle or last, and ask what that product is made of. Count the answers that name only its own materials, as `lib.claims` checks them, for the {{out:answer-position:products asked about at each place}} products whose data names one:

[Example: answer-position](../examples/answer-position.py)

With the product first, {{out:answer-position:product placed first, answers that name only its own materials}} answers name only its materials; last, {{out:answer-position:product placed last, answers that name only its own materials}}; in the middle, {{out:answer-position:product placed middle, answers that name only its own materials}}. A product in the middle is found least often, the pattern Liu and colleagues found in much larger models. The counts are small, but the gap is wide. The context here is short, a few thousand characters, well within the model's limit: the middle loses details long before a context is full.

**What to do:** put the details a model most needs at the start or the end of its context, not in the middle, and give it as little unrelated text as the task allows. Test retrieval from a longer context at each position, as here, before trusting a model to find details among many.

## Adapt a model to your text {#adapt-a-model}

[Chapter 10](#inside-an-open-model) found the general model far worse than the store's small model on the store's names, per word. Instruction tuning changes how a model answers, not what it knows about one store. The way to teach it the store's text is **fine-tuning**: continue training it on that text, the same training loop as [chapter 8](#update-with-adamw), starting from the general model's weights instead of random ones.

Full fine-tuning updates every weight, which for a model of hundreds of millions of parameters takes time and memory. **LoRA**, low-rank adaptation, freezes the model's weights and trains a small correction beside some of them. For a weight matrix that maps k numbers to d numbers, it trains a matrix B of d by r and a matrix A of r by k, and adds their product to the weights: W + BA ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)). With a small rank r, there's far less to train and store, and Hu and colleagues found it matched full fine-tuning on the tasks they tested.

For the store, count what LoRA would train on SmolLM2-360M, adapting the query and value weights of every layer, at two ranks, against full fine-tuning. The query weights map 960 numbers to 960; the value weights, because keys and values are grouped, map 960 to 320:

[Example: lora-parameters](../examples/lora-parameters.py)

At rank 8, LoRA trains {{out:lora-parameters:rank 8, parameters trained}} parameters, {{out:lora-parameters:rank 8, share of the model's parameters}} of the model; at rank 64, {{out:lora-parameters:rank 64, parameters trained}}, or {{out:lora-parameters:rank 64, share of the model's parameters}}. Full fine-tuning trains all {{out:lora-parameters:parameters trained by full fine-tuning}}. A rank-8 correction is smaller than the store's own small model, and it can be stored and swapped per task while the base weights stay shared.

This book doesn't run the fine-tuning: our judgement is that even LoRA, over the store's text, would take far longer on a laptop's processor than any recording here. It's the next step for a store that wants a general model to write in its own voice: fine-tune on the catalog, then measure the result against the store's own model, per word, as [chapter 10](#inside-an-open-model) did. Fine-tuning has risks of its own. A model trained on the catalog learns its gaps along with its text, as the store's small model learned to leave a bulb's description empty in [chapter 9](#forbid-an-empty-description). And training on one narrow text can weaken what the model knew before; Biderman and colleagues found LoRA forgets less of it than full fine-tuning ([Biderman et al., 2024](https://arxiv.org/abs/2405.09673)). Both are measured the same way as everything else in this book: on held-out text, before and after.

**What to do:** to teach a general model your text, fine-tune it on that text, with LoRA when memory or storage is tight, and measure the result on held-out text of the same kind. Don't expect instruction tuning or a cleverer prompt to supply knowledge the model never learned.

## Look back at the yardstick {#look-back-at-the-yardstick}

This book has scored every model the same way on names, from [chapter 3](#counting-further-back) on: perplexity per token of the store's tokenizer, on the store's validation names. The test names were kept aside for one final comparison, so that choices made by looking at the validation names, such as the kept epochs and the mixing shares, couldn't flatter the results. Now they're used. For the store, score the five models that were chosen between on the {{out:test-names:test names}} test names, with the mixing shares chosen on validation:

[Example: test-names](../examples/test-names.py)

Set side by side, the scores tell the book's story in one table, and the test column agrees with the validation one to within a few tenths: the choices made on validation flattered the results very little.

| Model | Chapter | Validation names | Test names |
| --- | --- | --- | --- |
| Counts of single tokens, a 1-gram | [3](#smooth-with-kneser-ney) | {{out:kneser-ney-by-order:1-gram validation perplexity}} | |
| Counts of pairs, a 2-gram | [3](#smooth-with-kneser-ney) | {{out:kneser-ney-by-order:2-gram validation perplexity}} | |
| Counts of four tokens, the 4-gram baseline | [3](#the-baseline) | {{out:kneser-ney-by-order:4-gram validation perplexity}} | {{out:test-names:test perplexity, the 4-gram baseline}} |
| The window model, a small neural network | [5](#compare-with-the-baseline) | {{out:neural-vs-baseline:window model validation perplexity}} | {{out:test-names:test perplexity, the window model}} |
| The window model mixed with the counts | [5](#mix-with-the-counts) | {{out:mix-neural-and-counts:validation perplexity, network share 0.2}} | {{out:test-names:test perplexity, the window model mixed with the counts, share 0.2}} |
| The store's transformer | [8](#compare-on-names) | {{out:transformer-on-names:validation perplexity on names, transformer share 1.0}} | {{out:test-names:test perplexity, the store's transformer}} |
| The store's transformer mixed with the counts | [8](#compare-on-names) | {{out:transformer-on-names:validation perplexity on names, transformer share 0.5}} | {{out:test-names:test perplexity, the store's transformer mixed with the counts, share 0.5}} |

Three lessons run through it. Counting goes a long way on short, formulaic text: no learned model in this book beat the 4-gram baseline on names alone, and both mixes beat it. A learned model earns its place by being wrong in different places from the counts, which is why every mix beat both its parts. And on longer text, where distant context matters, the transformer pulled ahead on its own, as [chapter 8](#compare-on-descriptions) measured on descriptions. Size alone didn't decide anything: [chapter 10's](#score-the-store-text) general model, hundreds of times larger, scored {{out:open-model-on-the-store:names, per-word perplexity, SmolLM2-360M}} per word on the store's names, against the store's transformer's {{out:open-model-on-the-store:names, per-word perplexity, the store's model}}.

None of these numbers says whether a model is useful. Perplexity measures how well a model predicts text like its training text, which is what autocomplete needs. It doesn't measure whether a description is true, whether an answer follows a request, or whether a shopper finds what they came for. This chapter's tests, made-up materials, confidence and the place of a fact in a context, are a first step past perplexity, toward measuring the job itself.

**What to do:** keep one fixed yardstick for comparing models, the same text, tokenizer and measure every time, as this book did. Then measure the job itself, with tests built for it, before choosing a model to serve it: a lower perplexity is a reason to test a model, not a reason to ship it.

## What goes wrong in production {#instruction-models-in-production}

A tuned model is the kind people can use without help, and its failures sound as fluent as its successes:

- **It answers when it should say it doesn't know.** The base and the tuned SmolLM2 models made up a material for most products whose data named none, and neither ever said the data didn't say. Check every factual claim against the product's data before it reaches a shopper.
- **Its confidence lets errors through.** Made-up answers were less confident on average, but a threshold that kept most good answers still let {{out:answer-confidence:tuned model, made up answers above 0.5 per token}} made-up ones through.
- **Its format is part of its training.** Without its chat template, the tuned model wrote a full stop. Apply the template from the model's files, and record the full prompt with every answer.
- **The middle of a context gets lost.** Put the details that matter at the start or the end, and test retrieval at each position.
- **It doesn't know your text.** Instruction tuning teaches the form of answers, not a store's catalog. Fine-tune on the catalog, and measure.
- **Fluency hides gaps.** The tuned model's descriptions read better than the base model's and contain no more facts. Judge generated text by checks on its claims, not by how it reads.
- **Its behaviour depends on its revision and its template's default messages.** Pin the model's revision, as the recording does, and rerun these measurements after any update.

## What this chapter decided {#instruction-decisions}

- **The book compares base and tuned models on the same requests,** recorded once at pinned revisions and decoded greedily, and measures their answers with checks anyone can rerun, such as `lib.claims` for materials.
- **Made-up details are measured on questions the input can't answer** as well as ones it can. On the store's products, both SmolLM2 models made up a material for most products whose data named none, and neither declined once, so generated product facts are always checked against the product's data.
- **A model's confidence is a signal, not a filter.** It's used only with a threshold measured on the store's own questions, and the errors it lets through are counted.
- **The details a model needs go at the start or the end of its context,** never buried among other products.
- **Perplexity stays the book's yardstick for comparing models,** on the same text, tokenizer and events every time, and the job each model is chosen for is measured with tests built for that job.
- **Teaching a general model the store's text means fine-tuning,** measured per word against the store's own model; instruction tuning changes how a model answers, not what it knows. Measuring a model inside an application, beyond the next-token tests of this book, is a subject of its own.
