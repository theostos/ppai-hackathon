# Optimizing LLM FHE Inference with Custom Mask Strategy

---

## The Custom Attention Mask Strategy

### Introduction to the Attention Mechanism

In transformer models, the attention mechanism plays a critical role in how each token (word or sub-word) in the input relates to others. This mechanism uses **Queries (Q)**, **Keys (K)**, and **Values (V)** to compute attention scores, which determine the relevance of each token to others.

The **attention mask** is a matrix that determines which tokens can attend to which others—commonly used for causal attention by blocking future tokens in language models.

### The Custom Mask Strategy

In this proposed strategy, clear tokens (non-sensitive) only attend to other clear tokens, which means all key values associated with them are computed without involving FHE (so remains in clear in the whole computation). This significantly reduces the need for FHE computations. 

Instead of the usual lower triangular matrix for causal attention, this strategy introduce “holes” in the mask, where only masked (sensitive) tokens can attend to each other.

Below is an illustration of the **custom attention mask**:

For the sentence 

`[I] [am] [Alice,] [and] [you] [?]`

For the standard causal mask, it would have been:

```[1, 0, 0, 0, 0, 0, 0]
[1, 1, 0, 0, 0, 0, 0]
[1, 1, 1, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 0, 0]
[1, 1, 1, 1, 1, 1, 0]
[1, 1, 1, 1, 1, 1, 1]
```
Basically, each tokens attends all its predecessors and itself.

Now if we mask the token [Alice,] with the custom mask strategy:

```
[1, 0, 0, 0, 0, 0, 0] <-- [I] only attends to itself
[1, 1, 0, 0, 0, 0, 0] <-- [am] attends to its predecessor and itself
[1, 1, 1, 0, 0, 0, 0] <-- [Alice,] attends all its predecessors and itself
[1, 1, 1, 0, 1, 0, 0] <-- [and] does not attends the secret token (clear token)
[1, 1, 1, 0, 1, 1, 0] <-- [you] does not attends the secret token (clear token)
[1, 1, 1, 1, 1, 1, 1] <-- [?] attends all its predecessors and itself since its the last token, because we want a full flow of information for the next token prediction task.
```
In this custom mask:
- **Clear tokens**: Attend only to previous clear tokens.
- **FHE tokens**: Attends to all (previous) tokens, which allows the full sentence context for modelisation.
- **Last token**: Always attends to all tokens, which allows the full sentence context for prediction.

### Challenges in the Flow of Information

The restriction introduced by this mask strategy has a major impact on the flow of information in the attention mechanism.
In the case of multiple masked tokens, the situation is actually very bad, and a simple question such as "What is the `<sensitive>capital</sensitive>` of `<sensitive>Tonga</sensitive>`?" gives bad results.

The model struggles with this reduced capacity for propagation of information.

### Evaluating the Impact

To understand how the change in attention mask would affect an already trained LLM, we created evaluation datasets by modifying existing ones, such as **BoolQ** and **MMLU**. We used a language model to “obfuscate” sensitive parts of these datasets, marking them as `<sensitive>`, which forces them to be treated with FHE in our approach. 

#### Example from (mini-z)BoolQ:

- **Original context**:  
  "The Land Rover Range Rover (generally known simply as a Range Rover) is a full-sized luxury sport utility vehicle (SUV) from Land Rover, a marque of Jaguar Land Rover. The Range Rover was launched in 1970 by British Leyland. This flagship model is now in its fourth generation."

- **Original question**: 
  "is range rover and land rover the same company"

- **Obfuscated context**:  
  "The `<sensitive>Land Rover Range Rover</sensitive>` (generally known simply as a `<sensitive>Range Rover</sensitive>`) is a `<sensitive>full-sized luxury sport utility vehicle</sensitive>` (SUV) from `<sensitive>Land Rover</sensitive>`, a marque of `<sensitive>Jaguar Land Rover</sensitive>`. The `<sensitive>Range Rover</sensitive>` was launched in `<sensitive>1970</sensitive>` by `<sensitive>British Leyland</sensitive>`. This `<sensitive>flagship model</sensitive>` is now in its `<sensitive>fourth generation</sensitive>`."

- **Obfuscated question**: 
  "is `<sensitive>Range Rover</sensitive>` and `<sensitive>Land Rover</sensitive>` the same company?"

### Mitigating the Flow Issue

Initial results on benchmarks like BoolQ and MMLU were negative (see [ReadMe](https://github.com/theostos/ppai-hackathon/blob/main/README.md)), likely due to the disrupted flow of data. To address this, we fine-tuned the model on two datasets:
- **SQuAD** (text comprehension in a QA format)
- **SlimOrca** (instruction tuning dataset)

These datasets were converted to be compatible with the custom mask by marking sensitive words (same technique than before), allowing us to do training.

You can view examples of these datasets and experiment with the finetuned model on [Hugging Face](https://huggingface.co/theostos) and on [this Hugging Face space](https://huggingface.co/spaces/ppaihack/zLlamaskClear).

---

## 2. Expected Gain

### Initial Gain Estimate

With our naive implementation, the expected gain is **inversely proportional to the number of sensitive tokens**. For instance, if **1/5th** of the tokens are sensitive (masked), the number of FHE computations required is reduced to just 1/5th. Given that clear inference can be processed on GPUs (effectively "free" relative to FHE), this leads to an approximate **5x speed-up**.

### Further Potential Improvement

A additionnal significant improvement can be achieved with an additional observation:

For masked tokens, once the keys for clear tokens are computed, there are two types of dot products to compute:
1. **FHE-Token Queries vs. FHE-Token Keys**: Requires FHE computation for both queries and keys (1/5th of computations, if 1/5th of tokens are FHE).
2. **FHE-Token Queries vs. Clear-Token Keys**: Requires FHE computation for queries but not for keys (4/5th of computations, if 1/5th of tokens are FHE).

Since FHE-queries against clear keys are far less demanding than FHE-queries against FHE-keys, the overall computational complexity reduces significantly. For a scenario where **20% of tokens are masked**, this leads to a cumulated improvement of approximately **25x**.

Thus, in a generalized case where **1/N** tokens are masked, the **expected gain is on the order of N²** (e.g., **20% masked** -> roughly **25x** improvement).

---

## Current limitation and perspective

Even if the first results are very encouraging, it should be tests with more aggressive training (to iterate fast I had to do very low rank LoRA training). I only trained Llama 3.1 8b-instruct, but it would be also interesting to see the effect of scaling of models: there is way smaller and way bigger models to test. I used this intermediate size model since smaller models gives bad results on MCQ datasets, so it would have been very hard to evaluate efficiently the trained models.

It would also be interesting to find a practical use case where LLM are used non auto regressively (classification/decision, embedding etc.).
