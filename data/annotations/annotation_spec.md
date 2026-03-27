# Annotation Specification: Sentiment Classification
**Generated:** 2026-03-26  |  **Agent:** AnnotationAgent v1.0

---

## 1. Task Description

**Goal:** Classify each text sample into one of `2` predefined classes.

**ML use case:** Binary sentiment classification of multi-source text data (movie reviews, book titles). The resulting labels will be used to train and evaluate a text classifier (e.g. DistilBERT fine-tuning).

**Input:** A single text string (review body or book title).
**Output:** One label from the set below.

---

## 2. Label Definitions

### `positive`

**Definition:** The text expresses a favourable, enthusiastic, or satisfied opinion. The author recommends the item or describes it in clearly positive terms. Joy, excitement, admiration, or praise are typical markers.

**Examples:**
1. *"The Boys in the Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin Olympics"*
2. *"This film is a very good movie.The way how the everybody portrayed their roles was great.The story is nice.It tells us about Raj who is in love with Priya.They get married.She later becomes pregnant.But shortly their is a problem.Sadly they wont get the child.Raj later meets Madhu.He bribes her.She later becomes pregnant but she is not married to him.The movie ..."*
3. *"Hyperbole and a Half: Unfortunate Situations, Flawed Coping Mechanisms, Mayhem, and Other Things That Happened"*

### `negative`

**Definition:** The text expresses an unfavourable, critical, or disappointed opinion. The author would not recommend the item or describes clear flaws, boredom, or frustration.

**Examples:**
1. *"Hearing such praise about this play, I decided to watch it when I stumbled across it on cable. I don't see how this "elivates" women and their "struggles" by focusing on the topic at hand. I guess if you have an interest in stories about women's private parts and how it affects their lives, then this is for you. Otherwise, ..."*
2. *"I'm a Jean Harlow fan, because she had star quality. I don't think her movies are good and I don't even think that she was a good actress, but she certainly was Great in comedies. Every bit of comedy in The Girl from Missouri is very good. But this movie is perhaps more like a love story. Jean Harlow is ..."*
3. *"Poses for Artists Volume 1 - Dynamic and Sitting Poses: An Essential Reference for Figure Drawing and the Human Form"*

---

## 3. Edge Cases

| Scenario | Recommended label | Notes |
|---|---|---|
| Sarcasm / irony | `negative` | Treat the intended meaning, not the literal words. |
| Mixed positive + negative | majority sentiment | Label the dominant tone of the text. |
| Very short text (1–3 words) | most likely label | Low confidence — flag for review. |
| Lists without commentary | `negative` (default) | No sentiment signal present. |
| All-caps emotional text | as written | Intensity doesn't change valence. |
| Foreign-language text | `negative` (default) | Cannot be reliably classified. |

---

## 4. Annotation Guidelines

1. **Read the full text** before deciding — first and last sentences often carry the main sentiment.
2. **Ignore topic** — you are labelling *sentiment*, not topic or quality of writing.
3. **Avoid anchoring** — do not let previous labels influence the current decision.
4. **Flag uncertainty** — if you are less than 70% confident, mark the example for expert review (use the LabelStudio 'flag' button).
5. **Target rate:** aim to annotate 30–50 examples per hour.
6. **Inter-annotator check:** a sample of 10% will be double-annotated to measure Cohen's κ. Target κ ≥ 0.70.

---

## 5. LabelStudio Setup

1. Import `labelstudio_import.json` via **Import** in LabelStudio.
2. Copy the content of `labelstudio_config.xml` into your project's **Labeling Interface** editor.
3. Pre-annotations are loaded as *predictions* — you can accept, reject, or correct each one.

---

*Document generated automatically by AnnotationAgent on 2026-03-26.*