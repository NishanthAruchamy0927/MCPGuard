# Feature engineering

Implementation: `src/features/risk.py`, `src/features/build.py`, `src/preprocessing/*`.

## Families
| Family | Dim. | Description |
|---|---|---|
| `text` | 64 | TF-IDF word 1–2-grams (4,000 features, English stop words) and char_wb 3–5-grams (3,000) on the tool document, reduced with TruncatedSVD (LSA). Components are labelled with their top terms in explanations |
| `emb` | 48 | all-MiniLM-L6-v2 (384-d, L2-normalised) embedding of `"<name words>: <description> Parameters: <name (type, required): description; …>"`, reduced with PCA fitted on train |
| `schema` | 26 | Parameter counts, nested parameters, required/optional counts and ratio, type count and entropy, untyped parameters, nesting depth, object/array/enum/nullable counts, constrained-parameter ratio, defaults, examples, documented-parameter ratio, parameter-description length, schema issues and validity, schema size, documented-required ratio, hinted-parameter ratio |
| `lexical` | 25 | Length, tokens, sentences, average word length, vocabulary, lexical diversity, stop-word ratio, punctuation / special-character / upper-case density, digits, vague / instruction / constraint term counts, examples, format hints, instruction and constraint density, example presence, parameters mentioned in the description, and tool-name features (word count, length, verb prefix, generic, namespaced) |
| `risk` | 6 | Engineered scores (below) |
| `tfidf` | 7,000 | Sparse TF-IDF, used only by the classic baselines E1/E2 |

## Engineered risk scores (all clipped to [0, 1])
Notation: *T* = description tokens, *S* = sentences, *v* = vague terms, *I*/*C* = instruction/constraint terms, *P* = all (nested) parameters, *D* = schema depth, *E* = enum values, *H* = parameter-type entropy (bits), *h* = parameters with enum/pattern/format/bounds/default/examples or a format hint in their description.

```
ambiguity_score            = clip(5·v/max(T,1) + 0.5·[T < 6])
missing_constraint_score   = 1 − h/P                        (0 if P = 0)
parameter_complexity_score = 0.5·ln(1+P)/ln(21) + 0.3·min(D/4,1) + 0.2·(#object + #array)/max(P,1)
schema_complexity_score    = 0.4·min(P/15,1) + 0.3·min(D/4,1) + 0.15·min(E/30,1) + 0.15·min(H/2.5,1)
description_quality_score  = mean( min(T/40,1), described_ratio, 1 − ambiguity, min(examples,1), min(C/3,1) )
instruction_clarity_score  = 0.5·min((I+C)/6,1) + 0.25·min(S/3,1) + 0.25·(1 − ambiguity)
instruction_density        = I / T,    constraint_density = C / T
```

The lexicons (vague, instruction and constraint terms) are listed in `src/preprocessing/text.py`.

## Dimensionality reduction
- **LSA (TruncatedSVD)** on TF-IDF and **PCA** on embeddings are fitted on the training fold only, inside every CV fold and the final fit (`embedding_config.json` records the explained variance).
- **UMAP** (cosine, 15 neighbours) and 2-D PCA are used for visualisation only (`artifacts/metrics/projections.json`, figures 06/07).
