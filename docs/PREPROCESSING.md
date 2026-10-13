# Preprocessing decisions

| Issue | Decision | Rationale |
|---|---|---|
| Unicode variants, zero-width characters, irregular whitespace | NFKC normalisation, zero-width characters removed, whitespace collapsed (`normalize_text`) | Identical text must give identical features and embedding-cache keys |
| Case | Lower-cased for tokens and TF-IDF; original case kept for upper-case-ratio features | Case carries style signal (e.g. ALL-CAPS warnings) |
| Identifiers | `getUserProfile` / `get_user.profile` → `get user profile` | Exposes the semantics of tool and parameter names to text models |
| Schema as string, or invalid JSON | Parsed when possible; otherwise kept with an empty `properties` and flagged (`invalid_json`, `schema_valid = 0`) | A malformed schema is itself a failure cause and must not be dropped |
| Non-standard types (`dict`, `float`, `tuple`, `any`, …) | Mapped to JSON-schema types; counted | Real BFCL definitions use Python-style aliases |
| Missing `properties` / `required` | Treated as empty; flagged as a schema issue | Zero-parameter tools are legitimate |
| `required` naming an undefined parameter | Kept and flagged `required_unknown_param` | Inconsistent contracts are a risk signal |
| Exact duplicates (name, description, schema) | Removed | Avoid double counting |
| Near-duplicates (same name and description) | Kept, sharing one `lineage_id` | Prevents train/test leakage through the grouped split |
| Missing numeric features | Training-set median imputation inside the feature builder | Robust; fitted on train only |
| Outliers (very long descriptions or schemas) | Not clipped for models (trees are invariant; linear models use standard scaling); clipped at p98 only for histograms | Extreme values are informative |
| Missing labels | Rows dropped (none in v1.0) | Cannot be used for supervised learning |
| Class imbalance | Class weights + threshold tuning; no SMOTE on text/embedding features | Synthetic interpolations of embeddings do not correspond to real tools |
