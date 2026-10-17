//! Family profile registry: the isolated seam where model-family recipes
//! live (family specifics belong in the model
//! directory's manifest and this module, never in product vocabulary).
//!
//! A family profile is the complete recipe for turning protocol-layer
//! requests into model inputs and raw logits into served values:
//! sequence rendering/truncation, option/criteria composition, calibration
//! schema, and response rounding. It is selected by the manifest's
//! `family` key, which is required: a manifest that omits it is rejected at
//! load, as is an unknown name -- never a silent fallback to the wrong
//! recipe.
//!
//! Manifest keys (all family-specific beyond the shared `family`/`graph`/
//! budgets/special-token names; each profile defines which it reads):
//!
//! ```json
//! {
//!   "family": "marker_head",
//!   "graph": "model.onnx",
//!   "max_len": 512, "head_max_len": 192,
//!   "temperature": [1.6, 1.2, 1.9],
//!   "temperature_by_options": {"choice:2": 1.9}
//! }
//! ```
//!
//! The independent-options profile additionally reads:
//!
//! ```json
//! {
//!   "family": "independent_options",
//!   "max_len": 1024,
//!   "sliding_window": 64,
//!   "calibration_formula": {"bias": 2.0, "entropy": -3.2, "log_tokens": 11.5,
//!                            "n_options": -3.6, "lo": 0.3, "hi": 12.0},
//!   "noul_band": {"edge": 0.8, "slope": 0.1},
//!   "noul_prior": {"a": -0.5, "b": 0.3}
//! }
//! ```
//!
//! The letter-readout profile (decoder backbones scored at single answer
//! slots) reads:
//!
//! ```json
//! {
//!   "family": "letter_readout",
//!   "max_len": 32768,
//!   "pad_token": "<|endoftext|>",
//!   "temperature": 1.145,
//!   "temperature_by_type": {"choice": 1.164, "noul": 1.624, "score": 1.124}
//! }
//! ```

pub mod independent_options;
pub mod letter_readout;
pub mod marker_head;

/// Which family recipe a model.json declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// The marker-head recipe: task-prefixed question head, per-option `[MASK]`
    /// spans under a shared attention field, marker-gather + qtype graph
    /// inputs, band-table temperature calibration. Re-typed from this family's
    /// official Python implementation; see `marker_head`.
    MarkerHead,
    /// The independent-options recipe: premise packing, block-diagonal
    /// attention (an option attends to premise + itself only), per-option
    /// position-id restart, additive 4D mask graph inputs, formula
    /// temperature calibration, noul commit band. Re-typed from this family's
    /// official Python SDK; see `independent_options`.
    IndependentOptions,
    /// The letter-readout recipe (decoder backbones): one typed question per
    /// row at a single "Answer: (" slot, option-letter logits, by-type
    /// temperature, isolated score levels, prefix-counted usage. Re-typed
    /// from this family's official Python SDK; see `letter_readout`.
    LetterReadout,
}

impl Family {
    /// Parse the manifest's `family` key. Required: a missing value fails the
    /// load instead of falling back to the v0.1 recipe, which would serve a
    /// package through a layout it was not built for (silently, since the
    /// layout decides how rows are assembled).
    pub fn from_manifest(cfg: &serde_json::Value) -> Result<Self, String> {
        match cfg.get("family").and_then(|v| v.as_str()) {
            Some("marker_head") => Ok(Family::MarkerHead),
            Some("independent_options") => Ok(Family::IndependentOptions),
            Some("letter_readout") => Ok(Family::LetterReadout),
            Some(other) => Err(format!(
                "model.json declares unknown family {other:?} (this build knows: \
                 \"marker_head\", \"independent_options\", \"letter_readout\")"
            )),
            None => Err(
                "model.json is missing the 'family' field (one of \"marker_head\", \
                 \"independent_options\", \"letter_readout\")"
                    .to_string(),
            ),
        }
    }
}
