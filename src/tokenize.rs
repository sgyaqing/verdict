//! Sequence construction: tokenizer + layout loading and the family dispatch.
//! The rendering/truncation recipe itself lives in the family profile
//! (`crate::family`); this module owns the generic machinery (tokenizer
//! access, special-token resolution, manifest layout) that recipes share.
//!
//! Special-token names are externalized to model.json so a differently named
//! tokenizer loads unchanged; ids are resolved from the tokenizer at load
//! time, never hardcoded. A wrong name is not a silent failure: the vocab
//! lookup misses and `load` fails loudly.

use crate::family::letter_readout::MAX_OPTIONS;

use std::path::Path;

use thiserror::Error;

use crate::family::Family;

#[derive(Debug, Error)]
pub enum TokenizeError {
    #[error("failed to load tokenizer: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("special token {0:?} missing from the tokenizer vocab")]
    MissingSpecial(String),
    #[error("invalid model.json: {0}")]
    ConfigParse(#[from] serde_json::Error),
    #[error("model.json is missing field {0}")]
    MissingConfig(&'static str),
    #[error("invalid model.json: {0}")]
    Family(String),
    #[error("letter table: {0}")]
    LetterTable(String),
}

/// Token ids and length budgets pinned by the golden corpus meta.
/// Specials are resolved from the tokenizer at load time, never hardcoded.
#[derive(Debug, Clone)]
pub struct Layout {
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub pad_id: u32,
    /// Serving sequence ceiling: the padding bound and (for the
    /// independent-options family) the state-fit window.
    pub max_len: usize,
    /// Head budget for the marker-head family's shrink path; the
    /// independent-options family has no head budget (value = max_len).
    pub head_max_len: usize,
    /// The mask token's surface form; recipes use it both as the option
    /// marker and for user-text hygiene.
    pub mask_literal: String,
    /// The separator token's surface form (packing and hygiene).
    pub sep_literal: String,
    /// Sequence-length alignment for the collate pad (model.json's
    /// `seq_align`; default 1 = no alignment). Some execution providers
    /// require padded lengths to be a multiple of the graph's internal
    /// chunk size (e.g. Qwen3.5 linear attention chunks by 64).
    pub seq_align: usize,
    /// Fixed-shape serving classes (model.json's `fixed_shapes` keys,
    /// ascending; empty = dynamic geometry). The collate pads each batch
    /// to the smallest class that fits and the engine runs the matching
    /// static-S session.
    pub fixed_classes: Vec<usize>,
}

#[derive(Clone)]
pub struct TokenizerBundle {
    pub tok: tokenizers::Tokenizer,
    pub layout: Layout,
    /// Which family recipe builds sequences for this model (model.json's
    /// `family` key; required, and a missing or unknown value fails the load).
    pub family: Family,
    /// Space out every digit run before encoding. **Only the independent-options
    /// recipe reads it** -- the key is not consulted at all for the other
    /// families, so their manifests never carry it (see the family-dispatched
    /// read in the builder below).
    pub digit_split: bool,
    /// The letter-readout family's 255 single-token option labels, derived
    /// from this tokenizer at load (empty for the marker families, which
    /// have no letter head).
    pub letter_table: Vec<u32>,
}

/// A built request sequence: model input ids (right-padding happens at collate
/// time), `[MASK]` marker positions (one per option), and the uncapped rendered
/// token count (protocol-layer usage.input_tokens).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltSequence {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub full_len: usize,
    /// Token count of the (fitted) state text, no specials -- the
    /// formula-temperature feature input; unused by the band-table family.
    pub state_tokens: usize,
    /// Letter-readout family: option count per slot (one slot per row here).
    /// The engine's row for a slot is the first `nopts` entries of the
    /// 255-wide letter row; `real_k` trims to exactly that.
    pub nopts: Vec<usize>,
}

impl BuiltSequence {
    /// How many real logits the engine returns for this sequence: one per
    /// marker (marker dialects), or the slot's option count (letter readout).
    pub fn real_k(&self) -> usize {
        if self.nopts.is_empty() {
            self.markers.len()
        } else {
            self.nopts.iter().sum()
        }
    }
}

/// Derive the 255 single-token option labels (official label_table): the
/// upper-case names A..Z, AA, AB, ... keeping every name that encodes to
/// exactly one token, in that order, until the table is full. Derived from
/// the tokenizer itself -- never hardcoded -- and loud when a tokenizer
/// cannot fill the table.
fn derive_letter_table(tok: &tokenizers::Tokenizer) -> Result<Vec<u32>, TokenizeError> {
    let single = |name: &str| -> Option<u32> {
        let enc = tok
            .encode(name, false)
            .expect("tokenizer encode cannot fail on &str input");
        let ids = enc.get_ids();
        (ids.len() == 1).then(|| ids[0])
    };
    let mut out: Vec<u32> = Vec::with_capacity(MAX_OPTIONS);
    for c in b'A'..=b'Z' {
        if let Some(id) = single(&(c as char).to_string()) {
            out.push(id);
        }
    }
    'outer: for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            if out.len() >= MAX_OPTIONS {
                break 'outer;
            }
            if let Some(id) = single(&format!("{}{}", a as char, b as char)) {
                out.push(id);
            }
        }
    }
    if out.len() < MAX_OPTIONS {
        return Err(TokenizeError::LetterTable(format!(
            "this tokenizer has only {} single-token upper-case letter names; the \
             letter-readout family needs {MAX_OPTIONS}",
            out.len()
        )));
    }
    let mut seen = std::collections::HashSet::new();
    if !out.iter().all(|id| seen.insert(*id)) {
        return Err(TokenizeError::LetterTable(
            "two option labels tokenize to the same id".into(),
        ));
    }
    Ok(out)
}

impl TokenizerBundle {
    /// Load from a model directory holding `tokenizer/tokenizer.json` and
    /// `model.json` (model-family specifics live in the manifest, not
    /// product code). Grading tokenization or calibration needs no weights,
    /// so a directory with just those two files is a valid model dir.
    pub fn load(model_dir: &Path) -> Result<Self, TokenizeError> {
        let tok_path = model_dir.join("tokenizer").join("tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(TokenizeError::Tokenizer)?;
        let cfg_path = model_dir.join("model.json");
        let raw = std::fs::read_to_string(&cfg_path).map_err(|source| TokenizeError::Io {
            path: cfg_path.display().to_string(),
            source,
        })?;
        let cfg: serde_json::Value =
            serde_json::from_str(&raw).map_err(TokenizeError::ConfigParse)?;
        let family = Family::from_manifest(&cfg).map_err(TokenizeError::Family)?;
        let special = |name: &str| {
            tok.token_to_id(name)
                .ok_or_else(|| TokenizeError::MissingSpecial(name.to_string()))
        };
        let num = |key: &'static str| {
            cfg.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or(TokenizeError::MissingConfig(key))
        };
        let max_len = num("max_len")?;
        // Special-token names are required, not defaulted: they fence every
        // row, so a name that disagrees with the tokenizer's own convention
        // would mis-encode silently. The packager fills them in from the
        // tokenizer. WHICH names are required is per family -- see the arms.
        let name = |key: &'static str| -> Result<String, TokenizeError> {
            cfg.get(key)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or(TokenizeError::MissingConfig(key))
        };
        let pad_token = name("pad_token")?;
        // The head budget only exists in the marker-head recipe; the other
        // recipes have no head structure (max_len stands in so downstream
        // budget arithmetic stays total).
        let head_max_len = match family {
            Family::MarkerHead => num("head_max_len")?,
            Family::IndependentOptions | Family::LetterReadout => cfg
                .get("head_max_len")
                .and_then(|v| v.as_u64())
                .map_or(max_len, |v| v as usize),
        };
        let layout = match family {
            // The letter-readout recipe assembles no premise/option structure
            // (its row builder in family/letter_readout.rs never reads
            // mask_literal or sep_literal), so pad_token is the only name it
            // needs; the unused slots hold the pad id.
            Family::LetterReadout => {
                let pad_id = special(&pad_token)?;
                Layout {
                    cls_id: pad_id,
                    sep_id: pad_id,
                    mask_id: pad_id,
                    pad_id,
                    max_len,
                    head_max_len: max_len,
                    mask_literal: pad_token.clone(),
                    sep_literal: pad_token.clone(),
                    seq_align: 1,
                    fixed_classes: Vec::new(),
                }
            }
            _ => {
                let cls_token = name("cls_token")?;
                let sep_token = name("sep_token")?;
                let mask_token = name("mask_token")?;
                Layout {
                    cls_id: special(&cls_token)?,
                    sep_id: special(&sep_token)?,
                    mask_id: special(&mask_token)?,
                    pad_id: special(&pad_token)?,
                    max_len,
                    head_max_len,
                    mask_literal: mask_token,
                    sep_literal: sep_token,
                    seq_align: 1,
                    fixed_classes: Vec::new(),
                }
            }
        };
        // seq_align: manifest-driven; 1 (no alignment) unless the family's
        // graph requires a multiple.
        let seq_align: usize = cfg
            .get("seq_align")
            .and_then(|v| v.as_u64())
            .map_or(1, |v| v.max(1) as usize);
        // fixed_shapes: static-S serving classes. Keys must be integer
        // lengths -- a dropped key would silently narrow the served shapes.
        let mut fixed_classes: Vec<usize> = match cfg.get("fixed_shapes") {
            None => Vec::new(),
            Some(v) => {
                let obj = v.as_object().ok_or_else(|| {
                    TokenizeError::Family(
                        "model.json 'fixed_shapes' must map integer lengths to graph files".into(),
                    )
                })?;
                let mut out = Vec::with_capacity(obj.len());
                for key in obj.keys() {
                    let len: usize = key.parse().map_err(|_| {
                        TokenizeError::Family(format!(
                            "fixed_shapes key {key:?} is not an integer length"
                        ))
                    })?;
                    out.push(len);
                }
                out
            }
        };
        fixed_classes.sort_unstable();
        let layout = Layout {
            seq_align,
            fixed_classes,
            ..layout
        };
        let letter_table = match family {
            Family::LetterReadout => derive_letter_table(&tok)?,
            _ => Vec::new(),
        };
        // Read per family, not for everyone: only the independent-options
        // recipe has a row builder that consults this, so a manifest that
        // never mentions the key is the norm for the other two -- the field
        // just stays false there.
        let digit_split = match family {
            Family::IndependentOptions => cfg
                .get("digit_split")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Family::MarkerHead | Family::LetterReadout => false,
        };
        Ok(Self {
            tok,
            layout,
            family,
            digit_split,
            letter_table,
        })
    }

    /// Encode without special tokens; family recipes add structure tokens
    /// themselves, mirroring the official Python paths.
    pub(crate) fn encode_plain(&self, text: &str) -> Vec<u32> {
        self.tok
            .encode(text, false)
            .expect("tokenizer encode cannot fail on &str input")
            .get_ids()
            .to_vec()
    }

    /// Build EVERY engine row of one task request (official plan_rows): the
    /// marker families render a single sequence; the letter-readout family
    /// renders one row per isolated score level (choice/noul: one row).
    pub fn build_sequences(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
    ) -> Vec<BuiltSequence> {
        match self.family {
            Family::MarkerHead | Family::IndependentOptions => {
                vec![self.build_sequence(task, question, state, options)]
            }
            Family::LetterReadout => {
                crate::family::letter_readout::build_rows(self, task, question, state, options)
            }
        }
    }

    /// Option texts in marker order, per the model's family recipe.
    pub fn render_options(&self, task: &str, options: &[String]) -> Vec<String> {
        match self.family {
            Family::MarkerHead => crate::family::marker_head::render_options(task, options),
            Family::IndependentOptions => {
                crate::family::independent_options::render_options(task, options)
            }
            Family::LetterReadout => crate::family::letter_readout::render_options(task, options),
        }
    }

    /// Build the request sequence per the model's family recipe (single-row
    /// form; see [`TokenizerBundle::build_sequences`] for the multi-row task
    /// entry point the pipeline uses).
    pub fn build_sequence(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
    ) -> BuiltSequence {
        match self.family {
            Family::MarkerHead => {
                crate::family::marker_head::build_sequence(self, task, question, state, options)
            }
            Family::IndependentOptions => crate::family::independent_options::build_sequence(
                self, task, question, state, options,
            ),
            Family::LetterReadout => {
                let rows =
                    crate::family::letter_readout::build_rows(self, task, question, state, options);
                // Loud, not silent: a letter score request is multi-row and
                // must go through build_sequences (the pipeline's path).
                assert!(
                    rows.len() == 1,
                    "letter score requests render one row per level; use build_sequences"
                );
                rows.into_iter().next().expect("checked len above")
            }
        }
    }
}
