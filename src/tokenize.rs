//! Sequence construction: prompt rendering + encoding + collate constants.
//!
//! Every rule is re-typed from rl_common.py (build_sequence / render_options /
//! serialize_state) and pinned by the golden vectors; authoritative facts
//! live in each golden file's `meta`
//! block.
//!
//! Special-token names are externalized to model.json (DEFAULT_*_TOKEN) so a
//! differently named tokenizer loads unchanged; the remaining constants are
//! the first family's (Laya / ModernBERT) and not per-model yet. Do not grow
//! new family specifics here.

use std::path::Path;

use thiserror::Error;

/// Default special-token surface forms: the first family's (Laya / ModernBERT).
/// A model whose tokenizer names them differently declares all four in
/// model.json -- mmBERT uses `<bos>`/`<eos>`/`<mask>`/`<pad>`. A wrong name is
/// not a silent failure: the vocab lookup misses and `load` fails loudly.
const DEFAULT_CLS_TOKEN: &str = "[CLS]";
const DEFAULT_SEP_TOKEN: &str = "[SEP]";
const DEFAULT_MASK_TOKEN: &str = "[MASK]";
const DEFAULT_PAD_TOKEN: &str = "[PAD]";
/// Per-option token cap: one mask token + at most 48 text tokens (rl_common line 60).
const OPT_TEXT_CAP: usize = 48;
/// If the option block leaves less head budget than this, every option is shrunk
/// evenly and the head shrinks to the remainder (rl_common lines 62-66).
const MIN_HEAD_BUDGET: usize = 16;
/// Options may never shrink below this many tokens (rl_common line 63).
const MIN_OPT_TOKENS: usize = 4;
/// The fixed noul pair (rl_common.render_options): `p[1] == P(true)` depends
/// on this exact wording. The bare descriptions back the /v1/systemone
/// adapter's one-sided Jev criteria fallback; PAIR[i] is
/// `"{false|true}: {DESCRIPTIONS[i]}"`.
pub const NOUL_DEFAULT_PAIR: [&str; 2] = [
    "false: no, the statement does not hold",
    "true: yes, the statement holds",
];
pub const NOUL_DEFAULT_DESCRIPTIONS: [&str; 2] = [
    "no, the statement does not hold",
    "yes, the statement holds",
];

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
}

/// Token ids and length budgets pinned by the golden corpus meta.
/// Specials are resolved from the tokenizer at load time, never hardcoded.
#[derive(Debug, Clone)]
pub struct Layout {
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub pad_id: u32,
    pub max_len: usize,
    pub head_max_len: usize,
    /// The mask token's surface form. Literal occurrences in user text are
    /// replaced by a space before encoding (mask_tok hygiene,
    /// rl_common.build_sequence); taken from the tokenizer's own name for the
    /// token.
    pub mask_literal: String,
}

#[derive(Clone)]
pub struct TokenizerBundle {
    pub tok: tokenizers::Tokenizer,
    pub layout: Layout,
}

/// A built request sequence: model input ids (right-padding happens at collate
/// time), `[MASK]` marker positions (one per option), and the uncapped rendered
/// token count (protocol-layer usage.input_tokens).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltSequence {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub full_len: usize,
}

impl TokenizerBundle {
    /// Load from a model directory holding `tokenizer/tokenizer.json` and
    /// `model.json` (model-family specifics live in the manifest, not
    /// product code).
    /// (weights are not needed to grade tokenization or calibration).
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
        // Special-token names come from the manifest so a second model
        // family needs no code change here.
        let name = |key: &str, default: &str| {
            cfg.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or(default)
                .to_string()
        };
        let mask_token = name("mask_token", DEFAULT_MASK_TOKEN);
        let layout = Layout {
            cls_id: special(&name("cls_token", DEFAULT_CLS_TOKEN))?,
            sep_id: special(&name("sep_token", DEFAULT_SEP_TOKEN))?,
            mask_id: special(&mask_token)?,
            pad_id: special(&name("pad_token", DEFAULT_PAD_TOKEN))?,
            max_len: num("max_len")?,
            head_max_len: num("head_max_len")?,
            mask_literal: mask_token,
        };
        Ok(Self { tok, layout })
    }

    fn encode_plain(&self, text: &str) -> Vec<u32> {
        // add_special_tokens=false: [CLS]/[SEP] are added by this module,
        // mirroring the official Python path.
        self.tok
            .encode(text, false)
            .expect("tokenizer encode cannot fail on &str input")
            .get_ids()
            .to_vec()
    }

    /// Option texts in marker order, re-typed from rl_common.render_options:
    /// choice passes options through (crit values are None); score renders
    /// "level %d: %s"; noul renders the two fixed sentences so `p[1] == P(true)`.
    pub fn render_options(&self, task: &str, options: &[String]) -> Vec<String> {
        match task {
            "choice" => options.to_vec(),
            "score" => options
                .iter()
                .enumerate()
                .map(|(i, text)| format!("level {i}: {text}"))
                .collect(),
            // A caller-supplied 2-item pair (Jev criteria via /v1/systemone)
            // replaces the default sentences; the native protocol passes none.
            _ => {
                if options.len() == 2 {
                    options.to_vec()
                } else {
                    NOUL_DEFAULT_PAIR.map(|s| s.to_string()).to_vec()
                }
            }
        }
    }

    /// Build the request sequence: `[CLS]` head `[SEP]` (`[MASK]` opt)k `[SEP]` state `[SEP]`,
    /// with the full model-layer truncation semantics (48-cap, 192 shrink, 512 room).
    /// `options` carries the option/level texts (choice/score); noul uses its
    /// fixed pair unless the caller supplies two custom sentences (Jev criteria).
    pub fn build_sequence(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
    ) -> BuiltSequence {
        let layout = &self.layout;
        let ins = question.replace(&layout.mask_literal, " ");
        let head_full = self.encode_plain(&format!("{task} question: {ins}"));

        // Uncapped option encodings: `[MASK]` + " " + text (mask included in each).
        let opt_full: Vec<Vec<u32>> = self
            .render_options(task, options)
            .iter()
            .map(|opt| {
                let mut ids = vec![layout.mask_id];
                ids.extend(
                    self.encode_plain(&format!(" {}", opt.replace(&layout.mask_literal, " "))),
                );
                ids
            })
            .collect();
        // Uncapped rendered length: `[CLS]` head `[SEP]` opts `[SEP]` state [SEP].
        let mut full_len =
            1 + head_full.len() + 1 + opt_full.iter().map(|o| o.len()).sum::<usize>() + 1;

        // Per-option cap: the mask token is kept and at most OPT_TEXT_CAP text tokens
        // follow it (rl_common line 60 caps the text, then prepends the mask -- the
        // cap never eats the mask).
        let mut opt_ids: Vec<Vec<u32>> = opt_full
            .iter()
            .map(|o| o[..(1 + OPT_TEXT_CAP).min(o.len())].to_vec())
            .collect();
        let mut opt_budget = layout
            .head_max_len
            .saturating_sub(opt_ids.iter().map(|o| o.len()).sum::<usize>());
        if opt_budget < MIN_HEAD_BUDGET {
            let per = ((layout.head_max_len - MIN_HEAD_BUDGET) / opt_ids.len().max(1))
                .max(MIN_OPT_TOKENS);
            opt_ids = opt_ids
                .iter()
                .map(|o| o[..per.min(o.len())].to_vec())
                .collect();
            opt_budget = layout
                .head_max_len
                .saturating_sub(opt_ids.iter().map(|o| o.len()).sum::<usize>());
        }
        let head_ids = &head_full[..head_full.len().min(opt_budget.max(8))];

        let mut ids = Vec::with_capacity(layout.max_len);
        ids.push(layout.cls_id);
        ids.extend_from_slice(head_ids);
        ids.push(layout.sep_id);
        let mut markers = Vec::with_capacity(opt_ids.len());
        for opt in &opt_ids {
            markers.push(ids.len());
            ids.extend_from_slice(opt);
        }
        ids.push(layout.sep_id);

        let state_full = self.encode_plain(&state.replace(&layout.mask_literal, " "));
        full_len += state_full.len() + 1;
        // API path default: truncate_right (keep the head of the state).
        let room = layout.max_len.saturating_sub(ids.len() + 1);
        ids.extend_from_slice(&state_full[..state_full.len().min(room)]);
        ids.push(layout.sep_id);
        ids.truncate(layout.max_len);

        BuiltSequence {
            markers: markers
                .into_iter()
                .filter(|m| *m < layout.max_len)
                .collect(),
            ids,
            full_len,
        }
    }
}
