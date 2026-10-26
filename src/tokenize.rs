use crate::family::Family;
use crate::family::letter_readout::MAX_OPTIONS;
use std::path::Path;
use thiserror::Error;
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

#[derive(Debug, Clone)]
pub struct Layout {
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub pad_id: u32,

    pub max_len: usize,

    pub head_max_len: usize,

    pub mask_literal: String,

    pub sep_literal: String,

    pub seq_align: usize,

    pub fixed_classes: Vec<usize>,
}
#[derive(Clone)]
pub struct TokenizerBundle {
    pub tok: tokenizers::Tokenizer,
    pub layout: Layout,

    pub family: Family,

    pub digit_split: bool,

    pub letter_table: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BuiltSequence {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub full_len: usize,

    pub state_tokens: usize,

    pub nopts: Vec<usize>,

    pub media: Vec<MediaItem>,

    pub text: String,

    pub prefix_len: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaFrame {
    pub format: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MediaItem {
    Image(MediaFrame),
    Video { frames: Vec<MediaFrame>, fps: f64 },
}
impl BuiltSequence {
    pub fn real_k(&self) -> usize {
        if self.nopts.is_empty() {
            self.markers.len()
        } else {
            self.nopts.iter().sum()
        }
    }
}

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

fn derive_branch_table(tok: &tokenizers::Tokenizer) -> Result<Vec<u32>, TokenizeError> {
    let round_trip = |name: &str| -> Option<u32> {
        let enc = tok
            .encode(name, false)
            .expect("tokenizer encode cannot fail on &str input");
        let ids = enc.get_ids();
        if ids.len() != 1 {
            return None;
        }
        let decoded = tok
            .decode(ids, false)
            .expect("tokenizer decode cannot fail on one known id");
        (decoded == name).then(|| ids[0])
    };
    let labels = crate::family::letter_branches::slot_labels();
    let mut out: Vec<u32> = Vec::with_capacity(labels.len());
    for label in &labels[..26] {
        let id = round_trip(label).ok_or_else(|| {
            TokenizeError::LetterTable(format!(
                "answer slot {label:?} is not a single round-trip token in this tokenizer"
            ))
        })?;
        out.push(id);
    }
    let pairs: Vec<Option<u32>> = labels[26..].iter().map(|l| round_trip(l)).collect();
    if pairs.iter().all(Option::is_some) {
        out.extend(pairs.into_iter().flatten());
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
    pub fn load(model_dir: &Path) -> Result<Self, TokenizeError> {
        let tok_path = [
            model_dir.join("tokenizer").join("tokenizer.json"),
            model_dir.join("tokenizer.json"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| model_dir.join("tokenizer").join("tokenizer.json"));
        let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(TokenizeError::Tokenizer)?;
        let cfg: serde_json::Value =
            crate::manifest::read_manifest(model_dir).map_err(|err| match err {
                crate::manifest::ManifestError::Io { path, source } => {
                    TokenizeError::Io { path, source }
                }
                crate::manifest::ManifestError::Parse(source) => TokenizeError::ConfigParse(source),
                crate::manifest::ManifestError::Invalid(msg) => TokenizeError::Family(msg),
            })?;
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
        let name = |key: &'static str| -> Result<String, TokenizeError> {
            cfg.get(key)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or(TokenizeError::MissingConfig(key))
        };
        let pad_token = name("pad_token")?;
        let head_max_len = match family {
            Family::MarkerHead => num("head_max_len")?,
            Family::IndependentOptions | Family::LetterReadout | Family::LetterBranches => cfg
                .get("head_max_len")
                .and_then(|v| v.as_u64())
                .map_or(max_len, |v| v as usize),
        };
        let layout = match family {
            Family::LetterReadout | Family::LetterBranches => {
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
        let seq_align: usize = cfg
            .get("seq_align")
            .and_then(|v| v.as_u64())
            .map_or(1, |v| v.max(1) as usize);
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
            Family::LetterBranches => derive_branch_table(&tok)?,
            _ => Vec::new(),
        };
        let digit_split = match family {
            Family::IndependentOptions => cfg
                .get("digit_split")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Family::MarkerHead | Family::LetterReadout | Family::LetterBranches => false,
        };
        Ok(Self {
            tok,
            layout,
            family,
            digit_split,
            letter_table,
        })
    }

    pub(crate) fn encode_plain(&self, text: &str) -> Vec<u32> {
        self.tok
            .encode(text, false)
            .expect("tokenizer encode cannot fail on &str input")
            .get_ids()
            .to_vec()
    }

    pub fn build_sequences(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        media: &[MediaItem],
    ) -> Vec<BuiltSequence> {
        match self.family {
            Family::MarkerHead | Family::IndependentOptions => {
                if !media.is_empty() {
                    panic!("media reached the marker families; the protocol layer gates it");
                }
                vec![self.build_sequence(task, question, state, options)]
            }
            Family::LetterReadout => {
                if !media.is_empty() {
                    panic!("media reached the letter-readout family; the protocol layer gates it");
                }
                crate::family::letter_readout::build_rows(self, task, question, state, options)
            }
            Family::LetterBranches => crate::family::letter_branches::build_rows(
                self, task, question, state, options, media,
            ),
        }
    }

    pub fn render_options(&self, task: &str, options: &[String]) -> Vec<String> {
        match self.family {
            Family::MarkerHead => crate::family::marker_head::render_options(task, options),
            Family::IndependentOptions => {
                crate::family::independent_options::render_options(task, options)
            }
            Family::LetterReadout => crate::family::letter_readout::render_options(task, options),
            Family::LetterBranches => crate::family::letter_branches::render_options(task, options),
        }
    }

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
                assert!(
                    rows.len() == 1,
                    "letter score requests render one row per level; use build_sequences"
                );
                rows.into_iter().next().expect("checked len above")
            }
            Family::LetterBranches => crate::family::letter_branches::build_rows(
                self,
                task,
                question,
                state,
                options,
                &[],
            )
            .into_iter()
            .next()
            .expect("letter-branches renders exactly one row"),
        }
    }
}
