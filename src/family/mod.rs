pub mod independent_options;
pub mod letter_branches;
pub mod letter_readout;
pub mod marker_head;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    MarkerHead,

    IndependentOptions,

    LetterReadout,

    LetterBranches,
}
impl Family {
    pub fn from_manifest(cfg: &serde_json::Value) -> Result<Self, String> {
        match cfg.get("family").and_then(|v| v.as_str()) {
            Some("marker_head") => Ok(Family::MarkerHead),
            Some("independent_options") => Ok(Family::IndependentOptions),
            Some("letter_readout") => Ok(Family::LetterReadout),
            Some("letter_branches") => Ok(Family::LetterBranches),
            Some(other) => Err(format!(
                "model.json declares unknown family {other:?} (this build knows: \
                 \"marker_head\", \"independent_options\", \"letter_readout\", \
                 \"letter_branches\")"
            )),
            None => Err(
                "model.json is missing the 'family' field (one of \"marker_head\", \
                 \"independent_options\", \"letter_readout\", \"letter_branches\")"
                    .to_string(),
            ),
        }
    }
}
