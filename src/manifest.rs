use serde_json::Value;
use std::path::Path;
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid model.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("invalid model.json: {0}")]
    Invalid(String),
}

pub fn read_manifest(model_dir: &Path) -> Result<Value, ManifestError> {
    let path = model_dir.join("model.json");
    let raw = std::fs::read_to_string(&path).map_err(|source| ManifestError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_manifest(&raw)
}

pub fn parse_manifest(raw: &str) -> Result<Value, ManifestError> {
    let parsed: Value = serde_json::from_str(raw)?;
    normalize(parsed).map_err(ManifestError::Invalid)
}

pub fn normalize(mut manifest: Value) -> Result<Value, String> {
    let version = manifest.get("manifest_version").and_then(Value::as_u64);
    match version {
        None | Some(1) => return Ok(manifest),
        Some(2) => {}
        Some(other) => {
            return Err(format!(
                "declares manifest_version {other} (this build reads v1 and v2)"
            ));
        }
    }
    for group in ["execution", "preprocess", "postprocess"] {
        let root = manifest.as_object_mut().expect("root is an object");
        let Some(body) = root.remove(group) else {
            continue;
        };
        let Value::Object(entries) = body else {
            return Err(format!(
                "the {group:?} group must be an object grouping manifest fields"
            ));
        };
        for (key, value) in entries {
            let root = manifest.as_object_mut().expect("root is an object");
            if root.contains_key(&key) {
                return Err(format!(
                    "{key:?} is declared both at the top level and inside \
                     {group:?} -- pick one"
                ));
            }
            root.insert(key, value);
        }
    }
    resolve_dialect_card(&mut manifest)?;
    Ok(manifest)
}

struct DialectCard {
    template: &'static str,

    family: &'static str,
    slot_rule: &'static str,
    confidence: &'static str,
    rounding: u64,
    noul_first: bool,
    chunked_choice: bool,
}
const DIALECT_CARDS: [DialectCard; 2] = [
    DialectCard {
        template: "qev-labels-v2",
        family: "letter_branches",
        slot_rule: "exclusion-256",
        confidence: "uniform-margin",
        rounding: 6,
        noul_first: true,
        chunked_choice: true,
    },
    DialectCard {
        template: "decider-v5",
        family: "letter_readout",
        slot_rule: "single-token-fill-255",
        confidence: "margin-clamp",
        rounding: 4,
        noul_first: false,
        chunked_choice: false,
    },
];

fn resolve_dialect_card(manifest: &mut Value) -> Result<(), String> {
    if manifest.get("family").and_then(Value::as_str) != Some("letter_readout") {
        return Ok(());
    }
    let Some(template) = manifest.get("template").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(card) = DIALECT_CARDS.iter().find(|c| c.template == template) else {
        let known = DIALECT_CARDS
            .iter()
            .map(|c| format!("{:?}", c.template))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "declares template {template:?} but no such dialect card is \
             installed (this build ships: {known})"
        ));
    };
    let expect_str = |field: &str, want: &str| -> Result<(), String> {
        let got = manifest.get(field).and_then(Value::as_str);
        match got {
            Some(got) if got == want => Ok(()),
            Some(got) => Err(format!(
                "template {template:?} is implemented with {field:?} = {want:?}, the manifest \
                 declares {got:?}"
            )),
            None => Err(format!(
                "template {template:?} requires the field {field:?}"
            )),
        }
    };
    expect_str("slot_rule", card.slot_rule)?;
    expect_str("confidence", card.confidence)?;
    let expect_num = |field: &str, want: u64| -> Result<(), String> {
        let got = manifest.get(field).and_then(Value::as_u64);
        if got == Some(want) {
            Ok(())
        } else if got.is_none() {
            Err(format!(
                "template {template:?} requires the field {field:?}"
            ))
        } else {
            Err(format!(
                "template {template:?} is implemented with {field:?} = {want}, the manifest \
                 declares {}",
                got.unwrap()
            ))
        }
    };
    expect_num("rounding", card.rounding)?;
    let expect_bool = |field: &str, want: bool| -> Result<(), String> {
        let got = manifest.get(field).and_then(Value::as_bool);
        match got {
            Some(got) if got == want => Ok(()),
            Some(got) => Err(format!(
                "template {template:?} is implemented with {field:?} = {want}, the manifest \
                 declares {got}"
            )),
            None => Err(format!(
                "template {template:?} requires the field {field:?}"
            )),
        }
    };
    expect_bool("noul_first", card.noul_first)?;
    expect_bool("chunked_choice", card.chunked_choice)?;
    manifest
        .as_object_mut()
        .expect("root is an object")
        .insert("family".to_string(), Value::String(card.family.to_string()));
    Ok(())
}
