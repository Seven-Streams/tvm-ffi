// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use serde::Deserialize;
use std::collections::{BTreeSet, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct TypeSchema {
    pub(crate) origin: String,
    pub(crate) args: Vec<TypeSchema>,
}

#[derive(Deserialize)]
struct TypeSchemaJson {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    args: Vec<TypeSchemaJson>,
}

pub(crate) fn extract_type_schema(metadata: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(metadata).ok()?;
    value
        .get("type_schema")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

pub(crate) fn parse_type_schema(schema: &str) -> Option<TypeSchema> {
    let json: TypeSchemaJson = serde_json::from_str(schema).ok()?;
    Some(parse_type_schema_json(&json))
}

pub(crate) fn collect_type_keys(
    schema: &TypeSchema,
    known: &HashSet<String>,
    out: &mut BTreeSet<String>,
) {
    if known.contains(&schema.origin) {
        out.insert(schema.origin.clone());
    }
    for arg in &schema.args {
        collect_type_keys(arg, known, out);
    }
}

fn parse_type_schema_json(json: &TypeSchemaJson) -> TypeSchema {
    TypeSchema {
        origin: json.ty.clone(),
        args: json.args.iter().map(parse_type_schema_json).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn extract_type_schema_from_metadata() {
        let meta = r#"{"type_schema":"{\"type\":\"ffi.Function\",\"args\":[{\"type\":\"int\"},{\"type\":\"int\"}]}"}"#;
        let raw = extract_type_schema(meta).expect("type_schema");
        let schema = parse_type_schema(&raw).expect("parse");
        assert_eq!(schema.origin, "ffi.Function");
        assert_eq!(schema.args.len(), 2);
        assert_eq!(schema.args[0].origin, "int");
    }

    #[test]
    fn extract_type_schema_invalid_json() {
        assert!(extract_type_schema("not json").is_none());
        assert!(extract_type_schema("{}").is_none());
    }

    #[test]
    fn parse_type_schema_optional_array_map() {
        let opt = parse_type_schema(r#"{"type":"Optional","args":[{"type":"int"}]}"#).unwrap();
        assert_eq!(opt.origin, "Optional");
        assert_eq!(opt.args[0].origin, "int");

        let arr = parse_type_schema(r#"{"type":"ffi.Array","args":[{"type":"str"}]}"#).unwrap();
        assert_eq!(arr.origin, "ffi.Array");

        let map = parse_type_schema(
            r#"{"type":"ffi.Map","args":[{"type":"str"},{"type":"int"}]}"#,
        )
        .unwrap();
        assert_eq!(map.args.len(), 2);
    }

    #[test]
    fn collect_type_keys_nested_and_ordered() {
        let schema = parse_type_schema(
            r#"{"type":"ffi.Function","args":[{"type":"testing.SchemaAllTypes"},{"type":"testing.TestIntPair"}]}"#,
        )
        .unwrap();
        let known: HashSet<String> = ["testing.SchemaAllTypes", "testing.TestIntPair", "other.X"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let mut out = BTreeSet::new();
        collect_type_keys(&schema, &known, &mut out);
        assert_eq!(
            out.into_iter().collect::<Vec<_>>(),
            vec![
                "testing.SchemaAllTypes".to_string(),
                "testing.TestIntPair".to_string(),
            ]
        );
    }

    #[test]
    fn collect_type_keys_ignores_unknown_origins() {
        let schema = parse_type_schema(r#"{"type":"ffi.Function","args":[{"type":"unknown.Type"}]}"#)
            .unwrap();
        let known: HashSet<String> = HashSet::new();
        let mut out = BTreeSet::new();
        collect_type_keys(&schema, &known, &mut out);
        assert!(out.is_empty());
    }
}
