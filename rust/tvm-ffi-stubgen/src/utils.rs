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

use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn normalize_prefix(prefix: &str) -> String {
    if prefix.is_empty() {
        return String::new();
    }
    if prefix.ends_with('.') {
        prefix.to_string()
    } else {
        format!("{}.", prefix)
    }
}

pub(crate) fn ensure_out_dir(
    out_dir: &Path,
    overwrite: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if out_dir.exists() {
        let mut has_entries = false;
        for entry in fs::read_dir(out_dir)? {
            let entry = entry?;
            if entry.file_name() != "." && entry.file_name() != ".." {
                has_entries = true;
                break;
            }
        }
        if has_entries && !overwrite {
            return Err("output directory is not empty (use --overwrite to proceed)".into());
        }
    } else {
        fs::create_dir_all(out_dir)?;
    }
    Ok(())
}

pub(crate) fn default_tvm_ffi_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let current = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = current.join("../tvm-ffi");
    if candidate.exists() {
        return Ok(candidate);
    }
    Err("unable to locate tvm-ffi path (use --tvm-ffi-path)".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn normalize_prefix_adds_trailing_dot() {
        assert_eq!(normalize_prefix(""), "");
        assert_eq!(normalize_prefix("testing"), "testing.");
        assert_eq!(normalize_prefix("testing."), "testing.");
    }

    #[test]
    fn ensure_out_dir_creates_missing() {
        let dir = std::env::temp_dir().join(format!(
            "tvm_ffi_stubgen_utils_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        ensure_out_dir(&dir, false).expect("create");
        assert!(dir.is_dir());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensure_out_dir_rejects_nonempty_without_overwrite() {
        let dir = std::env::temp_dir().join(format!(
            "tvm_ffi_stubgen_utils_nonempty_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("marker.txt"), "x").unwrap();
        let err = ensure_out_dir(&dir, false).unwrap_err();
        assert!(err.to_string().contains("not empty"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensure_out_dir_allows_nonempty_with_overwrite() {
        let dir = std::env::temp_dir().join(format!(
            "tvm_ffi_stubgen_utils_overwrite_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("marker.txt"), "x").unwrap();
        ensure_out_dir(&dir, true).expect("overwrite");
        let _ = fs::remove_dir_all(&dir);
    }
}
