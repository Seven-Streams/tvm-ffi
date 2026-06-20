/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
use tvm_ffi::*;

/// Build the fixture map `{ "a": 1, "b": 2, "c": 3 }`.
fn sample() -> Result<Map<String, i64>> {
    Map::from_pairs(vec![
        (String::from("a"), 1i64),
        (String::from("b"), 2i64),
        (String::from("c"), 3i64),
    ])
}

#[test]
fn test_map_core_lookup() -> Result<()> {
    let map = sample()?;

    assert_eq!(map.len()?, 3);
    assert!(!map.is_empty()?);

    assert_eq!(map.get(&String::from("a"))?, Some(1));
    assert_eq!(map.get(&String::from("b"))?, Some(2));
    assert_eq!(map.get(&String::from("c"))?, Some(3));
    // Absent key -> None (not the C++ MISSING sentinel).
    assert_eq!(map.get(&String::from("z"))?, None);

    assert!(map.contains_key(&String::from("a"))?);
    assert!(!map.contains_key(&String::from("z"))?);

    Ok(())
}

#[test]
fn test_map_iteration() -> Result<()> {
    let map = sample()?;

    // Order is unspecified; compare against the sorted multiset.
    let mut values = map.values()?;
    values.sort();
    assert_eq!(values, vec![1, 2, 3]);

    let keys = map.keys()?;
    assert_eq!(keys.len(), 3);

    // `items()` must agree with `keys`/`get`: every reported pair round-trips.
    let items = map.items()?;
    assert_eq!(items.len(), 3);
    for (k, v) in &items {
        assert_eq!(map.get(k)?, Some(*v));
    }

    Ok(())
}

#[test]
fn test_map_empty() -> Result<()> {
    let map = Map::<String, i64>::from_pairs(Vec::new())?;

    assert_eq!(map.len()?, 0);
    assert!(map.is_empty()?);
    assert!(map.items()?.is_empty());
    assert!(map.keys()?.is_empty());
    assert!(map.values()?.is_empty());
    assert_eq!(map.get(&String::from("a"))?, None);

    Ok(())
}

#[test]
fn test_map_any_roundtrip() -> Result<()> {
    let map = sample()?;

    // Owned Any roundtrip (move path).
    let any = Any::from(map.clone());
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIMap as i32);
    let back: Map<String, i64> = Map::try_from(any).expect("Any -> Map failed");
    assert_eq!(back.len()?, 3);
    assert_eq!(back.get(&String::from("b"))?, Some(2));

    // Borrowed AnyView roundtrip (copy path).
    let view = AnyView::from(&map);
    let from_view: Map<String, i64> = Map::try_from(view).expect("AnyView -> Map failed");
    assert_eq!(from_view.len()?, 3);

    Ok(())
}

/// Compile-only: mirrors the code shapes `tvm-ffi-stubgen` emits for a typed
/// `Map` field / parameter / return, so the generated bindings are guaranteed to
/// build against the crate (the functions are never called).
#[allow(dead_code)]
mod generated_shapes {
    use tvm_ffi::*;

    // `#[repr(C)]` field, as in a generated `...Obj` struct.
    #[repr(C)]
    pub struct Config {
        pub cfg: Map<String, i64>,
    }

    // Object argument: stubgen passes it as `AnyView::from(&arg)`.
    fn takes_map(m: Map<String, i64>) -> i32 {
        let _view = AnyView::from(&m);
        0
    }

    // Owning return: stubgen converts the packed-call result via `try_into`.
    fn returns_map(any: Any) -> Result<Map<String, i64>> {
        Ok(any.try_into()?)
    }
}

#[test]
fn test_map_last_key_wins() -> Result<()> {
    // The C++ `ffi.Map` constructor keeps the last value for a duplicate key.
    let map = Map::from_pairs(vec![
        (String::from("k"), 1i64),
        (String::from("k"), 2i64),
    ])?;
    assert_eq!(map.len()?, 1);
    assert_eq!(map.get(&String::from("k"))?, Some(2));
    Ok(())
}
