//! String interning for features, event types and entities.

use rustc_hash::FxHashMap;

/// Maps strings to dense `u32` ids and back. Append-only.
#[derive(Clone, Debug, Default)]
pub struct Interner {
    map: FxHashMap<Box<str>, u32>,
    names: Vec<Box<str>>,
}

impl Interner {
    /// The id of `s`, inserting it if new.
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = u32::try_from(self.names.len()).expect("more than 2^32 distinct strings");
        self.names.push(s.into());
        self.map.insert(s.into(), id);
        id
    }

    /// The id of `s`, if interned.
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }

    /// The string of an id.
    ///
    /// # Panics
    /// If `id` was not produced by this interner.
    pub fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    /// Number of interned strings.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// True if nothing is interned.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// The three vocabularies of a database.
#[derive(Clone, Debug, Default)]
pub struct Vocab {
    /// Feature tokens.
    pub features: Interner,
    /// Event types.
    pub types: Interner,
    /// Entities.
    pub entities: Interner,
}

impl Vocab {
    /// Sort feature ids by their names (the order the reference
    /// implementation uses for credit assignment).
    pub fn sort_by_name(&self, ids: &mut [u32]) {
        ids.sort_by(|&a, &b| self.features.name(a).cmp(self.features.name(b)));
    }
}
