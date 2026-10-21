//! Identifiers and credentials (contract §3): database ids, key ids, API
//! keys and their SHA-256 hashes.

use rand::Rng;
use sha2::{Digest, Sha256};

const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Prefix of live API keys.
pub const API_KEY_PREFIX: &str = "hdb_live_";

fn random_base62(n: usize) -> String {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| BASE62[rng.random_range(0..BASE62.len())] as char)
        .collect()
}

fn is_base62(s: &str) -> bool {
    s.bytes().all(|b| b.is_ascii_alphanumeric())
}

/// `db_` + 20 base62 characters.
pub fn is_database_id(s: &str) -> bool {
    s.strip_prefix("db_")
        .is_some_and(|r| r.len() == 20 && is_base62(r))
}

fn is_tagged(s: &str, prefix: &str) -> bool {
    s.strip_prefix(prefix).is_some_and(|r| {
        !r.is_empty()
            && r.len() <= 64
            && r.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    })
}

/// `key_` + 1..64 of `[A-Za-z0-9_-]`.
pub fn is_key_id(s: &str) -> bool {
    is_tagged(s, "key_")
}

/// `org_` + 1..64 of `[A-Za-z0-9_-]`.
pub fn is_org_id(s: &str) -> bool {
    is_tagged(s, "org_")
}

/// 64 lowercase hex digits.
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A fresh database id.
pub fn new_database_id() -> String {
    format!("db_{}", random_base62(20))
}

/// A fresh API key: `hdb_live_` + 43 base62 characters (256 bits).
pub fn new_api_key() -> String {
    format!("{API_KEY_PREFIX}{}", random_base62(43))
}

/// A fresh node id (per process start).
pub fn new_node_id() -> String {
    format!("node_{}", random_base62(12))
}

/// Lowercase hex SHA-256.
pub fn sha256_hex(data: &[u8]) -> String {
    let d = Sha256::digest(data);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Constant-time equality of two secrets (compares their SHA-256 digests,
/// so the running time does not depend on where they differ or their
/// lengths).
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let (x, y) = (Sha256::digest(a), Sha256::digest(b));
    x.iter().zip(y.iter()).fold(0u8, |acc, (p, q)| acc | (p ^ q)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        let id = new_database_id();
        assert!(is_database_id(&id), "{id}");
        assert!(!is_database_id("db_short"));
        assert!(!is_database_id("db_0123456789012345678!"));
        assert!(is_key_id("key_abc-1_2"));
        assert!(!is_key_id("key_"));
        assert!(!is_key_id("abc"));
        let k = new_api_key();
        assert!(k.starts_with(API_KEY_PREFIX) && k.len() == 9 + 43);
        let h = sha256_hex(b"abc");
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(is_sha256_hex(&h));
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreT"));
        assert!(!ct_eq(b"secret", b"secret-longer"));
    }
}
