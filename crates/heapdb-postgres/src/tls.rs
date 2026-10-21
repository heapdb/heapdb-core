//! Connection strings and TLS, with libpq's `sslmode` meanings.
//!
//! The `postgres` crate understands `sslmode=disable|prefer|require` and
//! rejects libpq's `verify-ca` / `verify-full` and `sslrootcert`, so those
//! are taken out of the URL here and handled with rustls:
//!
//! | sslmode | encryption | certificate check |
//! | --- | --- | --- |
//! | `disable` | no | - |
//! | `prefer` (default) | if the server offers it | none (as libpq) |
//! | `require` | yes | none (as libpq) |
//! | `verify-ca` | yes | chain to a trusted root |
//! | `verify-full` | yes | chain and host name |
//!
//! Trusted roots: the platform store plus `sslrootcert=<pem file>`.

use std::sync::Arc;

use heapdb::{Error, Result};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::WebPkiServerVerifier;
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{CertificateError, ClientConfig, DigitallySignedStruct, RootCertStore, SignatureScheme};
use tokio_postgres_rustls::MakeRustlsConnect;

/// `sslmode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Disable,
    Prefer,
    Require,
    VerifyCa,
    VerifyFull,
}

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::invalid(format!("bad Postgres connection string: {msg}"))
}

/// Split `sslmode` / `sslrootcert` out of a URL (or key=value string) and
/// parse the rest with the `postgres` crate.
pub fn parse(url: &str) -> Result<(postgres::Config, Mode, Option<String>)> {
    let mut mode = Mode::Prefer;
    let mut root = None;
    let mut take = |k: &str, v: &str| -> Result<bool> {
        match k {
            "sslmode" => {
                mode = match v {
                    "disable" | "allow" => Mode::Disable,
                    "prefer" => Mode::Prefer,
                    "require" => Mode::Require,
                    "verify-ca" => Mode::VerifyCa,
                    "verify-full" => Mode::VerifyFull,
                    other => return Err(bad(format!("unknown sslmode {other:?}"))),
                };
                Ok(true)
            }
            "sslrootcert" => {
                root = Some(v.to_string());
                Ok(true)
            }
            _ => Ok(false),
        }
    };
    let rest = if url.contains("://") {
        match url.split_once('?') {
            None => url.to_string(),
            Some((base, q)) => {
                let mut keep = Vec::new();
                for part in q.split('&').filter(|p| !p.is_empty()) {
                    let (k, v) = part.split_once('=').unwrap_or((part, ""));
                    if !take(k, v)? {
                        keep.push(part);
                    }
                }
                if keep.is_empty() {
                    base.to_string()
                } else {
                    format!("{base}?{}", keep.join("&"))
                }
            }
        }
    } else {
        // key=value form (values with spaces must be quoted; kept simple).
        let mut keep = Vec::new();
        for part in url.split_whitespace() {
            let (k, v) = part.split_once('=').unwrap_or((part, ""));
            if !take(k, v.trim_matches('\''))? {
                keep.push(part);
            }
        }
        keep.join(" ")
    };
    let mut cfg: postgres::Config = rest.parse().map_err(bad)?;
    cfg.ssl_mode(match mode {
        Mode::Disable => postgres::config::SslMode::Disable,
        Mode::Prefer => postgres::config::SslMode::Prefer,
        _ => postgres::config::SslMode::Require,
    });
    Ok((cfg, mode, root))
}

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn roots(extra: Option<&str>) -> Result<RootCertStore> {
    let mut store = RootCertStore::empty();
    let native = rustls_native_certs::load_native_certs();
    store.add_parsable_certificates(native.certs);
    if let Some(path) = extra {
        let certs = CertificateDer::pem_file_iter(path)
            .map_err(|e| bad(format!("sslrootcert {path}: {e}")))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| bad(format!("sslrootcert {path}: {e}")))?;
        store.add_parsable_certificates(certs);
    }
    if store.is_empty() {
        return Err(bad("no trusted root certificates (set sslrootcert=<file>)"));
    }
    Ok(store)
}

/// A rustls connector for `mode` (not `Disable`).
pub fn connector(mode: Mode, root: Option<&str>) -> Result<MakeRustlsConnect> {
    let p = provider();
    let builder = ClientConfig::builder_with_provider(p.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| Error::Backend(format!("TLS setup: {e}")))?;
    let verifier: Arc<dyn ServerCertVerifier> = match mode {
        Mode::VerifyCa | Mode::VerifyFull => {
            let webpki = WebPkiServerVerifier::builder_with_provider(Arc::new(roots(root)?), p.clone())
                .build()
                .map_err(|e| Error::Backend(format!("TLS setup: {e}")))?;
            Arc::new(Verify {
                inner: webpki,
                check_name: mode == Mode::VerifyFull,
                provider: p,
            })
        }
        _ => Arc::new(Encrypt { provider: p }),
    };
    let cfg = builder
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth();
    Ok(MakeRustlsConnect::new(cfg))
}

/// `verify-ca` / `verify-full`: WebPKI chain validation; the host name
/// only for `verify-full`.
#[derive(Debug)]
struct Verify {
    inner: Arc<WebPkiServerVerifier>,
    check_name: bool,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for Verify {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp: &[u8],
        now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        match self
            .inner
            .verify_server_cert(end_entity, intermediates, server_name, ocsp, now)
        {
            Err(rustls::Error::InvalidCertificate(
                CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. },
            )) if !self.check_name => Ok(ServerCertVerified::assertion()),
            other => other,
        }
    }
    fn verify_tls12_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(m, c, d, &self.provider.signature_verification_algorithms)
    }
    fn verify_tls13_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(m, c, d, &self.provider.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// `prefer` / `require`: encryption without certificate verification, as
/// libpq does (handshake signatures are still checked).
#[derive(Debug)]
struct Encrypt {
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for Encrypt {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(m, c, d, &self.provider.signature_verification_algorithms)
    }
    fn verify_tls13_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(m, c, d, &self.provider.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sslmode_is_taken_out() {
        let (cfg, mode, root) =
            parse("postgres://u:p@h:5433/db?sslmode=verify-full&sslrootcert=/ca.pem&application_name=x")
                .unwrap();
        assert_eq!((mode, root.as_deref()), (Mode::VerifyFull, Some("/ca.pem")));
        assert_eq!(cfg.get_application_name(), Some("x"));
        assert_eq!(cfg.get_ports(), [5433]);
        let (_, mode, _) = parse("postgresql://h/db").unwrap();
        assert_eq!(mode, Mode::Prefer);
        let (cfg, mode, _) = parse("host=h dbname=d sslmode=disable").unwrap();
        assert_eq!((mode, cfg.get_dbname()), (Mode::Disable, Some("d")));
        assert!(parse("postgres://h/db?sslmode=sometimes").is_err());
        assert!(connector(Mode::Require, None).is_ok());
    }
}
