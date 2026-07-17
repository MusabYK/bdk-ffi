use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

// block-style sha256t_hash_newtype!
// spdk defines SpSharedSecretHash and BIP0352LabelHash as pub(crate), so we
// re-implement them with identical tags. This is safe, same math, same output.
use bitcoin_hashes::{sha256t_hash_newtype, Hash, HashEngine};
use secp256k1::{ecdh::shared_secret_point, PublicKey, Scalar, Secp256k1, SecretKey};

use bdk_wallet::bitcoin::Network;
// sp_lib = the silentpayments crate (aliased to avoid name collision)
use sp_lib::{
    utils::{sending::calculate_partial_secret, OutPoint as SpOutPoint},
    Network as SpNetwork, SilentPaymentAddress as SpAddress, SpVersion,
};

// BIP-352 Tagged Hashes
// spdk defines SpSharedSecretHash and BIP0352LabelHash as pub(crate), so we
// re-implement them with identical tags. This is safe, same math, same output.
sha256t_hash_newtype! {
    struct SpSharedSecretTag = hash_str("BIP0352/SharedSecret");
    #[hash_newtype(forward)]
    struct SpSharedSecretHash(_);
}

sha256t_hash_newtype! {
    struct BIP0352LabelTag = hash_str("BIP0352/Label");
    #[hash_newtype(forward)]
    struct BIP0352LabelHash(_);
}

/// Convert bitcoin::Network → sp_lib::Network (for address encoding).
/// BIP-352: testnet AND signet both use "tsp" HRP — both map to SpNetwork::Testnet.
fn to_sp_network(network: &Network) -> SpNetwork {
    match network {
        Network::Bitcoin => SpNetwork::Mainnet,
        Network::Testnet | Network::Testnet4 | Network::Signet => SpNetwork::Testnet,
        Network::Regtest => SpNetwork::Regtest,
    }
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum SilentPaymentError {
    #[error("Invalid key: {msg}")]
    InvalidKey { msg: String },
    #[error("Invalid address: {msg}")]
    InvalidAddress { msg: String },
    #[error("Cryptography error: {msg}")]
    CryptoError { msg: String },
    #[error("Encoding error: {msg}")]
    EncodingError { msg: String },
}

impl From<sp_lib::Error> for SilentPaymentError {
    fn from(e: sp_lib::Error) -> Self {
        SilentPaymentError::InvalidAddress { msg: e.to_string() }
    }
}

impl From<secp256k1::Error> for SilentPaymentError {
    fn from(e: secp256k1::Error) -> Self {
        SilentPaymentError::CryptoError { msg: e.to_string() }
    }
}

/// A BIP-352 Silent Payment address with component public keys.
/// sp1q... (mainnet) or tsp1q... (testnet/signet) — bech32m encoded.
#[derive(uniffi::Record, Debug, Clone)]
pub struct SilentPaymentAddress {
    /// Fully encoded bech32m address. Share this with senders.
    pub address: String,
    /// 33-byte compressed scan public key (hex).
    /// Pass to Frigate: subscribe(scanSecretHex: ..., spendPubkeyHex: ...).
    pub scan_pubkey_hex: String,
    /// 33-byte compressed spend public key (hex).
    pub spend_pubkey_hex: String,
    /// Which network this address is valid for (bdk_wallet::bitcoin::Network).
    pub network: Network,
}

/// Wraps a hex-encoded value (key bytes, tweak scalars, addresses).
#[derive(uniffi::Record, Debug, Clone)]
pub struct HexStringResult {
    pub value: String,
}

/// A payment found during transaction scanning.
/// Store tweak_hex — you MUST have it to spend the output (Phase 2).
#[derive(uniffi::Record, Debug, Clone)]
pub struct FoundPayment {
    /// vout index in the transaction.
    pub output_index: u32,
    /// 32-byte tweak scalar (hex, 64 chars). STORE THIS — needed for spending.
    /// Phase 2 will add: spend_key = spend_secret + H(b_scan × tweak_point || k)
    pub tweak_hex: String,
    /// Output value in satoshis. Caller fills this from transaction data.
    pub amount_sats: u64,
    /// None = standard address. Some(m) = labeled sub-address m.
    /// Eg:
    ///   Some(1)  → donations
    ///   Some(2)  → shop payments
    ///   etc.
    pub label: Option<u32>,
}

#[derive(uniffi::Record, Debug, Clone)]
pub struct ScanTransactionResult {
    pub payments: Vec<FoundPayment>,
}

/// Maps a Silent Payment output pubkey to its recipient address.
/// pubkey_hex is a 33-byte compressed pubkey.
/// P2TR scriptPubKey = OP_1 (0x51) OP_PUSHBYTES_32 (0x20) <pubkey_hex[1..] as bytes>
#[derive(uniffi::Record, Debug, Clone)]
pub struct OutputWithKey {
    pub pubkey_hex: String,
    pub recipient_address: String,
}

#[derive(uniffi::Record, Debug, Clone)]
pub struct PaymentRecipient {
    pub address: String,
    pub amount_sats: u64,
}

/// One input spent in a SP transaction. Requires the raw private key.
/// Phase 3 will add BDK-integrated sending that derives keys internally.
/// BIP-352 requires:
///   * ALL eligible input private keys (P2PKH, P2WPKH, P2SH-P2WPKH, P2TR)
///   * The corresponding outpoints (txid + vout) for the input hash
///   * Whether each input is taproot — BIP-352 negates P2TR keys with odd parity
///     before summing, which prevents certain cross-input linking attacks
#[derive(uniffi::Record, Debug, Clone)]
pub struct SendingInput {
    /// Private key for this input (32 bytes as hex, 64 chars)
    pub secret_key_hex: String,
    pub is_taproot: bool,
    /// Txid of the UTXO being spent (big-endian hex, as shown in explorers).
    pub txid: String,
    /// Output index of the UTXO (the "n" in txid:n)
    pub vout: u32,
}

/// Compute the BIP-352 compliant P2TR output pubkeys for a Silent Payment transaction.
///
/// Call this BEFORE building the BDK transaction. The returned pubkey_hex
/// is a 33-byte compressed pubkey. Build the P2TR output:
///   scriptPubKey = OP_1 (0x51) OP_PUSHBYTES_32 (0x20) <pubkey_hex bytes [1..]>
///
/// /// # BIP-352 sending protocol
/// ```text
/// 1. For each P2TR input: if parity is ODD, negate the private key
/// 2. a_sum = sum of all (possibly negated) input private keys
/// 3. A_sum = a_sum × G
/// 4. smallest_outpoint = lexicographically smallest (txid_le || vout_le) outpoint
/// 5. input_hash = H_BIP0352/Inputs(smallest_outpoint || A_sum)
/// 6. partial_secret = a_sum × input_hash         ← calculate_partial_secret()
/// 7. For each recipient:
///    shared_secret = partial_secret × B_scan     ← ECDH
///    t_k = H_BIP0352/SharedSecret(shared_secret || k)
///    P_k = B_spend + t_k × G                    ← output pubkey
/// ```
///
/// Steps 1–6 are handled by sp_lib::utils::sending::calculate_partial_secret.
/// Steps 7+ use our own public-API implementation (same as scan_transaction).
///
/// # Multiple recipients with the same scan key
/// BIP-352 requires using k = 0, 1, 2,... when sending to the same scan key
/// multiple times in one transaction. This function handles that automatically.
#[uniffi::export]
pub fn create_silent_payment_outputs(
    inputs: Vec<SendingInput>,
    recipients: Vec<PaymentRecipient>,
) -> Result<Vec<OutputWithKey>, SilentPaymentError> {
    if inputs.is_empty() {
        return Err(SilentPaymentError::InvalidKey {
            msg: "At least one input private key is required".into(),
        });
    }
    if recipients.is_empty() {
        return Err(SilentPaymentError::InvalidAddress {
            msg: "At least one recipient is required".into(),
        });
    }

    // Parse inputs
    let mut sp_keys: Vec<(SecretKey, bool)> = Vec::with_capacity(inputs.len());
    let mut sp_outpoints: Vec<SpOutPoint> = Vec::with_capacity(inputs.len());

    for input in &inputs {
        let key = SecretKey::from_slice(&hex_to_32_bytes(&input.secret_key_hex)?).map_err(|e| {
            SilentPaymentError::InvalidKey {
                msg: format!("Invalid input key: {e}"),
            }
        })?;
        sp_keys.push((key, input.is_taproot));

        // from_txid_and_vout accepts the txid as displayed (big-endian)
        // and handles the byte reversal to little-endian internally
        let outpoint =
            SpOutPoint::from_txid_and_vout(input.txid.clone(), input.vout).map_err(|e| {
                SilentPaymentError::EncodingError {
                    msg: format!("Invalid outpoint ({}:{}): {e}", input.txid, input.vout),
                }
            })?;
        sp_outpoints.push(outpoint);
    }

    // compute partial_secret via sp_lib
    //
    // calculate_partial_secret does:
    //   • Taproot negation: P2TR keys with odd parity are negated
    //   • Key summation: a_sum = sum of (possibly negated) keys
    //   • Input hash:    H_BIP0352/Inputs(smallest_outpoint || A_sum)
    //   • Multiplication: partial_secret = a_sum × input_hash
    //
    // PartialSecret.secret_bytes() is used to extract the scalar
    // and use it with shared_secret_point (constant-time ECDH).
    let partial_secret = calculate_partial_secret(&sp_keys, &sp_outpoints).map_err(|e| {
        SilentPaymentError::CryptoError {
            msg: format!("Partial secret computation failed: {e}"),
        }
    })?;
    let partial_scalar = SecretKey::from_slice(&partial_secret.secret_bytes()).map_err(|e| {
        SilentPaymentError::CryptoError {
            msg: format!("Partial secret is invalid scalar: {e}"),
        }
    })?;

    // Derive output pubkeys for each recipient
    //
    // BIP-352: if the same scan key appears multiple times, use k = 0, 1, 2, …
    // Track the next k value per scan pubkey (as a hex key in a small vec).
    let mut scan_key_counters: Vec<(String, u32)> = Vec::new();
    let mut outputs = Vec::with_capacity(recipients.len());

    for recipient in &recipients {
        let (scan_pk, spend_pk) = parse_sp_address(&recipient.address)?;
        let scan_pk_hex = hex::encode(scan_pk.serialize());

        // BIP-352: same scan key in one tx → increment k counter
        let k = match scan_key_counters
            .iter_mut()
            .find(|(h, _)| h == &scan_pk_hex)
        {
            Some(entry) => {
                let k = entry.1;
                entry.1 += 1;
                k
            }
            None => {
                scan_key_counters.push((scan_pk_hex, 1));
                0
            }
        };

        // shared_secret = partial_secret × B_scan
        // shared_secret_point is constant-time (important since we're using a private key)
        let raw_ecdh = shared_secret_point(&scan_pk, &partial_scalar);
        let ecdh_pubkey = uncompressed_to_pubkey(&raw_ecdh)?;
        // t_k = H_BIP0352/SharedSecret(compressed_ecdh || k)
        // P_k = B_spend + t_k × G
        let t_k = compute_t_n(&ecdh_pubkey, k)?;
        let p_k = compute_p_n(&spend_pk, &t_k)?;

        outputs.push(OutputWithKey {
            pubkey_hex: hex::encode(p_k.serialize()),
            recipient_address: recipient.address.clone(),
        });
    }

    Ok(outputs)
}

/// Build the tsp1q.../sp1q... address from the keys stored in the scanner.
/// Used to form the server-side subscription parameters (without needing the spend secret).
/// In Frigate for example, this is used to generate the sp1q... address.
#[uniffi::export]
pub fn build_sp_address(
    scan_secret_hex: String,
    spend_pubkey_hex: String,
    network: Network,
) -> Result<HexStringResult, SilentPaymentError> {
    let secp = Secp256k1::new();
    let scan_secret = SecretKey::from_slice(&hex_to_32_bytes(&scan_secret_hex)?)
        .map_err(|e| SilentPaymentError::InvalidKey { msg: e.to_string() })?;
    let scan_public = PublicKey::from_secret_key(&secp, &scan_secret);
    let spend_public = PublicKey::from_slice(&hex_to_33_bytes(&spend_pubkey_hex)?)
        .map_err(|e| SilentPaymentError::InvalidKey { msg: e.to_string() })?;

    let sp_addr = SpAddress::new(
        scan_public,
        spend_public,
        to_sp_network(&network),
        SpVersion::ZERO,
    );
    let address: String = sp_addr.into();
    Ok(HexStringResult { value: address })
}

/// Compute the tweak data (33-byte compressed pubkey) a Frigate-equivalent server would serve.
/// Mathematical identity: tweak_data = partial_secret × G = (a_sum × input_hash) × G = a_sum × (input_hash × G) = A_sum × input_hash. This is the tweak data
/// In production: the tweak index server computes and caches this.
/// In tests/demos: use this to bridge send and scan without a server.
#[uniffi::export]
pub fn compute_sender_tweak_data(
    inputs: Vec<SendingInput>,
) -> Result<HexStringResult, SilentPaymentError> {
    if inputs.is_empty() {
        return Err(SilentPaymentError::InvalidKey {
            msg: "inputs must not be empty".into(),
        });
    }

    let mut sp_keys: Vec<(SecretKey, bool)> = Vec::with_capacity(inputs.len());
    let mut sp_outpoints: Vec<SpOutPoint> = Vec::with_capacity(inputs.len());

    for input in &inputs {
        let key = SecretKey::from_slice(&hex_to_32_bytes(&input.secret_key_hex)?)
            .map_err(|e| SilentPaymentError::InvalidKey { msg: e.to_string() })?;
        sp_keys.push((key, input.is_taproot));

        let outpoint = SpOutPoint::from_txid_and_vout(input.txid.clone(), input.vout)
            .map_err(|e| SilentPaymentError::EncodingError { msg: e.to_string() })?;
        sp_outpoints.push(outpoint);
    }

    // partial_secret = a_sum × input_hash  (scalar, via calculate_partial_secret)
    let partial_secret = calculate_partial_secret(&sp_keys, &sp_outpoints)
        .map_err(|e| SilentPaymentError::CryptoError { msg: e.to_string() })?;
    let partial_scalar = SecretKey::from_slice(&partial_secret.secret_bytes())
        .map_err(|e| SilentPaymentError::CryptoError { msg: e.to_string() })?;

    // tweak_data = partial_secret × G
    // This is the public key the mobile scanner receives from the index server
    let secp = Secp256k1::new();
    let tweak_pubkey = PublicKey::from_secret_key(&secp, &partial_scalar);
    Ok(HexStringResult {
        value: hex::encode(tweak_pubkey.serialize()),
    })
}

/// Full BIP-352 keypair (scan secret + spend secret).
/// Generates SP addresses, derive labeled sub-addresses, export keys
/// for SilentPaymentScanner.
/// NOTE: Holds private keys, dispose() when done.
#[derive(uniffi::Object)]
pub struct SilentPaymentRecipient {
    scan_secret: SecretKey,
    spend_secret: SecretKey,
    scan_public: PublicKey,
    spend_public: PublicKey,
    network: Network,
}

#[uniffi::export]
impl SilentPaymentRecipient {
    /// Generate a random BIP-352 keypair. For tests/demos only.
    /// For real wallets use mnemonic/descriptor.
    #[uniffi::constructor]
    pub fn generate(network: Network) -> Result<Arc<Self>, SilentPaymentError> {
        let secp = Secp256k1::new();
        let mut rng = secp256k1::rand::thread_rng();
        let scan_secret = SecretKey::new(&mut rng);
        let spend_secret = SecretKey::new(&mut rng);
        let scan_public = PublicKey::from_secret_key(&secp, &scan_secret);
        let spend_public = PublicKey::from_secret_key(&secp, &spend_secret);
        Ok(Arc::new(Self {
            scan_secret,
            spend_secret,
            scan_public,
            spend_public,
            network,
        }))
    }

    /// Restore from raw private key hex strings.
    #[uniffi::constructor]
    pub fn from_secret_keys(
        scan_secret_hex: String,
        spend_secret_hex: String,
        network: Network,
    ) -> Result<Arc<Self>, SilentPaymentError> {
        let secp = Secp256k1::new();
        let scan_secret =
            SecretKey::from_slice(&hex_to_32_bytes(&scan_secret_hex)?).map_err(|e| {
                SilentPaymentError::InvalidKey {
                    msg: format!("Invalid scan key: {e}"),
                }
            })?;
        let spend_secret =
            SecretKey::from_slice(&hex_to_32_bytes(&spend_secret_hex)?).map_err(|e| {
                SilentPaymentError::InvalidKey {
                    msg: format!("Invalid spend key: {e}"),
                }
            })?;
        let scan_public = PublicKey::from_secret_key(&secp, &scan_secret);
        let spend_public = PublicKey::from_secret_key(&secp, &spend_secret);
        Ok(Arc::new(Self {
            scan_secret,
            spend_secret,
            scan_public,
            spend_public,
            network,
        }))
    }

    /// Returns the bech32m BIP-352 Standard (unlabeled) address.
    /// SpAddress::new() returns Self (not Result),
    /// then Into<String> via encode feature.
    ///
    /// Delegates entirely to SpAddress.
    /// Mainnet sp1q...
    /// Testnet/Signet  tsp1q...
    /// Regtest sprt1q...
    pub fn get_address(&self) -> SilentPaymentAddress {
        let sp_addr = SpAddress::new(
            self.scan_public,
            self.spend_public,
            to_sp_network(&self.network),
            SpVersion::ZERO,
        );
        // SpAddress implements Into<String> (bech32m via the library)
        let address: String = sp_addr.into();
        SilentPaymentAddress {
            address,
            scan_pubkey_hex: hex::encode(self.scan_public.serialize()),
            spend_pubkey_hex: hex::encode(self.spend_public.serialize()),
            network: self.network,
        }
    }

    /// Derive a labeled sub-address for label m.
    ///
    /// BIP-352 label derivation:
    ///   label_pubkey = H_BIP0352/Label(b_scan || m) × G
    ///   B_m = B_spend + label_pubkey
    ///   address = bech32m(hrp, version || B_scan || B_m)
    ///
    /// Share this address with a specific payer. All payments to it are
    /// detected in a single scan pass alongside unlabeled payments.
    ///
    /// Label 0 is reserved by BIP-352 convention. Use labels >= 1.
    pub fn get_labeled_address(
        &self,
        label: u32,
    ) -> Result<SilentPaymentAddress, SilentPaymentError> {
        let label_pubkey = compute_label_pubkey(&self.scan_secret, label)?;
        let b_m = PublicKey::combine_keys(&[&self.spend_public, &label_pubkey]).map_err(|e| {
            SilentPaymentError::CryptoError {
                msg: format!("Failed to derive labeled spend key for m={label}: {e}"),
            }
        })?;

        let sp_addr = SpAddress::new(
            self.scan_public,
            b_m,
            to_sp_network(&self.network),
            SpVersion::ZERO,
        );
        // Encode with the library's bech32m implementation
        let address: String = sp_addr.into();

        Ok(SilentPaymentAddress {
            address,
            scan_pubkey_hex: hex::encode(self.scan_public.serialize()),
            spend_pubkey_hex: hex::encode(b_m.serialize()), // B_m, not B_spend
            network: self.network,
        })
    }

    /// Scan private key hex. Pass to Frigate subscribe(scanSecretHex: ...).
    /// NOTE: This is private. Store in secure_storage.
    pub fn export_scan_secret_hex(&self) -> HexStringResult {
        HexStringResult {
            value: hex::encode(self.scan_secret.secret_bytes()),
        }
    }

    /// Spend private key hex. Needed to sign SP output spending transactions.
    /// NOTE: This is private. Store in secure_storage.
    pub fn export_spend_secret_hex(&self) -> HexStringResult {
        HexStringResult {
            value: hex::encode(self.spend_secret.secret_bytes()),
        }
    }

    /// Spend PUBLIC key hex. Safe to share with Frigate.
    /// Pass to Frigate subscribe(spendPubkeyHex: ...).
    pub fn export_spend_pubkey_hex(&self) -> HexStringResult {
        HexStringResult {
            value: hex::encode(self.spend_public.serialize()),
        }
    }
}

/// Watch-only BIP-352 scanner. This holds scan secret + spend pubkey only.
/// Cannot spend, it's safe for background services and servers.
#[derive(uniffi::Object)]
pub struct SilentPaymentScanner {
    scan_secret: SecretKey,
    spend_public: PublicKey,
}

#[uniffi::export]
impl SilentPaymentScanner {
    /// Create from scan secret + spend public key.
    /// Use SilentPaymentRecipient.export_*() methods to get these values.
    #[uniffi::constructor]
    pub fn watch_only(
        scan_secret_hex: String,
        spend_pubkey_hex: String,
    ) -> Result<Arc<Self>, SilentPaymentError> {
        let scan_secret =
            SecretKey::from_slice(&hex_to_32_bytes(&scan_secret_hex)?).map_err(|e| {
                SilentPaymentError::InvalidKey {
                    msg: format!("Invalid scan key: {e}"),
                }
            })?;
        let spend_public =
            PublicKey::from_slice(&hex_to_33_bytes(&spend_pubkey_hex)?).map_err(|e| {
                SilentPaymentError::InvalidKey {
                    msg: format!("Invalid spend pubkey: {e}"),
                }
            })?;
        Ok(Arc::new(Self {
            scan_secret,
            spend_public,
        }))
    }

    /// Scan a transaction for SP payments using pre-computed tweak data.
    ///
    /// # Arguments
    ///
    /// * `sender_input_pubkeys_hex` — In mobile/light-client mode, this is a
    ///   **single** 33-byte compressed pubkey: the pre-computed tweak data
    ///   `A_sum × H_BIP0352/Inputs(smallest_outpoint || A_sum)` returned by
    ///   the tweak index server. The server does the input hash computation;
    ///   we only do the ECDH + output derivation.
    ///
    ///   In full-verification mode (no server), pass the raw sender input
    ///   pubkeys and use [`scan_transaction_full`] which also takes outpoints.
    ///
    /// * `tx_output_pubkeys_hex` — 33-byte compressed pubkeys from tx outputs.
    ///
    /// * `output_amounts_sats` — Parallel with output pubkeys.
    ///
    /// # BIP-352 computation (correct)
    ///
    /// ```text
    /// shared_secret = b_scan × tweak_data
    /// t_k = H_BIP0352/SharedSecret(shared_secret.serialize() || k.to_be_bytes())
    /// P_k = B_spend + t_k × G
    /// match P_k against each output pubkey
    /// ```
    pub fn scan_transaction(
        &self,
        sender_input_pubkeys_hex: Vec<String>,
        tx_output_pubkeys_hex: Vec<String>,
    ) -> Result<ScanTransactionResult, SilentPaymentError> {
        // In mobile mode, sender_input_pubkeys_hex contains a single entry:
        // the pre-computed tweak from the index server.
        if sender_input_pubkeys_hex.is_empty() {
            return Err(SilentPaymentError::InvalidKey {
                msg: "sender_input_pubkeys_hex must contain at least one entry (the Frigate tweak key)".into(),
            });
        }

        // Step 1: ECDH - Parse the tweak data
        // sender_input_pubkeys_hex[0] = tweak data from index server
        //   = A_sum × H_BIP0352/Inputs(smallest_outpoint || A_sum)
        // (server pre-computed the input hash; we only do ECDH + output scan)
        let tweak_data = parse_pubkey_33(&sender_input_pubkeys_hex[0])?;
        // Step 2: ECDH - b_scan × tweak_data
        // shared_secret_point: constant-time scalar multiplication
        // Returns 64 bytes (x,y coordinates of the EC point)
        let ecdh_pubkey = ecdh_point(&tweak_data, &self.scan_secret)?;

        let mut payments = Vec::new();
        let mut k: u32 = 0;

        // t_k = H_BIP0352/SharedSecret(compressed_ecdh_point || k)
        // P_k (expected_pk) = B_spend + t_k × G
        // Match P_k against each output pubkey; stop when no match found.
        loop {
            let t_k = compute_t_n(&ecdh_pubkey, k)?;
            let expected_pk = compute_p_n(&self.spend_public, &t_k)?;
            let (full, xonly) = pubkey_to_hex_pair(&expected_pk);

            let matched_vout = match_output(&tx_output_pubkeys_hex, &full, &xonly);
            if let Some(vout) = matched_vout {
                payments.push(FoundPayment {
                    output_index: vout,
                    tweak_hex: hex::encode(t_k.secret_bytes()),
                    amount_sats: 0,
                    label: None,
                });
                k += 1;
            } else {
                break;
            }
        }
        Ok(ScanTransactionResult { payments })
    }

    /// Scan for both standard AND labeled payments in a single pass.
    /// More efficient than calling scan_transaction per label — label pubkeys
    /// are computed once (outside the k loop).
    ///
    /// # Performance
    /// Label pubkeys are pre-computed once per scan call (not per output).
    /// Cost: O(outputs × labels) point comparisons per k step.
    /// For typical wallets (< 50 labels, < 100 outputs), this is negligible.
    ///
    /// # Arguments
    /// * `labels` — label integers to check (e.g. `[1, 2, 3]`).
    ///   Pass an empty vec to behave identically to `scan_transaction`.
    ///
    /// # Returns
    /// `FoundPayment.label` tells you WHICH sub-address received the payment.
    pub fn scan_transaction_with_labels(
        &self,
        sender_input_pubkeys_hex: Vec<String>,
        tx_output_pubkeys_hex: Vec<String>,
        labels: Vec<u32>,
    ) -> Result<ScanTransactionResult, SilentPaymentError> {
        if sender_input_pubkeys_hex.is_empty() {
            return Err(SilentPaymentError::InvalidKey {
                msg: "sender_input_pubkeys_hex must not be empty".into(),
            });
        }

        // ECDH (identical to scan_transaction)
        let tweak_data = parse_pubkey_33(&sender_input_pubkeys_hex[0])?;
        let ecdh_pubkey = ecdh_point(&tweak_data, &self.scan_secret)?;

        // Pre-compute label pubkeys once, reused for every k iteration
        // label_pubkeys[i] = (m, H_BIP0352/Label(b_scan || m) × G)
        let label_pubkeys: Vec<(u32, PublicKey)> = labels
            .iter()
            .map(|&m| compute_label_pubkey(&self.scan_secret, m).map(|pk| (m, pk)))
            .collect::<Result<_, _>>()?;

        let mut payments = Vec::new();
        let mut k: u32 = 0;

        loop {
            // t_k = H_BIP0352/SharedSecret(compressed_ecdh || k)
            // P_k = B_spend + t_k × G   (unlabeled expected output)
            let t_k = compute_t_n(&ecdh_pubkey, k)?;
            let p_k = compute_p_n(&self.spend_public, &t_k)?;
            let (p_k_full, p_k_xonly) = pubkey_to_hex_pair(&p_k);

            let mut found_this_k = false;

            // Check unlabeled
            if let Some(vout) = match_output(&tx_output_pubkeys_hex, &p_k_full, &p_k_xonly) {
                payments.push(FoundPayment {
                    output_index: vout,
                    tweak_hex: hex::encode(t_k.secret_bytes()),
                    amount_sats: 0,
                    label: None,
                });
                found_this_k = true;
            }

            // Check each label:
            // P_k_m = P_k + label_pubkey_m
            //       = (B_spend + t_k × G) + (label_hash_m × G)
            //       = B_m + t_k × G       where B_m = B_spend + label_hash_m × G
            for &(m, ref label_pk) in &label_pubkeys {
                let p_k_m = PublicKey::combine_keys(&[&p_k, label_pk]).map_err(|e| {
                    SilentPaymentError::CryptoError {
                        msg: format!("Point addition failed for label {m}: {e}"),
                    }
                })?;
                let (p_k_m_full, p_k_m_xonly) = pubkey_to_hex_pair(&p_k_m);

                // Check labeled
                if let Some(vout) = match_output(&tx_output_pubkeys_hex, &p_k_m_full, &p_k_m_xonly)
                {
                    payments.push(FoundPayment {
                        output_index: vout,
                        tweak_hex: hex::encode(t_k.secret_bytes()),
                        amount_sats: 0,
                        label: Some(m),
                    });
                    found_this_k = true;
                }
            }
            // Stop scanning when the current k produces no match
            // (neither unlabeled nor any label). Further k values won't match.
            if !found_this_k {
                break;
            }
            k += 1;
        }

        // Return payments in the order they appeared in the transaction
        // Sort by output index to match expected order
        //payments.sort_by_key(|p| p.output_index);

        Ok(ScanTransactionResult { payments })
    }
}

/// t_k = H_BIP0352/SharedSecret(compressed_ecdh_point || k.to_be_bytes())
fn compute_t_n(ecdh_pubkey: &PublicKey, k: u32) -> Result<SecretKey, SilentPaymentError> {
    let mut engine = SpSharedSecretHash::engine();
    engine.input(&ecdh_pubkey.serialize());
    engine.input(&k.to_be_bytes());
    let hash = SpSharedSecretHash::from_engine(engine).to_byte_array();
    SecretKey::from_slice(&hash).map_err(|e| SilentPaymentError::CryptoError { msg: e.to_string() })
}

/// P_k = B_spend + t_k × G
fn compute_p_n(spend_public: &PublicKey, t_k: &SecretKey) -> Result<PublicKey, SilentPaymentError> {
    let secp = Secp256k1::verification_only();
    let scalar =
        Scalar::from_be_bytes(t_k.secret_bytes()).map_err(|_| SilentPaymentError::CryptoError {
            msg: "t_k produced out-of-range scalar".into(),
        })?;
    spend_public
        .add_exp_tweak(&secp, &scalar)
        .map_err(|e| SilentPaymentError::CryptoError { msg: e.to_string() })
}

/// label_pubkey_m = H_BIP0352/Label(b_scan || m) × G
fn compute_label_pubkey(b_scan: &SecretKey, m: u32) -> Result<PublicKey, SilentPaymentError> {
    let secp = Secp256k1::new();
    let mut engine = BIP0352LabelHash::engine();
    engine.input(&b_scan.secret_bytes());
    engine.input(&m.to_be_bytes());
    let hash = BIP0352LabelHash::from_engine(engine).to_byte_array();
    let label_scalar =
        SecretKey::from_slice(&hash).map_err(|e| SilentPaymentError::CryptoError {
            msg: format!("Label {m} hash is an invalid scalar: {e}"),
        })?;
    Ok(PublicKey::from_secret_key(&secp, &label_scalar))
}

// HELPERS

/// Run ECDH and convert the raw 64-byte output to a compressed PublicKey.
/// shared_secret_point returns (x,y) without a prefix byte; we reconstruct
/// the full uncompressed form (0x04 || x || y) before parsing.
fn ecdh_point(pubkey: &PublicKey, scalar: &SecretKey) -> Result<PublicKey, SilentPaymentError> {
    // Step 2: ECDH - b_scan × tweak_data
    // shared_secret_point: constant-time scalar multiplication
    // Returns 64 bytes (x,y coordinates of the EC point)
    let raw_ecdh = shared_secret_point(pubkey, scalar);
    uncompressed_to_pubkey(&raw_ecdh)
}

/// Reconstruct the full uncompressed EC point (0x04 || x || y)
/// then parse so we can call .serialize() for the 33-byte compressed form
fn uncompressed_to_pubkey(raw_ecdh: &[u8; 64]) -> Result<PublicKey, SilentPaymentError> {
    let mut uncompressed = [0u8; 65];
    uncompressed[0] = 0x04;
    uncompressed[1..].copy_from_slice(raw_ecdh);
    PublicKey::from_slice(&uncompressed).map_err(|e| SilentPaymentError::CryptoError {
        msg: format!("Failed to reconstruct ECDH point: {e}"),
    })
}

/// Parse a 33-byte compressed pubkey from a 66-char hex string.
fn parse_pubkey_33(hex_str: &str) -> Result<PublicKey, SilentPaymentError> {
    let bytes = hex_to_33_bytes(hex_str)?;
    PublicKey::from_slice(&bytes).map_err(|e| SilentPaymentError::InvalidKey {
        msg: format!("Invalid compressed pubkey: {e}"),
    })
}

/// Produce both hex representations of a compressed pubkey:
///   full  = "02..." or "03..." (66 chars)
///   xonly = the 64-char x-coordinate without the prefix byte
fn pubkey_to_hex_pair(pk: &PublicKey) -> (String, String) {
    let bytes = pk.serialize();
    (hex::encode(&bytes), hex::encode(&bytes[1..]))
}

/// Find the first output pubkey matching full or xonly form.
fn match_output(outputs: &[String], full: &str, xonly: &str) -> Option<u32> {
    outputs
        .iter()
        .enumerate()
        .find(|(_, pk)| pk.as_str() == full || pk.as_str() == xonly) // 66-char compressed match OR 64-char x-only match (taproot uses x-only) *pk_hex == &expected_full || *pk_hex == &expected_xonly
        .map(|(i, _)| i as u32)
}

fn hex_to_32_bytes(s: &str) -> Result<[u8; 32], SilentPaymentError> {
    hex::decode(s)
        .map_err(|e| SilentPaymentError::EncodingError { msg: e.to_string() })?
        .try_into()
        .map_err(|_| SilentPaymentError::EncodingError {
            msg: format!("Expected 32 bytes (64 hex chars), got: {s}"),
        })
}

fn hex_to_33_bytes(s: &str) -> Result<[u8; 33], SilentPaymentError> {
    hex::decode(s)
        .map_err(|e| SilentPaymentError::EncodingError { msg: e.to_string() })?
        .try_into()
        .map_err(|_| SilentPaymentError::EncodingError {
            msg: format!("Expected 33 bytes (66 hex chars), got: {s}"),
        })
}

/// Parse a bech32m BIP-352 address using the sp_lib library.
/// Delegates to the library's TryFrom<&str> implementation which validates:
/// - bech32m checksum
/// - correct HRP (sp / tsp / sprt)
/// - correct payload length (107 base32 chars = version + 33 + 33 bytes)
/// - correct version byte (0)
fn parse_sp_address(addr: &str) -> Result<(PublicKey, PublicKey), SilentPaymentError> {
    let sp_addr = SpAddress::try_from(addr)
        .map_err(|e| SilentPaymentError::InvalidAddress { msg: e.to_string() })?;
    Ok((sp_addr.get_scan_key(), sp_addr.get_spend_key()))
}
