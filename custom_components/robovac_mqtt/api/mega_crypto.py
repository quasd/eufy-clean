"""Eufy "eufy_mega" v6 transport crypto.

Python port of the eufy_mega ECDH transport. Two upstream implementations were
used, and they agree on every derivation:

* ``src/core/crypto.ts`` in **mega-yfue/eufy-sdk** (Apache License 2.0) —
  https://github.com/mega-yfue/eufy-sdk — actively maintained, and the source
  of the per-category bootstrap keys below.
* ``src/http/megaCrypto.ts`` in **bropat/eufy-security-client** (MIT) —
  https://github.com/bropat/eufy-security-client — archived 2026-09-12; the
  structure of this module follows it.

Modification notice (Apache-2.0 §4b): this file is a translation of the above
TypeScript into Python for Home Assistant; no algorithmic changes were made.

    Copyright (c) 2026 mega-yfue (Apache License 2.0)
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

    Copyright (c) 2021-2024 bropat <patrick.broetto@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The behaviour is a faithful translation; the structure follows the original so
the two stay comparable as upstream evolves.

Two layers:
  1. Bootstrap (handshake): body + signature use a STATIC per-app ``preset key``.
  2. Regular requests (post-handshake): body + signature use the per-cluster
     ECDH ``shared key`` derived from the key/exchange.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from base64 import b64decode, b64encode
from dataclasses import dataclass

from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# The eufy_mega app bootstrap key: one AES-128 key wrapping the ECDH exchange
# across the whole mega host stack (openapi + passport + app-*). Verified here
# against captured key/exchange responses from app-openapi-eu-pr.eufy.com and
# app-openapi-us-pr.eufy.com, both of which decrypt to valid P-256 points.
MEGA_PRESET_KEY = "2500a7d5617812f9d52515b2c8f20a3d"

# A SEPARATE bootstrap key for the `*.eufylife.com` gateway, which runs its own
# key exchange at /v3/openapi/oauth/key/exchange and rejects the mega key.
# Verified the same way against a captured security-app-eu.eufylife.com
# response. Unused by the robovac path, which lives entirely on the mega hosts,
# but kept so a eufylife host can be addressed without rediscovering this.
EUFYLIFE_PRESET_KEY = "118c12c81e211149304bd70a0c071d01"

# NIST P-256 (prime256v1).
_CURVE = ec.SECP256R1()

_SERVER_PUB_RE = re.compile(r"^04[0-9a-f]{128}$", re.IGNORECASE)


def x_signature(
    key_ascii: str, ts: str, nonce: str, encrypted_body: str | None = None
) -> str:
    """X-Signature: HMAC-SHA256 hex over ``ts+nonce[+encrypted_body]``.

    The HMAC key is the **ASCII string** of the key material, not its hex
    decoding — passing bytes.fromhex() here yields a signature the server
    rejects.
    """
    parts = [ts, nonce] if encrypted_body is None else [ts, nonce, encrypted_body]
    return hmac.new(
        key_ascii.encode("utf-8"), "+".join(parts).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def generate_key_ident() -> str:
    """Random 32-hex client-generated X-Key-Ident (one per cluster identity)."""
    return os.urandom(16).hex()


def _aes_cbc_encrypt(plaintext: str, key: bytes) -> str:
    """AES-128-CBC/PKCS7 with a fresh random IV → ``base64(IV ++ ciphertext)``."""
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    data = padder.update(plaintext.encode("utf-8")) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return b64encode(iv + encryptor.update(data) + encryptor.finalize()).decode("ascii")


def _aes_cbc_decrypt(b64: str, key: bytes) -> str:
    """Inverse of :func:`_aes_cbc_encrypt`."""
    blob = b64decode(b64)
    decryptor = Cipher(algorithms.AES(key), modes.CBC(blob[:16])).decryptor()
    padded = decryptor.update(blob[16:]) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode("utf-8")


def preset_encrypt(plaintext: str, preset_key_hex: str = MEGA_PRESET_KEY) -> str:
    """Wrap a payload under the static preset key (used for the EC public key)."""
    return _aes_cbc_encrypt(plaintext, bytes.fromhex(preset_key_hex))


def preset_decrypt(b64: str, preset_key_hex: str = MEGA_PRESET_KEY) -> str:
    """Inverse of :func:`preset_encrypt`."""
    return _aes_cbc_decrypt(b64, bytes.fromhex(preset_key_hex))


@dataclass
class MegaIdentity:
    """Result of a key/exchange handshake, cached per cluster host."""

    key_ident: str
    #: ECDH shared secret as lowercase hex (raw 32-byte X coordinate).
    shared_key: str
    #: Our ephemeral public key (uncompressed ``04…``, hex) sent in the exchange.
    client_public_key: str


@dataclass
class KeyExchangeRequest:
    """Material for a key/exchange call; keeps the private key for finalisation."""

    private_key: ec.EllipticCurvePrivateKey
    client_public_key_body: str
    client_public_key: str
    key_ident: str


def build_key_exchange() -> KeyExchangeRequest:
    """Build the key/exchange request material.

    The exchange is ECIES-bootstrapped: the ephemeral EC public key is wrapped
    with the static preset key. The session shared key is not derivable until
    the server replies — see :func:`finalize_key_exchange`.
    """
    private_key = ec.generate_private_key(_CURVE)
    client_pub_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    ).hex()
    return KeyExchangeRequest(
        private_key=private_key,
        client_public_key_body=preset_encrypt(client_pub_hex),
        client_public_key=client_pub_hex,
        key_ident=generate_key_ident(),
    )


def finalize_key_exchange(
    request: KeyExchangeRequest, server_public_key_enc: str
) -> MegaIdentity:
    """Derive the session shared key from the server's (preset-wrapped) key."""
    server_pub_hex = preset_decrypt(server_public_key_enc)
    if not _SERVER_PUB_RE.match(server_pub_hex):
        raise ValueError("key/exchange: unexpected server public key format")
    server_key = ec.EllipticCurvePublicKey.from_encoded_point(
        _CURVE, bytes.fromhex(server_pub_hex)
    )
    shared = request.private_key.exchange(ec.ECDH(), server_key)
    return MegaIdentity(
        key_ident=request.key_ident,
        shared_key=shared.hex(),
        client_public_key=request.client_public_key,
    )


def shared_key_signing_key(shared_key_hex: str) -> str:
    """HMAC key for regular requests: first 32 hex CHARS, used as ASCII."""
    return shared_key_hex[:32]


def shared_key_to_aes_key(shared_key_hex: str) -> bytes:
    """AES body key: ``bytes.fromhex(shared_key[:32])`` → 16 bytes (AES-128)."""
    return bytes.fromhex(shared_key_hex[:32])


def mega_encrypt_body(plaintext: str, aes_key: bytes) -> str:
    """Encrypt a post-handshake body — same envelope as the key/exchange body."""
    return _aes_cbc_encrypt(plaintext, aes_key)


def mega_decrypt_body(b64: str, aes_key: bytes) -> str:
    """Inverse of :func:`mega_encrypt_body`."""
    return _aes_cbc_decrypt(b64, aes_key)
