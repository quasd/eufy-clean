"""Eufy "eufy_mega" v6 HTTP transport — login and device discovery.

Ported to Python from ``src/http/megaApi.ts`` (and ``encryptAPIData`` from
``src/http/utils.ts``) in **bropat/eufy-security-client** (MIT, archived
2026-09-12). The request envelope, signing rules and the login server key were
cross-checked against ``src/core/crypto.ts`` and ``src/transport/http/
mega-client.ts`` in **mega-yfue/eufy-sdk** (Apache License 2.0,
https://github.com/mega-yfue/eufy-sdk), which is the maintained implementation
of the same protocol and agrees on every derivation.

Modification notice (Apache-2.0 §4b): this file is a translation into Python
for Home Assistant; no algorithmic changes were made.

    Copyright (c) 2021-2024 bropat <patrick.broetto@gmail.com>
    https://github.com/bropat/eufy-security-client

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

Only the slice this integration needs is ported: estimate_domain, the
key/exchange handshake, signed requests, login, and the two calls that matter
for a robovac (get_devs_list and get_user_mqtt_info). Push registration, FCM,
captcha image fetching and the legacy-fallback state machine are not included.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from base64 import b64encode
from typing import Any

import aiohttp
from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .mega_crypto import (
    MEGA_PRESET_KEY,
    MegaIdentity,
    build_key_exchange,
    finalize_key_exchange,
    generate_key_ident,
    mega_decrypt_body,
    mega_encrypt_body,
    shared_key_signing_key,
    shared_key_to_aes_key,
    x_signature,
)

_LOGGER = logging.getLogger(__name__)

_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Login password ECDH server key — the same constant the legacy API uses.
LOGIN_SERVER_PUBLIC_KEY = (
    "04c5c00c4f8d1197cc7c3167c52bf7acb054d722f0ef08dcd7e0883236e0d72a"
    "3868d9750cb47fa4619248f3d83f0f662671dadc6e2d31c2f41db0161651c7c076"
)

# Response codes the caller acts on.
CODE_OK = 0
CODE_NEED_VERIFY_CODE = 26052        # email 2FA required
CODE_LOGIN_NEED_CAPTCHA = 100032     # picture captcha required
CODE_LOGIN_CAPTCHA_ERROR = 100033    # captcha answer wrong
CODE_NEED_NEGOTIATE_KEY = 4404       # identity stale — re-handshake
CODE_SIGNATURE_ERROR = 4416          # signature rejected — re-handshake

# The Eufy WAF rate-limits aggressive probing; upstream throttles to 1 request
# per ~3s and we keep that.
_MIN_REQUEST_INTERVAL = 3.0


class MegaApiError(Exception):
    """A v6 backend call failed."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class MegaLoginRequiresVerification(MegaApiError):
    """Login needs an email 2FA code or a captcha answer.

    Home Assistant's config flow has no step for either, so this surfaces as a
    clear error telling the user to fall back to pasting a token.
    """


def encrypt_login_password(password: str) -> tuple[str, str]:
    """Encrypt the password for /passport/login.

    ECDH against the static login server key; the raw shared secret is the
    AES-256 key and its first 16 bytes are the IV. Returns
    ``(encrypted_b64, client_public_key_hex)``.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    server_key = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), bytes.fromhex(LOGIN_SERVER_PUBLIC_KEY)
    )
    secret = private_key.exchange(ec.ECDH(), server_key)

    padder = padding.PKCS7(128).padder()
    data = padder.update(password.encode("utf-8")) + padder.finalize()
    encryptor = Cipher(algorithms.AES(secret), modes.CBC(secret[:16])).encryptor()
    encrypted = b64encode(encryptor.update(data) + encryptor.finalize()).decode("ascii")

    client_pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    ).hex()
    return encrypted, client_pub


class MegaHTTPApi:
    """Minimal client for the unified-app (eufy_mega) v6 backend."""

    def __init__(
        self,
        websession: aiohttp.ClientSession,
        openudid: str,
        ab: str = "eu",
        app_name: str = "eufy_mega",
        app_version: str = "6.0.90_29798",
        os_version: str = "29",
        phone_model: str = "Pixel 8",
    ) -> None:
        self._session = websession
        self.openudid = openudid
        self.ab = ab.lower()
        self.app_name = app_name
        self.app_version = app_version
        self.os_version = os_version
        self.phone_model = phone_model

        self._mega_domain = ""
        self._domains: dict[str, str] = {}
        self._identities: dict[str, MegaIdentity] = {}
        self.auth_token: str | None = None
        self.user_id: str | None = None
        self.token_expires_at: int | None = None
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ utils

    @property
    def gtoken(self) -> str | None:
        if not self.user_id:
            return None
        import hashlib

        return hashlib.md5(self.user_id.encode()).hexdigest()

    def cluster_host(self, service: str) -> str:
        """Region cluster host for a service, e.g. passport → app-passport-eu-pr.

        Derived from the domain estimate_domain returned — the server decides
        the region, we do not guess it. The us/eu fallback only applies before
        estimate_domain has run.
        """
        if self._mega_domain.startswith("mega-"):
            return self._mega_domain.replace("mega-", f"app-{service}-", 1)
        region = "us" if self.ab == "us" else "eu"
        return f"app-{service}-{region}-pr.eufy.com"

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < _MIN_REQUEST_INTERVAL:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    # --------------------------------------------------------------- requests

    async def _signed_post(
        self,
        host: str,
        path: str,
        payload: Any = None,
        identity: MegaIdentity | None = None,
        bootstrap: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Signed/encrypted POST to a v6 host.

        X-Signature covers the ENCRYPTED VALUE, not the JSON wrapper: for
        key/exchange the body is ``{"client_public_key": "<b64>"}`` but only
        ``<b64>`` is signed; for regular requests the body is the ciphertext, so
        the signed value and the body are the same string.

        No retry here — the timestamp, nonce and signature are computed once per
        call and the backend enforces a replay window, so resending a frozen
        signature would just be rejected.
        """
        ts = str(int(time.time()))
        nonce = generate_key_ident()

        if bootstrap is not None:
            key_ident, encrypted_value = bootstrap
            body = json.dumps({"client_public_key": encrypted_value})
            signed_value = encrypted_value
            signing_key = MEGA_PRESET_KEY
        else:
            if identity is None:
                raise MegaApiError("signed_post: identity required for non-bootstrap request")
            aes_key = shared_key_to_aes_key(identity.shared_key)
            body = mega_encrypt_body(json.dumps(payload), aes_key)
            signed_value = body
            signing_key = shared_key_signing_key(identity.shared_key)
            key_ident = identity.key_ident

        headers = {
            "accept": "application/json",
            "accept-charset": "UTF-8",
            "app-name": self.app_name,
            "app-version": self.app_version,
            "app_version": self.app_version,
            "os-type": "android",
            "os_type": "android",
            "os-version": self.os_version,
            "os_version": self.os_version,
            "model-type": "PHONE",
            "phone-model": self.phone_model,
            "phone_model": self.phone_model,
            "openudid": self.openudid,
            "test-flag": "false",
            "user-agent": "ktor-client",
            "content-type": "application/json",
            "x-encryption-info": "algo_ecdh",
            "x-key-ident": key_ident,
            "x-request-ts": ts,
            "x-request-once": nonce,
            "x-replay-info": "replay",
            "x-signature": x_signature(signing_key, ts, nonce, signed_value),
            "country": self.ab.upper(),
            "language": self.ab,
            "ab_code": self.ab,
        }
        if self.gtoken:
            headers["gtoken"] = self.gtoken
        if self.auth_token:
            headers["x-auth-token"] = self.auth_token
            headers["authorization"] = self.auth_token

        await self._throttle()
        async with self._session.post(
            f"https://{host}{path}",
            timeout=_REQUEST_TIMEOUT,
            headers=headers,
            data=body.encode("utf-8"),
        ) as response:
            text = await response.text()

        try:
            parsed = json.loads(text)
        except ValueError as err:
            raise MegaApiError(
                f"{host}{path} -> HTTP {response.status}, non-JSON body "
                f"({len(text)} bytes)"
            ) from err

        if parsed.get("code") in (CODE_NEED_NEGOTIATE_KEY, CODE_SIGNATURE_ERROR):
            _LOGGER.info(
                "eufy_mega identity rejected (code=%s); evicting cached identities",
                parsed.get("code"),
            )
            self._identities.clear()
        return parsed

    async def estimate_domain(self) -> dict[str, str]:
        """Resolve the region's mega domain and product domains (cleartext)."""
        host = f"mega-{'us' if self.ab == 'us' else 'eu'}-pr.eufy.com"
        await self._throttle()
        async with self._session.post(
            f"https://{host}/passport/estimate_domain",
            timeout=_REQUEST_TIMEOUT,
            headers={
                "app-name": self.app_name,
                "app-version": self.app_version,
                "os-type": "android",
                "content-type": "application/json",
            },
            json={"ab": self.ab, "mode": 1},
        ) as response:
            result = await response.json()

        if result.get("code") != CODE_OK:
            raise MegaApiError(
                f"estimate_domain failed: {result.get('code')} {result.get('msg')}",
                result.get("code"),
            )
        data = result.get("data") or {}
        self._mega_domain = data.get("domain", "")
        self._domains = data.get("product_domains", {})
        _LOGGER.debug(
            "eufy_mega estimate_domain: %s (%d product domains)",
            self._mega_domain,
            len(self._domains),
        )
        return self._domains

    async def key_exchange(self, openapi_host: str) -> MegaIdentity:
        """ECDH handshake against a cluster's openapi host; cached per host."""
        cached = self._identities.get(openapi_host)
        if cached:
            return cached

        request = build_key_exchange()
        result = await self._signed_post(
            openapi_host,
            "/openapi/oauth/key/exchange",
            bootstrap=(request.key_ident, request.client_public_key_body),
        )
        if result.get("code") != CODE_OK:
            raise MegaApiError(
                f"key/exchange failed on {openapi_host}: "
                f"{result.get('code')} {result.get('msg')}",
                result.get("code"),
            )
        server_pub = (result.get("data") or {}).get("server_public_key")
        identity = finalize_key_exchange(request, server_pub)
        self._identities[openapi_host] = identity
        _LOGGER.debug("eufy_mega key/exchange ok on %s", openapi_host)
        return identity

    async def call(self, host: str, path: str, payload: Any) -> dict[str, Any]:
        """Signed/encrypted call, establishing the cluster identity if needed."""
        identity = await self.key_exchange(self.cluster_host("openapi"))
        return await self._signed_post(host, path, payload, identity)

    async def call_decrypted(self, service: str, path: str, payload: Any = None) -> Any:
        """:meth:`call` plus decryption of the ``data`` field."""
        identity = await self.key_exchange(self.cluster_host("openapi"))
        result = await self._signed_post(
            self.cluster_host(service), path, payload or {}, identity
        )
        if result.get("code") != CODE_OK:
            raise MegaApiError(
                f"{path} failed: {result.get('code')} {result.get('msg')}",
                result.get("code"),
            )
        data = result.get("data")
        if not isinstance(data, str):
            _LOGGER.warning(
                "eufy_mega response data is not an encrypted string (protocol drift?): %s",
                path,
            )
            return data
        return json.loads(mega_decrypt_body(data, shared_key_to_aes_key(identity.shared_key)))

    # ------------------------------------------------------------------ login

    async def login(self, email: str, password: str) -> None:
        """Log in and store the auth token.

        The backend answers code 0 even when 2FA is still pending — the real
        state is in ``fa_info.step``. Neither 2FA nor captcha can be answered
        from a config-flow form, so both raise
        :class:`MegaLoginRequiresVerification` and the user falls back to
        pasting a token.
        """
        if not self._mega_domain:
            await self.estimate_domain()

        encrypted, client_pub = encrypt_login_password(password)
        payload = {
            "email": email,
            "password": encrypted,
            "ab": self.ab,
            "client_secret_info": {"public_key": client_pub},
            "answer": "",
            "captcha_id": "",
            "verify_code": "",
            "login_id": "",
        }
        result = await self.call(self.cluster_host("passport"), "/passport/login", payload)
        code = result.get("code")

        if code in (CODE_LOGIN_NEED_CAPTCHA, CODE_LOGIN_CAPTCHA_ERROR):
            raise MegaLoginRequiresVerification(
                "Eufy demanded a picture captcha, which cannot be answered here. "
                "Paste a token from the app instead.",
                code,
            )
        if code != CODE_OK or not result.get("data"):
            raise MegaApiError(
                f"eufy_mega login failed: {code} {result.get('msg')}", code
            )

        identity = self._identities[self.cluster_host("openapi")]
        decoded = json.loads(
            mega_decrypt_body(result["data"], shared_key_to_aes_key(identity.shared_key))
        )

        fa_info = decoded.get("fa_info") or {}
        if fa_info.get("step") == CODE_NEED_VERIFY_CODE:
            raise MegaLoginRequiresVerification(
                "Eufy requires an emailed 2FA code, which cannot be entered here. "
                "Paste a token from the app instead.",
                CODE_NEED_VERIFY_CODE,
            )

        self.auth_token = decoded.get("auth_token") or decoded.get("token")
        self.user_id = decoded.get("user_id") or decoded.get("userId")
        expires = decoded.get("token_expires_at")
        if isinstance(expires, int):
            self.token_expires_at = expires
        if not self.auth_token or not self.user_id:
            raise MegaApiError("eufy_mega login returned no usable token")
        _LOGGER.info("eufy_mega login successful")

    def has_valid_session(self) -> bool:
        """True when a non-expired token is held (60s safety margin)."""
        if not self.auth_token or not self.user_id or not self.token_expires_at:
            return False
        return time.time() < self.token_expires_at - 60

    # ---------------------------------------------------------------- queries

    async def get_devs_list(self) -> list[dict[str, Any]]:
        """Eufy-side device list (``house/get_devs_list``), decrypted."""
        data = await self.call_decrypted(
            "house",
            "/app/house/get_devs_list",
            {"device_sn": "", "num": 100, "orderby": ""},
        )
        if isinstance(data, dict):
            return data.get("devices") or []
        return []

    async def get_user_mqtt_info(self) -> dict[str, Any]:
        """MQTT broker endpoint and per-user certificate for the v6 backend."""
        return await self.call_decrypted(
            "devicemanage", "/app/devicemanage/get_user_mqtt_info", {}
        )
