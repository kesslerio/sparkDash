/**
 * Encrypted SSH password store — survives process/Docker restarts.
 *
 * Passwords are NEVER written to sparks.json and NEVER returned by the API.
 * They live in:
 *   - memory (Map) for SSH collectors
 *   - config/sparks-secrets.json (AES-256-GCM ciphertext, volume-mounted)
 *
 * Encryption key:
 *   - SPARKDASH_SECRETS_KEY env (passphrase or 64-char hex), or
 *   - auto-generated config/.secrets-key (persists with ./config volume)
 */
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { SPARKS_SECRETS_PATH, SECRETS_KEY_PATH, SPARKS_LLM_KEYS_PATH } from "./config.js";
import { atomicWrite } from "./util/atomicWrite.js";

const ALGO = "aes-256-gcm";
const IV_LEN = 12;
const TAG_LEN = 16;
const KEY_LEN = 32;

/** Cached key so we never regenerate mid-process. */
let _cachedKey = null;

function ensureDir(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function keyFromString(s) {
  const t = String(s).trim();
  if (/^[0-9a-fA-F]{64}$/.test(t)) return Buffer.from(t, "hex");
  return crypto.createHash("sha256").update(t, "utf8").digest();
}

/**
 * Resolve a 32-byte key.
 * NEVER overwrites an existing key file (that would orphan encrypted secrets).
 * @returns {Buffer}
 */
function resolveKey() {
  if (_cachedKey) return _cachedKey;

  const fromEnv = process.env.SPARKDASH_SECRETS_KEY;
  if (fromEnv && String(fromEnv).trim()) {
    _cachedKey = keyFromString(fromEnv);
    return _cachedKey;
  }

  if (fs.existsSync(SECRETS_KEY_PATH)) {
    try {
      const raw = fs.readFileSync(SECRETS_KEY_PATH, "utf8").trim();
      if (!raw) throw new Error("key file is empty");
      _cachedKey = keyFromString(raw);
      return _cachedKey;
    } catch (err) {
      // Do NOT generate a new key — that would make existing secrets unreadable
      throw new Error(
        `Cannot read secrets key at ${SECRETS_KEY_PATH}: ${err.message}. ` +
          `Fix permissions or set SPARKDASH_SECRETS_KEY.`
      );
    }
  }

  // No key yet — only create one if there is no secrets file either
  if (fs.existsSync(SPARKS_SECRETS_PATH)) {
    throw new Error(
      `Encrypted secrets exist at ${SPARKS_SECRETS_PATH} but key file is missing (${SECRETS_KEY_PATH}). ` +
        `Restore .secrets-key or re-enter passwords after deleting sparks-secrets.json.`
    );
  }

  const key = crypto.randomBytes(KEY_LEN);
  ensureDir(SECRETS_KEY_PATH);
  try {
    fs.writeFileSync(SECRETS_KEY_PATH, key.toString("hex") + "\n", { mode: 0o600 });
    try {
      fs.chmodSync(SECRETS_KEY_PATH, 0o600);
    } catch {
      /* best-effort */
    }
    console.log(`[secretsStore] Generated encryption key at ${SECRETS_KEY_PATH}`);
  } catch (err) {
    throw new Error(`Failed to write secrets key: ${err.message}`);
  }
  _cachedKey = key;
  return key;
}

function encrypt(plaintext, key) {
  const iv = crypto.randomBytes(IV_LEN);
  const cipher = crypto.createCipheriv(ALGO, key, iv);
  const enc = Buffer.concat([cipher.update(String(plaintext), "utf8"), cipher.final()]);
  const tag = cipher.getAuthTag();
  return Buffer.concat([iv, tag, enc]).toString("base64");
}

function decrypt(blobB64, key) {
  const buf = Buffer.from(blobB64, "base64");
  if (buf.length < IV_LEN + TAG_LEN + 1) throw new Error("ciphertext too short");
  const iv = buf.subarray(0, IV_LEN);
  const tag = buf.subarray(IV_LEN, IV_LEN + TAG_LEN);
  const data = buf.subarray(IV_LEN + TAG_LEN);
  const decipher = crypto.createDecipheriv(ALGO, key, iv);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(data), decipher.final()]).toString("utf8");
}

/**
 * Load an encrypted sparkId -> secret map from a given path.
 * Shared by SSH password and LLM API key stores.
 * @param {string} filePath
 * @param {string} label — for log messages
 * @returns {Map<string, string>}
 */
function loadEncryptedMap(filePath, label) {
  const map = new Map();
  if (!fs.existsSync(filePath)) return map;
  try {
    const key = resolveKey();
    const raw = fs.readFileSync(filePath, "utf8");
    const data = JSON.parse(raw);
    const entries = data?.secrets || {};
    if (typeof entries !== "object" || entries === null) return map;
    let failed = 0;
    for (const [id, blob] of Object.entries(entries)) {
      if (!id || typeof blob !== "string") continue;
      try {
        const val = decrypt(blob, key);
        if (val) map.set(id, val);
      } catch {
        failed += 1;
        console.error(`[secretsStore] Failed to decrypt ${label} for ${id} (wrong/missing key?)`);
      }
    }
    if (map.size > 0) console.log(`[secretsStore] Loaded ${map.size} ${label}(s) from encrypted store`);
    if (failed > 0) console.warn(`[secretsStore] ${failed} ${label}(s) could not be decrypted — re-enter via Edit Spark`);
  } catch (err) {
    console.error(`[secretsStore] Failed to load ${label}: ${err.message}`);
  }
  return map;
}

/**
 * Persist an encrypted sparkId -> secret map. Empty map removes the file.
 * @param {string} filePath
 * @param {Map<string, string>} secrets
 * @param {string} label — for log messages
 */
function saveEncryptedMap(filePath, secrets, label) {
  if (!secrets || secrets.size === 0) {
    if (fs.existsSync(filePath)) {
      try {
        fs.accessSync(filePath, fs.constants.W_OK);
        fs.unlinkSync(filePath);
      } catch (err) {
        throw new Error(`Failed to clear ${label} file (permission?): ${err.message}`);
      }
    }
    return;
  }
  const key = resolveKey();
  const entries = {};
  for (const [id, val] of secrets.entries()) {
    if (val) entries[id] = encrypt(val, key);
  }
  const payload = JSON.stringify({ version: 1, secrets: entries }, null, 2) + "\n";
  atomicWrite(filePath, payload, 0o644);
  console.log(`[secretsStore] Saved ${Object.keys(entries).length} ${label}(s)`);
}

/**
 * Load sparkId -> password map from disk.
 * @returns {Map<string, string>}
 */
export function loadSecrets() {
  return loadEncryptedMap(SPARKS_SECRETS_PATH, "SSH password");
}

/**
 * Persist sparkId -> password map (encrypted). Empty map removes the file.
 * @param {Map<string, string>} passwords
 */
export function saveSecrets(passwords) {
  saveEncryptedMap(SPARKS_SECRETS_PATH, passwords, "SSH password");
}

/**
 * Load sparkId -> LLM API key map from disk.
 * @returns {Map<string, string>}
 */
export function loadLlmKeys() {
  return loadEncryptedMap(SPARKS_LLM_KEYS_PATH, "LLM API key");
}

/**
 * Persist sparkId -> LLM API key map (encrypted). Empty map removes the file.
 * @param {Map<string, string>} keys
 */
export function saveLlmKeys(keys) {
  saveEncryptedMap(SPARKS_LLM_KEYS_PATH, keys, "LLM API key");
}
