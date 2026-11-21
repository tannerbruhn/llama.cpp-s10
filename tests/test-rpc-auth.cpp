#include "auth.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool check(const uint8_t digest[32], const char * expected) {
    char hex[65];
    for (int i = 0; i < 32; ++i) { snprintf(hex+2*i, 3, "%02x", digest[i]); }
    if (strcmp(hex, expected) == 0) { return true; }
    fprintf(stderr, "digest mismatch: %s != %s\n", hex, expected);
    return false;
}

int main() {
    uint8_t digest[32];
    bool ok = true;
    const char * messages[] = {"", "abc", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
    const char * hashes[] = {
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"};
    for (int i = 0; i < 3; ++i) {
        rpc_auth_sha256(messages[i], strlen(messages[i]), digest);
        ok &= check(digest, hashes[i]);
    }
    std::vector<uint8_t> key(20, 0x0b);
    rpc_auth_hmac(key.data(), key.size(), "Hi There", 8, digest);
    ok &= check(digest, "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    const char * text = "what do ya want for nothing?";
    rpc_auth_hmac(reinterpret_cast<const uint8_t *>("Jefe"), 4, text, strlen(text), digest);
    ok &= check(digest, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    key.assign(131, 0xaa);
    text = "Test Using Larger Than Block-Size Key - Hash Key First";
    rpc_auth_hmac(key.data(), key.size(), text, strlen(text), digest);
    ok &= check(digest, "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    return ok ? 0 : 1;
}
