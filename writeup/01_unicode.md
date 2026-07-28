# CS336 Assignment 1 — Written Responses: Unicode

## Problem (unicode1): Understanding Unicode (1 point)

### (a) What Unicode character does `chr(0)` return?

`chr(0)` returns the null character (NUL, code point U+0000), the string `'\x00'`.

### (b) How does this character's string representation (`__repr__()`) differ from its printed representation?

Its `__repr__()` shows the escaped form `'\x00'` (quotes plus an escape sequence), whereas printing it emits the raw NUL character itself, which is invisible and produces no visible output.

### (c) What happens when this character occurs in text?

The null character is silently embedded in the string (it still counts toward `len()` and appears as `\x00` in the repr), but when printed it renders as nothing visible, so `"this is a test" + chr(0) + "string"` prints looking like the two pieces joined with no visible character between them.

**Verification:**

```python
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
>>> len("this is a test" + chr(0) + "string")
21
```

## Problem (unicode2): Unicode Encodings (3 points)

### (a) Why prefer UTF-8 bytes over UTF-16 or UTF-32 for tokenizer training?

UTF-8 is far more compact for typical (largely ASCII/Latin) text — e.g. `"hello"` is 5 bytes in UTF-8 but 12 in UTF-16 and 24 in UTF-32 — and it contains no BOM or padding `\x00` bytes, whereas UTF-16/32 pad every ASCII character with null bytes, yielding sequences that are 2–4x longer and full of uninformative zero bytes that waste vocabulary merges and model capacity (UTF-8 is also the dominant encoding of web text, so it matches real training data).

```python
>>> for s in ["hello", "こんにちは", "hello! こんにちは!"]:
...     print(s, len(s.encode("utf-8")), len(s.encode("utf-16")), len(s.encode("utf-32")))
hello 5 12 24
こんにちは 15 12 24
hello! こんにちは! 23 28 56
>>> list("hello".encode("utf-16"))
[255, 254, 104, 0, 101, 0, 108, 0, 108, 0, 111, 0]   # BOM + null padding bytes
```

### (b) Why is `decode_utf8_bytes_to_str_wrong` incorrect? Example input.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

The function decodes each byte individually, but UTF-8 encodes non-ASCII characters as multi-byte sequences whose individual bytes are not valid UTF-8 on their own; for example the input `"牛".encode("utf-8")` = `b'\xe7\x89\x9b'` fails, because decoding the single byte `b'\xe7'` raises `UnicodeDecodeError` instead of being combined with its two continuation bytes.

**Verification:**

```python
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
>>> decode_utf8_bytes_to_str_wrong("牛".encode("utf-8"))   # b'\xe7\x89\x9b'
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 0: unexpected end of data
```

### (c) A two-byte sequence that does not decode to any Unicode character(s)

`b'\xc0\xaf'` does not decode under UTF-8, because `0xC0` is never a legal UTF-8 leading byte (it would be an "overlong" encoding of an ASCII character, which the standard forbids).

**Verification:**

```python
>>> b'\xc0\xaf'.decode("utf-8")
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 0: invalid start byte
```

(Another example: `b'\xc3\x28'` fails because `0x28` is not a valid continuation byte — continuation bytes must be in `0x80`–`0xBF`.)
