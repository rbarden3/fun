import base64
from base64_dict import url_safe_dict, primes
import decimal
from decimal import Decimal
import math

x = Decimal("0.1")
y = Decimal("0.1")
z = Decimal("0.1")

s = x + y + z

# print(s)

data = "Hello"

# URL and Filename Safe Base64 Encoding
encodedBytes = base64.urlsafe_b64encode(data.encode("utf-8"))
encodedStr = str(encodedBytes, "utf-8")

print(type(encodedStr))
print(encodedStr)


# Url Safe Base64 Decoding
decodedBytes = base64.urlsafe_b64decode(encodedStr)
decodedStr = str(decodedBytes, "utf-8")

print(decodedStr)  # hello world123!?$
# print(url_safe_dict)
def break_on_padding(encoded_str):
    delim = "="

    msg_list = encoded_str.split(delim)
    out = []
    for item in msg_list:
        if item != "":
            out.append(item)
        out.append(delim)
    return out[:-1]


def encoded_to_numeric(encoded_str):
    encoded_num = [f"{url_safe_dict[char]}" for char in encoded_str]
    # print(encoded_num)
    encoded_num = "".join(encoded_num)
    encoded_num = Decimal(encoded_num)
    return encoded_num


lst = break_on_padding(encodedStr)
# print(encoded_to_numeric(lst[0]))

test = 186214427660
print(test % 65521)
print((test - 10963) / 65521)
