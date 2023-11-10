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

data = "Printing the base64 encoding of this string"

# URL and Filename Safe Base64 Encoding
encodedBytes = base64.urlsafe_b64encode(data.encode("utf-8"))
encodedStr = str(encodedBytes, "utf-8")

# print(type(encodedStr))
# print(encodedStr)


# Url Safe Base64 Decoding
decodedBytes = base64.urlsafe_b64decode(encodedStr)
decodedStr = str(decodedBytes, "utf-8")

# print(type(decodedStr))  # hello world123!?$
# print(url_safe_dict)


def split_keep_delimiter(msg, delim):

    msg_list = msg.split(delim)
    out = []
    for item in msg_list:
        if item != "":
            out.append(item)
        out.append(delim)
    return out[:-1]


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


def printDivisors(n):
    list = []

    # List to store half of the divisors
    for i in range(1, int(math.sqrt(n) + 1)):

        if n % i == 0:

            # Check if divisors are equal
            if n / i == i:
                print(i, end=" ")
            else:
                # Otherwise print both
                print(i, end=" ")
                list.append(int(n / i))

    # The list will be printed in reverse
    for i in list[::-1]:
        print(i, end=" ")


def get_max_multiple(number):
    # vals = [Decimal(number) % Decimal(prime) for prime in primes]
    multiples = []
    length = len(str(number))
    counter = 1
    check = True
    while check:
        counter += 1
        val = Decimal(number / Decimal(counter))
        if val < counter:
            check = False
            break
        if len(str(val)) < length:
            multiples.append(val)

    # for prime in primes:
    #     mod = Decimal(number) % Decimal(prime)
    #     if mod == 0:
    #         multiples.append(prime)
    #     else:
    #         multiples.append(0)

    return max(multiples)


def minimize(encoded_str):
    encoded_num = encoded_to_numeric(encoded_str)
    precision = len(str(encoded_num)) * 10
    # print(f"precision: {precision}")
    decimal.getcontext().prec = precision
    remainder = encoded_num % 8
    # encoded_num = float(encoded_num - remainder)
    encoded_num = Decimal(encoded_num)
    count = 0
    conversions = []
    multiples = []
    last_value = Decimal("inf")
    while last_value > 63:
        count += 1

        last_value = Decimal(Decimal(encoded_num) / (Decimal(8) ** Decimal(count)))
        # print(last_value)
        conversions.append(last_value)
        multiples.append(Decimal(2) ** Decimal(count))

    lengths = [len(str(item)) for item in conversions]
    conv_data = {
        "min": {
            "length": min(lengths),
            "index": lengths.index(min(lengths)),
            "value": conversions[lengths.index(min(lengths))],
        },
        "max": {
            "length": max(lengths),
            "index": lengths.index(max(lengths)),
            "value": conversions[lengths.index(max(lengths))],
        },
    }
    print(conv_data)
    min_index = conv_data["min"]["index"]
    # print(f"count: {count}")
    # print(f"remainder: {remainder}")
    return conv_data["min"]["value"]


decimal.getcontext().prec = 300
separated_str = break_on_padding(encodedStr)
# print(separated_str)
# print("Numeric Value:")
# print(encoded_to_numeric(separated_str[0]))
# print(minimize(separated_str[0]))
# val = Decimal(
#     207941273917412738283229633378693328542054132137273813472563746255014725341522663751871352283837462548
# )
val = 207941273917412738283229633378693328542054132137273813472563746255014725341522663751871352283837462548
# print(get_max_multiple(val))
# test = get_max_multiple(val)
# test = printDivisors(val)

# print(test)
# print(val / Decimal(test))
vala = minimize(separated_str[0])
print(vala)
# print((1.4854640888423534 * 56 * 64) + 4)
print("here")
# multi = Decimal(64 ** 47)
multi = Decimal(1526639)
a = Decimal(val / multi)
# a = minimize(separated_str[0])
b = Decimal(a * multi)
# print(format(a, ".60g"))
print(a)
print(val)
print(b)
# print(format(multi, ".100g"))
# print(str(a * multi))
print(Decimal(val) - b)
print("------------------------------------")


def get_nearest_factor(number):
    current = Decimal(0)
    counter = Decimal(0)
    while current < number:
        counter += Decimal(1)
        current = Decimal(Decimal(2) ** Decimal(counter))
    return counter


def decode_num(encoded_num, remainder, itr):
    return str(int(encoded_num * (64 ** itr)) + remainder)


print(decode_num(1.4854640888423534, 4, 56))

# print([2 ** i for i in range(100)])

# val = 207941273917412724749993624425001520422307745430399003871714190007479777217144083000015313359513583620
print("Nearest Factor:")
print(get_nearest_factor(val))
print(Decimal(val) - Decimal(Decimal(2) ** Decimal(336)))

# print(val / (2 ** (get_nearest_factor(val) - 1)))
# for i in range(get_nearest_factor(val)):
#     print(format(val / (2 ** i), ".60g"))


def printDivisors(n):
    list = []

    # List to store half of the divisors
    for i in range(int(math.sqrt(n)), 0, -1):

        if n % i == 0:

            # Check if divisors are equal
            if n / i == i:
                print(i)
            else:
                # Otherwise print both
                print(i)
                list.append(int(n / i))
            break

    # The list will be printed in reverse
    for i in list[::-1]:
        print(i)


print("\n")
printDivisors(
    207941273917412724749993624425001520422307745430399003871714190007479777217144083000015313359513583620
)
# print(oct(8))
