# import sys

# data = b"hello"
# sys.stdout.buffer.write(data)

# for i in range(150, 200):
#     print(f"{i}\t->{chr(i)}")


def enc(c):
    return str(ord(c)).ljust(3, "0")


print(ord("`"))
print(chr(126126))

x = "hello"
new_x = ""
for i in range(0, len(x), 2):
    a = enc(x[i])
    b = enc(x[i + 1]) if i + 1 < len(x) else "000"
    new_x += chr(int(a + b))

print(new_x)
