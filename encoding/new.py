# import sys

# data = b"hello"
# sys.stdout.buffer.write(data)
count = 0
for i in range(32, 127):
    print(f"{i}\t-> {chr(i)}")
    count += 1

print(f"Total characters: {count}")


def enc(c):
    return str(ord(c) - 30).ljust(2, "0")


print(ord("`"))
print(chr(126126))

x = "hello"
new_x = ""
for i in range(0, len(x), 3):
    a = enc(x[i])
    b = enc(x[i + 1]) if i + 1 < len(x) else "00"
    c = enc(x[i + 2]) if i + 2 < len(x) else "00"
    new_x += chr(int(a + b + c))

print(new_x)

old_x = ""
for c in new_x:
    val = str(ord(c)).rjust(6, "0")
    a = chr(int(val[0:2]) + 30) if val[0:2] != "00" else ""
    b = chr(int(val[2:4]) + 30) if val[2:4] != "00" else ""
    c = chr(int(val[4:6]) + 30) if val[4:6] != "00" else ""
    old_x += a + b + c

print(old_x)
