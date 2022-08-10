a = 9.072 * (10 ** (15))
# l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
l = [a]
end = [1, 2, 4]

# for x in l:
#     mod = x % 2
#     exp = mod - 1
#     coef = 3 * (6 ** exp)
#     print(f"{coef}x + {mod}")

# for x in l:
#     print((3 * (6 ** (x % 2 - 1)) * x) + (x % 2))

# for _ in range(2):
#     print()
# for x in l:
#     while x not in end:
#         print(x)
#         x = (3 * (6 ** (x % 2 - 1)) * x) + (x % 2)
#     for _ in range(2):
#         print()
def simple_collatz(n):
    while n != 1:
        if n % 2 == 0:
            n = n / 2
            return int(n)
        else:
            n = 3 * n + 1
            return int(n)


def collatz(x):
    return int((3 * (6 ** (x % 2 - 1)) * x) + (x % 2))


def short_collatz(x):
    return int((0.5 * (3 ** (x % 2)) * x) + (x % 2))


# base = 3
# print(base)
# x = base
# for _ in range(2):
#     print(x := collatz(x))

# print()
# print(base)
# print(short_collatz(3))
# for x in range(2 ** 68, 2 ** 69):
def get_tabs(string: str) -> int:
    rtn = ""
    for _ in range(int(len(string) / 8)):
        rtn += "\t"
    return rtn


# for x in range(2**2 + 1, 2**8):
#     base_out = f"BASE: {x}"
#     print(base_out, end=f"    {get_tabs(base_out)}")
#     found = []
#     while x not in end:
#         if x not in found:
#             found.append(x)
#         else:
#             print(f"LOOP FOUND: {x}")
#             break
#         x = collatz(x)
#     print(f"Cycle Length: {len(found)}")


# We are trying to solve where f(f(...(f(x)))) / 2^y = x, where f(x) = 3x + 1


def g(y):
    # if y < 0:
    #     return 0
    # else:
    #     return (3 * g(y - 1)) + (2**y)

    counter = 0
    x = 0
    while counter <= y:
        x = (3 * x) + (2**counter)
        counter += 1
    return x


def simple_loop_detector(x=0, y=0):
    """
    Simple test for loop in colatz.
    Only solves for instances where 3x + 1 == x times some power of 2.
    For Example:
        for x = 0, y = 2, the result is 1.
        eqn: 3(1)+1 = 4 = (1)*2^2
        path: 1 -> 4 -> 2 -> 1

    Solve for x: 3x+1 = x*(2^y) ===> x = 1 / ((2^y) -3)
    Solve for x: 3((3x+1)/2)+1 = x*(2^y) ===> x = 5 / ((2^(y+1)) - 9)
    Solve for x: 3((3((3x+1)/2)+1)/2)+1 = x*(2^y) ===> x = 19 / ((2^(y+2)) - 27)
    """
    # assert x >= 0
    a = g(x)
    c = 3 ** (x + 1)
    b = 2 ** (y + x)
    out_y = y
    found = False
    while (b - c) <= a:
        if b > c:
            num = g(x) / ((2 ** (y + x)) - (3 ** (x + 1)))
            if num == int(num) or True:
                print(f"X: {x} Y: {y}")
                print(f"{g(x)} / ({b} - {c}) = {num}")
                out_y = y
                found = True
        y += 1
        b = 2 ** (y + x)
    return out_y, found


def loop_detector(x=0, y=0):
    """
    Solve for x: 3x+1 = x*(2^y) ===> x = 1 / ((2^y)-3)
    Solve for x: 3((3x+1)/(2^a))+1 = x*2^y ===> x = ((2^a)+3) / ((2^(a+y))-9)
    Solve for x: 3((3((3x+1)/(2^b))+1)/(2^a))+1 = x*2^y ===> x = ((2^(a+b))+(3*2^(b))+9) / ((2^(a+b+y))-27)
    Solve for x: 3((3((3((3x+1)/(2^c))+1)/(2^b))+1)/(2^a))+1 = x*2^y ===> x = ((2^(a+b+c))+(3*2^(b+c))+(9*2^(c))+27) / ((2^(a+b+c+y))-81)
    """
    pass


# print(g(1))
# x = [i for i in range(1, 101)]
# y = [g(i) for i in x]
# print(y)

# print(list(zip(x, y)))

print("Starting Search:")
y = 0
for x in range(100):
    y, found = simple_loop_detector(x, y)
    if found:
        print()

print("Search Complete")
