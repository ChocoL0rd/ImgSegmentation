import os

main_path = "all"
main_list_dir = os.listdir(main_path)

paths = [
    "with_thing",
    "without_thing",
    "two_boots",
    "tapochek_like"
]

list_dirs = [os.listdir(path) for path in paths]

sum_list_dir = []
for i in list_dirs:
    sum_list_dir = sum_list_dir + i


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def sub(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3


print("not aggregated:")
for i in sorted(sub(main_list_dir, sum_list_dir)):
    print(i)

print("check deleted")
for i in sub(sum_list_dir, main_list_dir):
    print(i)

print("check intersections")
for i in range(len(paths)):
    for j in range(len(paths)):
        if i != j:
            print(i, j, intersection(list_dirs[i], list_dirs[j]))

print("check size")
print(len(sum_list_dir), len(main_list_dir))