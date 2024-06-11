path = "data/datalists/hmdb51_split1_train.txt"
l = [0]*51
names = [""]*51
with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(" ")
        name = arr[1]
        idx = int(arr[2])
        names[idx] = name
        l[idx] += 1

print(l)
print(names)
path = "data/datalists/ucf101_split1_train.txt"
l = [0] * 101
names = [""] * 101
with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
        arr = line.split(" ")
        name = arr[1]
        idx = int(arr[2])
        names[idx] = name
        l[idx] += 1

print(l)
print(names)
