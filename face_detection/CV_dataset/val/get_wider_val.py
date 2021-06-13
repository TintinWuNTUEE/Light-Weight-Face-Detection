import os

imgs = []

for _, _, files in os.walk("./images"):
    for f in files:
        imgs.append(f)

with open('wider_val.txt', 'w') as fout:
    for img in imgs:
        fout.write(f"{img}\n")
