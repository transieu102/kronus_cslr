import pandas as pd

csv1 = pd.read_csv("predictions/test_20250624_062009/test.csv", delimiter=",")
csv2 = pd.read_csv("predictions/test_20250624_022055/test.csv", delimiter=",")

count_diff = 0
for i in range(len(csv1)):
    row1 = csv1.iloc[i]
    row2 = csv2.iloc[i]
    if row1["gloss"] != row2["gloss"]:
        print(row1["id"], row2["id"])
        print(row1["gloss"], "||",row2["gloss"])
        # input()
        count_diff += 1

print(count_diff/len(csv1) * 100)


