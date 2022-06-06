import pandas as pd
from apyori import apriori

# Operasi file membaca dataset
data = pd.read_csv('resto.csv', header=None)

records = []
for i in range(0, 21):
    records.append([str(data.values[i, j]) for j in range(0, 2)])

a = float(input('masukkan minimum support:'))
b = float(input('masukkan minimum confidence:'))

rules = apriori(records, min_support=a, min_confidence=b, min_lift=2, min_length=2)
result = list(rules)
panjang = str(len(result))

for item in result:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: "+items[0]+" -> "+items[1])

    print("Support: " + str(item[1]))

    print("Confidence:" + str(item[2][0][2]))
    print("===============")

print("Banyanknya Strong associaton rules:", len(result))
