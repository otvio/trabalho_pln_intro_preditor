from sklearn.metrics import cohen_kappa_score

correct_labels = [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]

annotator_labels = [
    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],  # murilo
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # raul
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],  # otavio
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1]   # vitor
]
avg_value = 0.0
for i in range(4):
    for j in range(4):
        if (i >= j):
            continue
        value = cohen_kappa_score(annotator_labels[i], annotator_labels[j])
        avg_value += value
        print("cohen kappa for annotators", i + 1, "and", j + 1, ":", "{0:.3f}".format(value))
print()
print("avg cohen kappa:", "{0:.3f}".format(avg_value / 6.0))
print()
for i in range(4):
    sumi = 0
    for j in range(30):
        if annotator_labels[i][j] == correct_labels[j]:
            sumi += 1
    print("annotator", i + 1, "correctness:", "{0:.3f}".format(sumi / 30.0))

agreed = 0
for j in range(30):
    sumi = sum([annotator_labels[i][j] for i in range(4)])
    agreed += 1 if sumi == 0 or sumi == 4 else 0
print("\ntotal agreement in:", "{0:.3f}".format(agreed / 30.0), "of the samples")
