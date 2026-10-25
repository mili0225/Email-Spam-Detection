from collections import Counter

file_path = "data/SMSSpamCollection"

with open(file_path, "r", encoding="utf-8") as file:
    labels = [line.split("\t", 1)[0] for line in file]

counts = Counter(labels)

print("Total messages:", len(labels))
print("Ham messages:", counts["ham"])
print("Spam messages:", counts["spam"])