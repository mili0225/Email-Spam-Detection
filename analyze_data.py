import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

df["message_length"] = df["message"].str.len()

print("\nMessage length statistics:")
print(df.groupby("label")["message_length"].describe())

df["label"].value_counts().plot(kind="bar")
plt.title("Spam vs Ham Messages")
plt.xlabel("Label")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.savefig("reports/label_distribution.png")
plt.show()