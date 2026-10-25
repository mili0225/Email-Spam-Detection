import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("models/spam_classifier.pkl")
predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Email Spam Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.show()

print("Confusion matrix saved to reports/confusion_matrix.png")