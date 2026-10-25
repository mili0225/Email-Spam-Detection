import joblib

model = joblib.load("models/spam_classifier.pkl")

while True:
    message = input("\nEnter a message (or type 'exit' to quit): ")

    if message.lower() == "exit":
        break

    prediction = model.predict([message])[0]

    if prediction == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM (Not Spam)")