
import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="SpamShield",
    page_icon="📩",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "models" / "spam_classifier.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Interface
st.title("📩 SpamShield")
st.subheader("SMS Spam Detection System")

st.write(
    "Enter an SMS message below to check whether it is "
    "spam or a genuine message."
)

st.divider()

message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            model = load_model()
            prediction = model.predict([message])[0]

            if str(prediction).lower() == "spam" or prediction == 1:
                st.error("🚨 SPAM MESSAGE")
                st.write(
                    "The model identifies this message as spam. "
                    "Be careful with suspicious links and requests."
                )

            elif str(prediction).lower() == "ham" or prediction == 0:
                st.success("✅ GENUINE MESSAGE (HAM)")
                st.write(
                    "The model identifies this message as genuine. "
                    "Still be cautious with unexpected messages."
                )

            else:
                st.info(f"Model prediction: {prediction}")

        except FileNotFoundError:
            st.error(
                "Model file not found. Make sure "
                "models/spam_classifier.pkl exists."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()

st.caption("SpamShield | Machine Learning SMS Classification")
