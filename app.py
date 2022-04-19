import streamlit as st
from PIL import Image

st.title('Hate-Speech-Detection')


st.markdown("**Overview**: This project uses the tweet data to fine tune the distilbert in order to classify if the tweet has offensive content, hate speech or just neutral content.")

st.markdown("**Why important**: Offensive or hate speech are the main causes of the internet violence. By classifying them as soon as they were posted online and blocking them could protect the mental health for people especially underaged kids.")

st.markdown("**Approach**:")

st.markdown("1.Dataset: This project uses the dataset called hate_speech_offensive, which contained over 20000 tweets all labeled manually into three categories, offensive, hate or newtral.")

data_distribution1 = Image.open('images/1.png')
st.image(data_distribution1)

st.markdown("However, as the figure shows, the data is heavily imbalanced, and an under-sampling method was applied. The result is as followings:")
data_distribution2 = Image.open('images/2.png')

st.image(data_distribution2)

st.markdown("2.Model selection: The distilbert model was fine tuned in order to achieve the goal. The very advantage for this model is that it is small and efficiency.")

st.markdown("3.Result: The model was trained for 2 Epochs and reached 0.83 accuracy on test set and 0.81 on validation set, which is a descent result.")

st.markdown("**Critical Analysis**: The importance of hardware! If only I have a better GPU and large RAM space, the model can be tuned even better. Also this model can be fine tuned on other tweet models as well.")

st.markdown("**Resource Links**")
st.markdown("Data: https://huggingface.co/datasets/hate_speech_offensive")
st.markdown("Model: https://huggingface.co/stevenlx96/distilbert-base-uncased-finetuned-hated")
st.markdown("Distilbert. https://huggingface.co/docs/transformers/model_doc/distilbert")
