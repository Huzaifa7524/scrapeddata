import streamlit as st
from bs4 import BeautifulSoup
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Pretrained Model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Pipeline
summarizer = pipeline(
    task="summarization", model="sshleifer/distilbart-cnn-12-6"
)

# Function to get data from a URL
def summarization(text):
    ARTICLE= ' '.join(text)

    #tokenize text into sentences
    ARTICLE = ARTICLE.replace(',', '<eos>')
    ARTICLE = ARTICLE.replace('!', '<eos>')
    ARTICLE = ARTICLE.replace('?', '<eos>')
    ARTICLE = ARTICLE.replace('\n', '<eos>')
    ARTICLE = ARTICLE.replace("\t", '<eos>')
    ARTICLE = ARTICLE.replace(' ', '<eos>')

    sentences = ARTICLE.split('<eos>')

    #sentences into smaller chunks
    max_chunk = 600
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))


    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])

    len(chunks[0].split((' ')))


    #text summarization 
    result = summarizer(chunks, max_length=150, min_length=50, do_sample=False)

    ap = []
    for i in result:
        c = i['summary_text']
        ap.append(c)
    return ' '.join(ap)
def get_data_from_url(url):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
    }
    r = requests.get(url=url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['h1', 'p'])
    post_title = soup.title.text
    text = [result.text for result in results]
    summary = summarization(text)
    return summary, post_title

# Function for sentiment analysis
def analyze_sentiment(input_text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment_result = sentiment_analyzer(input_text)
    return sentiment_result[0]

# Function to count and display most frequent words
def display_most_frequent_words(text):
    words = text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)  # Change 10 to the desired number of words to display
    st.subheader("Most Frequent Words:")
    for word, count in most_common_words:
        st.write(f"{word}: {count} times")

# Function to create and display a word cloud
def display_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.subheader("Word Cloud:")
    st.pyplot(plt)  # Pass the figure explicitly

# Streamlit App
st.title("SummarySphere")

# User input
st.header("Enter URL or Text:")
url = st.text_input("Enter URL:")
text = st.text_area("Enter Text:")

# Summarize and analyze
if st.button("Summarize and Analyze"):
    if url:
        summary, post_title = get_data_from_url(url)
        st.subheader(f"Title: {post_title}")
        st.write("Summary:")
        st.write(summary)

        # Sentiment Analysis
        st.write("Sentiment Analysis:")
        sentiment = analyze_sentiment(summary)
        st.write(sentiment)

        # Display most frequent words
        display_most_frequent_words(summary)
        
        # Display word cloud
        display_word_cloud(summary)
    
    elif text:
        st.write("Summary:")
        summary = summarization(text.splitlines())
        st.write(summary)

        # Sentiment Analysis
        st.write("Sentiment Analysis:")
        sentiment = analyze_sentiment(text)
        st.write(sentiment)

        # Display most frequent words
        display_most_frequent_words(summary)
        
        # Display word cloud
        display_word_cloud(summary)

st.markdown("Created by [Huzaifa Tahir](https://www.linkedin.com/in/huzaifatahir7524/)")
