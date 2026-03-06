import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
import pyLDAvis.gensim_models as gensim_lda
import pyLDAvis
import streamlit.components.v1 as components

# Load data
def load_data():
    file_path = "updated_data.xlsx"  # Ensure this file is in your working directory
    df = pd.read_excel(file_path)
    df["DATA"] = df["DATA"].astype(str).fillna("")
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def perform_lda(df, num_topics=5):
    df["Processed_DATA"] = df["DATA"].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(df["Processed_DATA"])
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)
    
    def get_topics(model, feature_names, num_words=10):
        topics = {}
        for idx, topic in enumerate(model.components_):
            topics[f"Topic {idx+1}"] = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        return topics
    
    return get_topics(lda_model, vectorizer.get_feature_names_out())

def plot_wordcloud(top_words_per_topic):
    fig, axes = plt.subplots(1, len(top_words_per_topic), figsize=(20, 5))
    for i, topic in enumerate(top_words_per_topic.keys()):
        wordcloud = WordCloud(width=400, height=400, background_color="white").generate(" ".join(top_words_per_topic[topic]))
        axes[i].imshow(wordcloud, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(topic)
    st.pyplot(fig)


# Function to plot Inter-Topic Distance Map
def plot_inter_topic_distance(lda_model, corpus, id2word):
    st.write("### 📍 Inter-Topic Distance Map")
    lda_vis = gensim_lda.prepare(lda_model, corpus, id2word, mds="tsne")  # Use t-SNE for better visualization
    pyLDAvis.save_html(lda_vis, "lda_visualization.html")

    # Render in Streamlit
    with open("lda_visualization.html", "r", encoding="utf-8") as f:
        components.html(f.read(), height=800, scrolling=True)

def plot_accident_trends(df):
    df["Year"] = df["DATE"].dt.year
    df["Month"] = df["DATE"].dt.month
    
    accidents_per_year = df["Year"].value_counts().sort_index()
    accidents_per_month = df["Month"].value_counts().sort_index()
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(accidents_per_year.index, accidents_per_year.values, marker="o", linestyle="-")
    ax[0].set_title("Accident Trends Over the Years")
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Number of Accidents")
    ax[0].grid(True)
    
    ax[1].bar(accidents_per_month.index, accidents_per_month.values, color="skyblue")
    ax[1].set_title("Accident Distribution by Month")
    ax[1].set_xlabel("Month")
    ax[1].set_ylabel("Number of Accidents")
    ax[1].set_xticks(range(1, 13))
    ax[1].grid(axis="y")
    
    st.pyplot(fig)

def plot_accident_hotspots(df):
    # Drop rows where latitude or longitude is missing
    df = df.dropna(subset=["Latitude", "Longitude"])
    
    # Create a base map centered around the accident locations
    if not df.empty:
        center_location = [df["Latitude"].mean(), df["Longitude"].mean()]
        m = folium.Map(location=center_location, zoom_start=6)
        
        # Add accident markers
        for _, row in df.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"Location: {row['LOCATION']}<br>Details: {row['DATA']}",
                tooltip=row["LOCATION"]
            ).add_to(m)

        # Render the accident markers map
        st.subheader("🚨 Accident Hotspots Map")
        folium_static(m)

        # Heatmap Layer
        heatmap_layer = folium.Map(location=center_location, zoom_start=6)
        HeatMap(data=df[['Latitude', 'Longitude']].values, radius=15, blur=10).add_to(heatmap_layer)

        # Render the heatmap
        st.subheader("🔥 Accident Density Heatmap")
        folium_static(heatmap_layer)

    else:
        st.warning("No valid latitude and longitude data available for mapping.")


def main():
    st.title("Railway Accident Analysis Dashboard")
    df = load_data()
    
    # Sidebar navigation
    menu = ["Home", "Accident Trends", "Topic Modeling", "Accident Hotspots"]
    choice = st.sidebar.selectbox("Select Analysis Type", menu)
    
    if choice == "Home":
        st.write("### Explore railway accident data and insights.")
        st.dataframe(df.head())
    
    elif choice == "Accident Trends":
        st.write("### Yearly and Monthly Trends in Railway Accidents")
        plot_accident_trends(df)
    
    elif choice == "Topic Modeling":
        st.write("### Topic Modeling (LDA) of Accident Reports")
        num_topics = st.slider("Select number of topics", min_value=2, max_value=10, value=5)
        topics = perform_lda(df, num_topics)
        st.write(topics)
        plot_wordcloud(topics)
    
    elif choice == "Accident Hotspots":
        st.write("### Accident Hotspots Visualization")
        plot_accident_hotspots(df)
    
def dynamic_risk_index(df):
    df["Year"] = df["DATE"].dt.year
    
    # 1️⃣ Frequency Score
    location_counts = df.groupby("LOCATION").size()
    freq_score = location_counts / location_counts.max()
    
    # 2️⃣ Sentiment Score
    from textblob import TextBlob
    df["Sentiment"] = df["DATA"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    sentiment_score = df.groupby("LOCATION")["Sentiment"].mean()
    sentiment_score = 1 - sentiment_score  # Negative sentiment = high risk
    
    # 3️⃣ Combine Scores
    risk = (0.6 * freq_score) + (0.4 * sentiment_score)
    risk = (risk * 100).round(2)
    
    risk_df = pd.DataFrame({
        "Location": risk.index,
        "Risk Score": risk.values
    }).sort_values(by="Risk Score", ascending=False)
    
    st.subheader("🚨 Dynamic Railway Risk Index")
    st.dataframe(risk_df.head(10))
    
    st.bar_chart(risk_df.set_index("Location").head(10))
    
if __name__ == "__main__":
    main()
