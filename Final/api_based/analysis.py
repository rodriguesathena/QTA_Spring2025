#!/usr/bin/env python

#SET UP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Topic Modelling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
#Sentiment Analysis
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

#CSV
path = r"C:\Users\athen\Documents\GitHub\QTA_Spring2025\Final\api_based\song_clean.csv"
df = pd.read_csv(path)
df = df.drop(columns=['lyrics'])
df.head

#GENRES
# Check genres to see if it should be main focus
#Unknown Genres
unknown_count = (df['genre'].str.lower() == 'unknown').sum()
print(f"Unknown genre count: {unknown_count}")

#Drop unknowns and create strings
genre_series = df['genre'].dropna().str.lower()
genre_series = genre_series[genre_series != 'unknown']

#Deal with multiple genres
genre_split = genre_series.str.split(', ')
genre_exploded = genre_split.explode()

#Count genres
genre_counts = genre_exploded.value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']

#Display top genres
print("\nTop 15 genres:")
print(genre_counts.head(15))
df.head

#Chose to not as unknowns are a good amount; will maybe look into it for further analysis
# Create subset with known genres
df_with_genre = df[~df['genre'].isin(['Unknown', '', None])].copy()


#LANGUAGES
# EDA of Spanish Songs
spanish_keywords = [
    'amor', 'corazón', 'beso', 'bailar', 'cielo', 'noche', 'día',
    'vida', 'latina', 'reggaeton', 'fiesta', 'cantar', 'mamí', 'mi amor',
    'contigo', 'latino', 'ven', 'quiero', 'baila', 'dame', 'morena', 'rico']

pattern = re.compile(r'\b(?:' + '|'.join(spanish_keywords) + r')\b', flags=re.IGNORECASE)

# Apply the regex to count Spanish keywords
def is_spanish(lyrics, threshold=2):
    if pd.isna(lyrics):
        return False
    matches = pattern.findall(lyrics.lower())
    return len(matches) >= threshold

# Apply the function
df['is_spanish'] = df['lyrics_clean'].apply(is_spanish)

# Show updated counts
print("Spanish:", df['is_spanish'].sum())
print("Non-Spanish:", (~df['is_spanish']).sum())
english_df = df[df['is_spanish'] == False].copy()


# Topic Modelling

# PRE PROCESSING
#setting stopwords and lemmitization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
custom_stopwords = set([ 
    'yeah', 'yea', 'yo', 'uh', 'uhh', 'uhm', 'uhhuh', 'huh', 'ha', 
    'haha', 'hee', 'nah', 'na', 'la', 'oo', 'ooh', 'ah', 'aah', 'whoa', 'woo', 'woah',
    'nigga', 'shit', 'fuck', 'fucking', 'bitch', 'ass', 'hoe', 'damn', 'nigger', 'niggas', 'niggers'
    'im', 'ive', 'ill', 'youre', 'youve', 'youll', 'dont', 'cant', 'wont', 
    'didnt', 'couldnt', 'shouldnt', 'aint', 'lets','baby', 'girl', 'boy', 'man', 
    'woman', 'thing', 'love', 'heart', 'mind', 'song', 'dance', 'feel', 'make', 'take', 
    'got', 'gonna', 'one', 'see', 'say', 'let', 'come', 'back',
    'hey', 'right', 'time', 'cause', 'wanna', 'gon', 'thats', 'new',
    'make', 'wont', 'need', 'want', 'never', 'little', 'away', 'thing',
    'oh', 'yeah', 'feel', 'life', 'way', 'night', 'day'
    'get', 'go', 'like', 'know', 'get','nana', 'lala', 'yeah', 'woo', 'dada',
    'woo', 'woohoo', 'mmm', 'mmmmm', 'la', 'lala', 'lalala', 
    'blah', 'blahblah', 'uh', 'ah', 'ooh', 'whoa', 'hey', 'ho'])
all_stopwords = stop_words.union(custom_stopwords)


def clean_lyrics(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)

    #Remove overused filler sounds (ooh, ayy, laaa, etc.)
    text = re.sub(r'\b(?:o+)+h*\b', '', text)  # ooh, ohhh
    text = re.sub(r'\b(?:a+)+y+\b', '', text)  # ayy, ayyy
    text = re.sub(r'\b(?:la+)+\b', '', text)   # laaa, lala

    #Normalize repetitive expressions
    text = re.sub(r'(na\s*){2,}', 'nana ', text)
    text = re.sub(r'(yeah\s*){2,}', 'yeah ', text)
    text = re.sub(r'(woo\s*){2,}', 'woo ', text)
    text = re.sub(r'(la\s*){2,}', 'lala ', text)
    text = re.sub(r'(da\s*){2,}', 'dada ', text)
    text = re.sub(r'\b(nana|yeah|woo|lala|dada)\b(\s+\1)+', r'\1', text)

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)

    #Tokenize
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in all_stopwords
        and len(word) > 2]

    return " ".join(words)


#Apply to English songs
english_df['lyrics_clean_final'] = english_df['lyrics_clean'].apply(clean_lyrics)


# VECTORIZING

vectorizer = CountVectorizer(
    stop_words=list(all_stopwords),
    ngram_range=(1, 1),       
    min_df=10,                # common words
    max_df=0.9)                # rare words

dtm = vectorizer.fit_transform(english_df['lyrics_clean_final'])
tf_feature_names = vectorizer.get_feature_names_out()




# LDA
lda = LatentDirichletAllocation(n_components=12, random_state=37)
lda.fit(dtm)


# TOPIC MODELLING TABLES
# TABLE 1
# Themes
topic_themes = {
    0: "Party & Club Vibes",
    1: "Love, Trust & Emotional Tension",
    2: "Reflective / Searching for Meaning",
    3: "Romantic Encounters & Anticipation",
    4: "Power, Ego & Provocative Energy",
    5: "Heartache & Vulnerability",
    6: "Missing Someone / Longing",
    7: "Energy & Escapism",
    8: "Dreams & Idealism",
    9: "Sensual & Intimate Moments",
    10: "Movement & Confidence",
    11: "Streetwise Attitude & Bragging"}

topic_values = lda.transform(dtm)
english_df['dominant_topic'] = topic_values.argmax(axis=1)

#loop to print and display table
for topic_num, topic in enumerate(lda.components_):
    print(f"\nTopic {topic_num + 1} — {topic_themes.get(topic_num, 'No Theme')}:")
    top_words = [tf_feature_names[i] for i in topic.argsort()[-10:][::-1]]
    print("Top words:", " | ".join(top_words))

    top_songs_df = english_df[english_df['dominant_topic'] == topic_num]

    if not top_songs_df.empty:
        sample_size = min(3, len(top_songs_df))
        top_songs = top_songs_df.sample(sample_size, random_state=1)

        for _, row in top_songs.iterrows():
            print(f" - {row['title']} by {row['artist']}")
    else:
        print(" - No songs found for this topic.")


# Sentiment Analysis

# RUNNING Sentiment Analysis
sia = SentimentIntensityAnalyzer()
english_df['sentiment_scores'] = english_df['lyrics_clean_final'].apply(lambda x: sia.polarity_scores(str(x)))

#Dictionary Organization
sentiment_df = english_df['sentiment_scores'].apply(pd.Series)
english_df = pd.concat([english_df, sentiment_df], axis=1)

#Label
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

english_df['sentiment_label'] = english_df['compound'].apply(classify_sentiment)


#VISUALISATIONS
# Distribution of Sentiment Labels
sns.set_theme(style="whitegrid")
custom_palette = ['#7FB3D5', '#D5D8DC', '#E59866']  # Positive, Negative, Neutral

#Plot
plt.figure(figsize=(5.5, 4))
sns.countplot(data=english_df, x='sentiment_label', palette=custom_palette)
plt.title('Sentiment Classification of Lyrics', fontsize=13, fontweight='bold')
plt.xlabel('Sentiment', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine()
plt.tight_layout()

# Sentiment by Year
avg_sentiment_by_year = english_df.groupby('year')['compound'].mean().reset_index()

#Plot
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=avg_sentiment_by_year,
    x='year',
    y='compound',
    marker='o',
    linewidth=2,
    color='#5DADE2')
plt.title('Average Sentiment Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Compound Sentiment', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



#SENTIMENT VS. TOPIC

# FIGURE 2
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5.5))

#Define the crosstab
topic_sentiment_ct = pd.crosstab(
    english_df['dominant_topic'],
    english_df['sentiment_label'],
    normalize='index')
topic_sentiment_ct.index = topic_sentiment_ct.index.map(topic_themes)

#Plot
sentiment_colors = {
    'Positive': '#9BBFF2',  
    'Neutral': '#E39A78',  
    'Negative': '#D9D9D9'}
topic_sentiment_ct[["Negative", "Neutral", "Positive"]].plot(
    kind='bar',
    stacked=True,
    figsize=(11, 5.5),
    color=[sentiment_colors['Negative'], sentiment_colors['Neutral'], sentiment_colors['Positive']])
plt.title("Sentiment Distribution Across Topics", fontsize=13, fontweight='bold')
plt.xlabel("Topic", fontsize=11)
plt.ylabel("Proportion", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.legend(title="Sentiment", fontsize=9, title_fontsize=10)
plt.tight_layout()

# Topics dominating Sentiment Group
sentiment_topic_ct = pd.crosstab(english_df['sentiment_label'], english_df['dominant_topic'], normalize='index')
sentiment_topic_ct.columns = sentiment_topic_ct.columns.map(topic_themes)

# Plot
sentiment_topic_ct.T.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
plt.title("Topic Distribution Within Each Sentiment")
plt.xlabel("Topic")
plt.ylabel("Proportion Within Sentiment Group")
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


#SENTIMENT-SPECIFIC TOPIC MODELLING

# RUNNING Sentiment-Specific Topic Modelling

# Define themes for positive and negative sentiment
positive_themes = {
    0: "Swagger & Hype Energy",
    1: "Optimism & Light",
    2: "Love & Waiting",
    3: "Fun, Fame & Social Buzz",
    4: "Sensuality & Confidence",
    5: "Friendship, Reassurance & Beauty"
}

negative_themes = {
    0: "Stress, Confusion & Burnout",
    1: "Hustle & Confrontation",
    2: "Street Talk & Bravado",
    3: "Club Vibes & Seduction",
    4: "Heartache & Disconnection",
    5: "Nostalgia & Longing"
}

positive_table = []
negative_table = []

# Loop through data and run LDA for both Positive and Negative Sentiment Topics
for label in ['Positive', 'Negative']:
    theme_dict = positive_themes if label == 'Positive' else negative_themes
    subset = english_df[english_df['sentiment_label'] == label]
    
    # VECTORIZING
    vectorizer = CountVectorizer(max_df=0.9, min_df=10, stop_words='english')
    dtm = vectorizer.fit_transform(subset['lyrics_clean_final'])
    tf_feature_names = vectorizer.get_feature_names_out()
    
    # LDA
    lda = LatentDirichletAllocation(n_components=6, random_state=42)
    topic_values = lda.fit_transform(dtm)
    
    # Add dominant topic for each song
    subset = subset.copy()
    subset['dominant_topic'] = topic_values.argmax(axis=1)
    
    # Count topics for each sentiment
    topic_counts = subset['dominant_topic'].value_counts().sort_index()
    total = topic_counts.sum()
    
    # LABELLING
    for topic_num, topic in enumerate(lda.components_):
        top_words = [tf_feature_names[i] for i in topic.argsort()[-10:][::-1]]
        percent = round((topic_counts.get(topic_num, 0) / total) * 100, 1)
        
        # Prepare row for the output table
        row = {
            'Topic #': f"Topic {topic_num + 1}",
            'Theme': theme_dict[topic_num],
            '% of Songs': percent,
            'Top Words': ", ".join(top_words)
        }
        
        # Add the row to the appropriate table (positive or negative)
        if label == 'Positive':
            positive_table.append(row)
        else:
            negative_table.append(row)

        # Print topic information for each sentiment category
        print(f"\n{label} Sentiment Topic {topic_num + 1} — {theme_dict[topic_num]}:")
        print(f"Top words: {', '.join(top_words)}")
        
        # Get sample songs for each topic (if applicable)
        top_songs_df = subset[subset['dominant_topic'] == topic_num]
        
        if not top_songs_df.empty:
            sample_size = min(3, len(top_songs_df))  # Limit sample size to 3 songs
            top_songs = top_songs_df.sample(sample_size, random_state=1)
            
            print("Sample songs:")
            for _, row in top_songs.iterrows():
                print(f" - {row['title']} by {row['artist']}")
        else:
            print(" - No songs found for this topic.")

# Convert tables to DataFrames for display
positive_df = pd.DataFrame(positive_table)
negative_df = pd.DataFrame(negative_table)

# Print out the final tables
print("\nPositive Sentiment Topics")
print(positive_df)

print("\nNegative Sentiment Topics")
print(negative_df)

#VISUALISATIONS 
# FIGURE 3
custom_palette = {
    "Friendship, Reassurance & Beauty": "#9BBFF2", 
    "Fun, Fame & Social Buzz": "#E39A78",      
    "Love & Waiting": "#A3D8A5",                    
    "Optimism & Light": "#F7C873",                  
    "Sensuality & Confidence": "#C39BD3",           
    "Swagger & Hype Energy": "#B2B2B2"}

#Data Work
positive_df = english_df[english_df['sentiment_label'] == 'Positive'].copy()
positive_df['topic_theme'] = positive_df['dominant_topic'].map(positive_themes)
topic_counts_by_year = positive_df.groupby(['year', 'topic_theme']).size().reset_index(name='count')
total_counts_by_year = topic_counts_by_year.groupby('year')['count'].transform('sum')
topic_counts_by_year['percentage'] = topic_counts_by_year['count'] / total_counts_by_year * 100

#Plot
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=topic_counts_by_year,
    x='year',
    y='percentage',
    hue='topic_theme',
    marker='o',
    palette=custom_palette)
plt.title('Temporal Trends of Positive Sentiment Topics')
plt.xlabel('Year')
plt.ylabel('% of Positive Sentiment Songs')
plt.legend(title='Topic Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# FIGURE 4
negative_palette = {
    "Club Vibes & Seduction": "#76B6C4",        
    "Heartache & Disconnection": "#F19990",      
    "Hustle & Confrontation": "#7DBA4F",          
    "Nostalgia & Longing": "#E15759",           
    "Street Talk & Bravado": "#BC8DBF",        
    "Stress, Confusion & Burnout": "#9E9E9E"}

#Data Work
negative_df = english_df[english_df['sentiment_label'] == 'Negative'].copy()
negative_df['topic_theme'] = negative_df['dominant_topic'].map(negative_themes)
neg_topic_counts_by_year = negative_df.groupby(['year', 'topic_theme']).size().reset_index(name='count')
neg_total_counts_by_year = neg_topic_counts_by_year.groupby('year')['count'].transform('sum')
neg_topic_counts_by_year['percentage'] = neg_topic_counts_by_year['count'] / neg_total_counts_by_year * 100

#Plot
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=neg_topic_counts_by_year,
    x='year',
    y='percentage',
    hue='topic_theme',
    marker='o',
    palette=negative_palette)
plt.title('Temporal Trends of Negative Sentiment Topics')
plt.xlabel('Year')
plt.ylabel('% of Negative Sentiment Songs')
plt.legend(title='Topic Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

