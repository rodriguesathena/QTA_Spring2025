#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import lyricsgenius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import re

#API KEYS (deleted for privacy)
#GENIUS_API_KEY = ""
#SPOTIFY_CLIENT_ID = ""
#SPOTIFY_CLIENT_SECRET = ""
#genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15, retries=3)
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
 #   client_id=SPOTIFY_CLIENT_ID,
  #  client_secret=SPOTIFY_CLIENT_SECRET
#))


#Load and clean Billboard CSV
file_path = r"C:\Users\athen\Documents\GitHub\QTA_Spring2025\Final\api_based\hot-100-current.csv"
df = pd.read_csv(file_path)

#Process date and filter by year
df['chart_week'] = pd.to_datetime(df['chart_week'], errors='coerce')
df['year'] = df['chart_week'].dt.year
df = df[df['year'].between(2000, 2015)]
print(f"Year range in filtered df: {df['year'].min()} - {df['year'].max()}")
print(f"Rows after filtering: {len(df)}")

#Normalize
df['title_clean'] = df['title'].astype(str).str.strip().str.lower()
df['performer_clean'] = df['performer'].astype(str).str.strip().str.lower()

#Drop duplicates (NOT including year)
unique_songs = df.drop_duplicates(subset=['title_clean', 'performer_clean']).copy()

#Use original casing from first appearance
unique_songs = unique_songs[['title', 'performer', 'year']].reset_index(drop=True)

print(f"Confirmed unique songs: {len(unique_songs)}")


#Functions to get lyrics and genre
def get_lyrics(title, artist):
    try:
        song = genius.search_song(title, artist)
        return song.lyrics if song else ""
    except:
        return ""

def get_genres(title, artist):
    try:
        results = sp.search(q=f"track:{title} artist:{artist}", type='track', limit=1)
        if results['tracks']['items']:
            artist_id = results['tracks']['items'][0]['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            return ", ".join(artist_info['genres']) if artist_info['genres'] else "Unknown"
    except:
        return "Unknown"



#Data Collection
all_songs = []
total_processed = 0
total_songs = len(unique_songs)

#loop through each year, filter songs for the chosen year, print status (Year being worked on)
for year in sorted(unique_songs['year'].unique()):
    year_songs = unique_songs[unique_songs['year'] == year].reset_index(drop=True)
    total_year_songs = len(year_songs)
    print(f"\nProcessing year {year} ({total_year_songs} unique songs)")
    #loop through each song in the year list, 
    for year_idx, row in year_songs.iterrows():
        title = row['title']
        artist = row['performer']
    #print status of year and the current song being worked on
        current_song_num = total_processed + 1
        print(f"[{year}] {title} — {artist} ({current_song_num}/{total_songs} | {year_idx + 1}/{total_year_songs} in {year})")
        #get the lyrics and genre using defined functions
        lyrics = get_lyrics(title, artist)
        genre = get_genres(title, artist)

        all_songs.append({
            'title': title,
            'artist': artist,
            'year': year,
            'lyrics': lyrics,
            'genre': genre
        })
        #song counter increments, sleep to avoid overload
        total_processed += 1
        time.sleep(1)
    #print status when year is completed
    print(f"Done with {year} ({total_processed}/{total_songs} total)\n")



#CSV
#Convert list to dataframe
df_songs = pd.DataFrame(all_songs)

output_path = r"C:\Users\athen\Documents\GitHub\QTA_Spring2025\Final\api_based\song_data_final.csv"
df_songs.to_csv(output_path, index=False)

print(f"Saved {len(df_songs)} songs to CSV at:\n{output_path}")


#Final Dataset clean
path = r"C:\Users\athen\Documents\GitHub\QTA_Spring2025\Final\api_based\song_data_final.csv"
df = pd.read_csv(path)

def clean_lyrics(text):
    if not isinstance(text, str):
        return ""

    #Trim everything before the first real section label
    section_start = re.search(r"(\[Verse.*?]|\[Intro.*?]|\[Chorus.*?])", text, re.IGNORECASE)
    if section_start:
        text = text[section_start.start():]

    #Remove common footer or meta phrases (seen in Genius lyrics)
    noise_phrases = [
        r"(?i)you might also like.*",
        r"(?i)embed$",
        r"(?i)produced by.*",
        r"(?i)written by.*",
        r"(?i)genius annotation.*",
        r"(?i)see.*live.*version.*",
        r"(?i)more on genius.*"
    ]
    for phrase in noise_phrases:
        text = re.sub(phrase, "", text)
  
    text = re.sub(r"\n{2,}", "\n", text).strip()

    return text

df['lyrics_clean'] = df['lyrics'].apply(clean_lyrics)
df.head


#Save final dataset
output_path = r"C:\Users\athen\Documents\GitHub\QTA_Spring2025\Final\api_based\redo.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df_songs)} songs to CSV at:\n{output_path}")

