from os import path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import STOPWORDS, WordCloud

df = pd.read_csv('C:\\Users\\Konio\\Desktop\\datasets\\q1\\train.csv')
stopwords = set(STOPWORDS)

wholedf = df.groupby(['Label'])
tech = df.loc[df['Label']=="Technology"]
techWordsForCloud = tech['Title'].astype(str) + tech['Content']
textTech = " ".join(t for t in techWordsForCloud)

business = df.loc[df['Label']=="Business"]
businessWordsForCloud = business['Title'].astype(str) + business['Content']
textBusiness = " ".join(t for t in businessWordsForCloud)

entertainment = df.loc[df['Label']=="Entertainment"]
entertainmentWordsForCloud = entertainment['Title'].astype(str) + entertainment['Content']
textEntertainment = " ".join(t for t in entertainmentWordsForCloud)

health = df.loc[df['Label']=="Health"]
healthWordsForCloud = health['Title'].astype(str) + health['Content']
textHealth = " ".join(t for t in healthWordsForCloud)

wordcloudTech = WordCloud(width=1600, height=800, max_font_size=300, max_words=150, background_color="white", stopwords=stopwords, colormap="gnuplot").generate(textTech)
plt.figure(figsize=(20,10), dpi=600)
plt.imshow(wordcloudTech, interpolation="bilinear")
plt.axis("off")
plt.savefig('tech.pdf')

wordcloudBusiness = WordCloud(width=1600, height=800, max_font_size=300, max_words=150, background_color="white", stopwords=stopwords, colormap="gnuplot").generate(textBusiness)
plt.figure(figsize=(20,10), dpi=600)
plt.imshow(wordcloudBusiness, interpolation="bilinear")
plt.axis("off")
plt.savefig('business.pdf')

wordcloudEntertainment = WordCloud(width=1600, height=800, max_font_size=300, max_words=150, background_color="white", stopwords=stopwords, colormap="gnuplot").generate(textentEntertainment)
plt.figure(figsize=(20,10), dpi=600)
plt.imshow(wordcloudEntertainment, interpolation="bilinear")
plt.axis("off")
plt.savefig('entertainment.pdf')

wordcloudHealth = WordCloud(width=1600, height=800, max_font_size=300, max_words=150, background_color="white", stopwords=stopwords, colormap="gnuplot").generate(textHealth)
plt.figure(figsize=(20,10), dpi=600)
plt.imshow(wordcloudHealth, interpolation="bilinear")
plt.axis("off")
plt.savefig('health.pdf')