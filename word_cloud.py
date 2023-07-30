import pandas as pd
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

df = pd.read_csv("D:\\HALLOTHON\comments-with-sentiment.csv")
type_of_sentiments=df["Sentiment"].unique()
# print(type_of_sentiments)
text=[]
for i in type_of_sentiments:
    text.append(str(df["body"][df["Sentiment"]==str(i)]))
    print(text)
   
postive_text=df["body"][df["Sentiment"]=="joy"]
text=str(text[0])
python_mask=np.array(PIL.Image.open("Old_Nike_logo.jpg"))
# plt.imshow(py)
wc= WordCloud(stopwords=STOPWORDS,mask=python_mask,contour_color="black",background_color="white",contour_width=3,min_font_size=3).generate(text)


plt.imshow(wc)
plt.axis("off")
plt.show()
# print(text[0])