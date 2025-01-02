# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:52:18 2025

@author: admin
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import plotly.io as pio
import webbrowser

data_googleplaystore=pd.read_csv("googleplaystore.csv")
data_userreviews=pd.read_csv("googleplaystore_user_reviews.csv")

data_googleplaystore.info()
data_userreviews.info()

data_googleplaystore.isnull()
data_userreviews.isnull()


data_googleplaystore=data_googleplaystore.dropna(subset=['Rating'])
data_userreviews=data_userreviews.dropna(subset=['Translated_Review'])
data_googleplaystore['Reviews']=pd.to_numeric(data_googleplaystore['Reviews'],errors='coerce')
#data_googleplaystore[data_googleplaystore['Category']=='GAMES']
data_googleplaystore['Category'].value_counts()


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk


nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

app_names=set(data_googleplaystore['App'].str.lower().unique())
excluded_words=stop_words.union(app_names)



filtered_apps=data_googleplaystore[(data_googleplaystore['Category'].str.contains("HEALTH_AND_FITNESS"))&(data_googleplaystore['Rating'] >=4.5)]

#filtered_apps

#f=set(filtered_apps['App'])

#for x in data_userreviews.App:
    #if x in f:
        #print(x)
        
        
merged_data=pd.merge(filtered_apps, data_userreviews,on='App',how='inner')
reviews_text=" ".join(merged_data['Translated_Review'].dropna().astype(str).tolist())

filtered_words = [
    word for word in reviews_text.split()
    if word.lower() not in excluded_words
]


wordcloud=WordCloud(width=400, height=400, background_color='white').generate(" ".join(filtered_words))
html_files_path = "build"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for >=4.5 Star Reviews in HEALTH_AND_FITNESS Category", fontsize=16)
wordcloud.to_file("build/wordcloud.png")
plt.show()
plt.close()

def convert_size(n):
    if 'M' in n:
        return float(n.replace('M',''))
    elif 'K' in n:
        return float(n.replace('K',''))/1024
    else:
        return np.nan
data_googleplaystore['Size']=data_googleplaystore['Size'].apply(convert_size)


data_googleplaystore['Last Updated']=pd.to_datetime(data_googleplaystore['Last Updated'],errors="coerce")

for col in data_googleplaystore.columns:
    data_googleplaystore[col].fillna(data_googleplaystore[col].mode()[0],inplace=True)
    
data_googleplaystore['Reviews']=data_googleplaystore['Reviews'].astype(int)
#data_googleplaystore['Installs']=data_googleplaystore['Installs'].str.replace(',','').str.replace('+','').astype(float)


# creating a list of dataframe columns


#for i in range(0,len(data_googleplaystore.columns)):

    # printing the third element of the colum
    #data_googleplaystore['Reviews'][i]=math.log(data_googleplaystore['Reviews'][i],10000)
    #data_googleplaystore[i][4]=math.log(data_googleplaystore[i][4],100)

#data_googleplaystore['Reviews']=np.log1p(data_googleplaystore['Reviews'])
#data_googleplaystore['Reviews']=(data_googleplaystore['Reviews']/100000)
    


def is_within_time_range1():
    current_time = datetime.now().time()
    return current_time >= datetime.strptime("15:00", "%H:%M").time() and current_time <= datetime.strptime("17:00", "%H:%M").time()



plot_containers=""
#wordcloud_filename=os.path.join(html_files_path, "wordcloud.png")
plot_containers+=f"""
<div class="plot-container">
    <img src="build/wordcloud.png" alt="Word Cloud of Categories" style="width:100%; height:auto;">
    <div class="insights">
        This word cloud represents the frequent words in the reviews of the apps rated  >=4.5 in HEALTH_AND_FITNESS Category .
    </div>
</div>
"""


def save_plot_as_html1(fig, filename, insight):
    global plot_containers
    html_content=pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    if is_within_time_range1():
        plot_containers+=f"""
        <div class="plot-container">
            <div class="plot">
                {html_content} 
            </div>
            <div class="insights">
                {insight} 
            </div>
        </div>
        """
    else:
        plot_containers+=f"""
        <div class="plot-container">
            <div class="message">
                This chart is only available between 3 PM IST to 5 PM IST.
            </div>
        </div>
        """
    fig.write_html(filename, full_html=False, include_plotlyjs='inline')


        
filtered_df1=data_googleplaystore[(data_googleplaystore['Rating']>=4.0)&(data_googleplaystore['Size']>=10)&(data_googleplaystore['Last Updated'].dt.month==1)]


filtered_df1['Installs']=filtered_df1['Installs'].str.replace(',','').str.replace('+','').astype(int)
filtered_df1['Reviews']=(data_googleplaystore['Reviews']/100000)

category_metrics = (
    filtered_df1.groupby('Category')
    .agg(
        Average_Rating=('Rating', 'mean'),
        Total_Reviews=('Reviews', 'sum'),
        Total_Installs=('Installs', 'sum')
    )
    .sort_values('Total_Installs', ascending=False)
    .head(10) 
)
print(category_metrics)

data=category_metrics.reset_index()  
melted_data = data.melt(id_vars='Category',value_vars=['Average_Rating','Total_Reviews'],var_name='Metric',value_name='Value')
    
fig=px.bar(
        melted_data,
        x='Category',
        y='Value',
        color='Metric',
        barmode='group',
        title="Top 10 App Categories: Average Rating & Total Reviews(in Lakhs)",
        labels={'Value': 'Value', 'Category': 'App Category'},
        text='Value'
    )

fig.update_layout(
        xaxis_title="App Category",
        yaxis_title="Value",
        legend_title="Metrics",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        title_font={'size':16},
        xaxis=dict(title_font={'size':10}),
        yaxis=dict(title_font={'size':10}),
        margin=dict(l=60,r=60,t=60,b=60)
    )    
fig.show()


save_plot_as_html1(fig,"Top 10 App Categories_Average Rating&Total Reviews.html","This grouped bar chart shows the Totalreviews in lakhs and average rating ofTop10 Categorieson installments,With Family as Top 1 Category")



def is_within_time_range2():
    current_time=datetime.now().time()
    return current_time>=datetime.strptime("17:00","%H:%M").time() and current_time<=datetime.strptime("19:00","%H:%M").time()

def save_plot_as_html2(fig, filename, insight):
    html_content=pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    if is_within_time_range2():
        global plot_containers
        plot_containers += f"""
        <div class="plot-container">
            <div class="plot">
                {html_content} 
            </div>
            <div class="insights">
                {insight} 
            </div>
        </div>
        """
    else:
        plot_containers += f"""
        <div class="plot-container">
            <div class="message">
                This chart is only available between 5 PM IST to 7 PM IST.
            </div>
        </div>
        """
    fig.write_html(filename, full_html=False, include_plotlyjs='inline')


data_googleplaystore.dropna(subset='Installs',inplace=True)

data_googleplaystore.drop_duplicates(inplace=True)
data_googleplaystore=data_googleplaystore[data_googleplaystore['Rating']>=0]
data_googleplaystore=data_googleplaystore[data_googleplaystore['Rating']<=5]

data_googleplaystore['Installs']=data_googleplaystore['Installs'].str.replace(',','').str.replace('+','').astype(int)

#data_googleplaystore['Installs']=data_googleplaystore['Installs'].str.replace(' ','0').astype('Int64')

print(data_googleplaystore)
      

#data_googleplaystore.dtypes

filtered_df2= data_googleplaystore[(data_googleplaystore['Rating']>3.5)&(data_googleplaystore['Category']=='GAME')&(data_googleplaystore['Installs']>50000)]
print(filtered_df2)

fig2= px.scatter(
    filtered_df2,
    x='Size',
    y='Rating',
    size='Installs',
    color='Category',
    title="Bubble Chart:App Size vs Rating Games > 3.5 Rating   &  > 50K Installs)",
    labels={'Size':'App Size (MB)','Rating':'Average Rating'},
    hover_name='App',
    hover_data={'Size':True,'Rating':True,'Installs':True}
)

# Update layout for better visualization
fig2.update_layout(
    xaxis_title="App Size (MB)",
    yaxis_title="Average Rating",
    legend_title="Category",
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':10}),
    yaxis=dict(title_font={'size':10}),
)


save_plot_as_html2(fig2, "Bubble_Chart_Games_Rating_Installs.html", "This bubble chart shows the relationship between app size, rating, and the number of installs for games with more than 50k installs and a rating greater than 3.5.Subway Surfers is the most installed app with an average rating of 4.5")
fig2.show()


plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':20}


dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            text-align: center;
            color: white;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            bottom: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            color: #fff;
            width: 250px;
            word-wrap: break-word;
            max-height: 150px;
            overflow: auto;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
        .message {{
            font-size: 1.5em;
            color: #fff;
            background-color: rgba(255, 0, 0, 0.7);
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            border: 2px solid #f00;
            margin: 20px auto;
            width: 80%;
            max-width: 600px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
    </div>
    <div class="container">
        {plots}
    </div>
</body>
</html>
"""


final_html= dashboard_html.format(plots=plot_containers, plot_width=650, plot_height=475)


dashboard_path=os.path.join(html_files_path,"index.html")

with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)


webbrowser.open('file://' + os.path.realpath(dashboard_path))
