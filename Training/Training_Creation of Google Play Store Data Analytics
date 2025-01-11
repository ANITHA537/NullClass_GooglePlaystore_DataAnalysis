#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_googleplaystore=pd.read_csv("E:/NullClass LiveProject/googleplaystore.csv/googleplaystore.csv")


# In[3]:


data_googleplaystore.head()


# In[4]:


data_userreviews=pd.read_csv("E:/NullClass LiveProject/googleplaystore_user_reviews.csv/googleplaystore_user_reviews.csv")


# In[5]:


data_userreviews.head()


# In[6]:


data_googleplaystore.isnull()


# In[7]:


data_userreviews.isnull()


# In[8]:


data_googleplaystore=data_googleplaystore.dropna(subset=['Rating'])#removes all the rows where rating colum is nan/null


# In[9]:


data_googleplaystore.info()


# In[10]:


for col in data_googleplaystore.columns:
    data_googleplaystore[col].fillna(data_googleplaystore[col].mode()[0],inplace=True)


# In[11]:


data_googleplaystore.info()
data_googleplaystore.drop_duplicates(inplace=True)


# In[12]:


data_googleplaystore=data_googleplaystore[data_googleplaystore['Rating']>=0]
data_googleplaystore=data_googleplaystore[data_googleplaystore['Rating']<=5]


# In[13]:


data_googleplaystore.head()


# In[14]:


data_googleplaystore['Installs']=data_googleplaystore['Installs'].str.replace(',','').str.replace('+','').astype(int)


# In[15]:


data_googleplaystore['Price']=data_googleplaystore['Price'].str.replace('$','').astype(float)
data_userreviews.dropna(subset='Translated_Review',inplace=True)
data_userreviews.head()


# In[16]:


merged_data=pd.merge(data_googleplaystore,data_userreviews,on='App',how='inner')
merged_data.head()


# In[17]:


data_googleplaystore.dtypes


# In[18]:


data_userreviews.dtypes


# In[19]:


#data transformation


# In[20]:


def convert_size(n):
    if 'M' in n:
        return float(n.replace('M',''))
    elif 'K' in n:
        return float(n.replace('K',''))/1024
    else:
        return np.nan
data_googleplaystore['Size']=data_googleplaystore['Size'].apply(convert_size)


# In[21]:


data_googleplaystore['Size']


# In[22]:


data_googleplaystore['Log_Installs']=np.log1p(data_googleplaystore['Installs'])


# In[23]:


data_googleplaystore.head()


# In[24]:


data_googleplaystore.dtypes


# In[25]:


data_googleplaystore['Reviews']=data_googleplaystore['Reviews'].astype(int)


# In[26]:


data_googleplaystore.dtypes


# In[27]:


data_googleplaystore['Log_Reviews']=np.log1p(data_googleplaystore['Reviews'])


# In[28]:


data_googleplaystore.head()


# In[29]:


def rating_group(n):
    if n >=4:
        return "Top Rated App"
    elif n >=3:
        return "Above average"
    elif n>=2:
        return "Average"
    else:
        return "Below Average"
data_googleplaystore['Rating_Group']=data_googleplaystore['Rating'].apply(rating_group)


# In[30]:


data_googleplaystore.head()


# In[31]:


data_googleplaystore['Revenue']=data_googleplaystore['Price']*data_googleplaystore['Installs']
data_googleplaystore.head()


# In[32]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


# In[33]:


sia=SentimentIntensityAnalyzer()


# In[34]:


def sentiment_score(x):
    s=sia.polarity_scores(x)
    return s['compound']
data_userreviews['Sentiment_Score']=data_userreviews['Translated_Review'].apply(sentiment_score)
data_userreviews


# In[35]:


data_googleplaystore['Last Updated']=pd.to_datetime(data_googleplaystore['Last Updated'],errors="coerce")


# In[36]:


data_googleplaystore['Year Last Updated']=data_googleplaystore['Last Updated'].dt.year
data_googleplaystore.head()


# In[37]:


import plotly.express as px
fig=px.bar(x=['A','B','C'],y=[5,6,2],title='Sample bar chart')
fig.show()


# In[38]:


fig.write_html('E:/NullClass LiveProject/Training/Samplebarchart.html')


# In[39]:


import os as os
import plotly.io as pio
html_files_path="E:/NullClass LiveProject/Training"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
plot_containers=""
def save_plot_as_html(fig,filename,insight):
    global plot_containers
    filepath=os.path.join(html_files_path,filename)
    html_content=pio.to_html(fig,full_html=False,include_plotlyjs='inline')
    plot_containers+= f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')" >
        <div class="plot" > {html_content}</div>
        <div class='insights'>{insight}</div>
    </div>
    """
    fig.write_html(filepath,full_html=False,include_plotlyjs='inline')
plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}


# In[40]:


category_count=data_googleplaystore['Category'].value_counts()
category_count


# In[41]:


Top_10_categories=category_count.nlargest(10)


# In[42]:


fig1=px.bar(
    x=Top_10_categories.index,
    y=Top_10_categories.values,
    labels={'x':'Category','y':'Count'},
    title='Top 10 Categories on play store',
    color=Top_10_categories.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)

fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)


save_plot_as_html(fig1,"Top 10 Categories.html","The top categoriesin playstore are dominated by tools,entertain")


# In[43]:


fig1.show()


# In[44]:


type_count=data_googleplaystore['Type'].value_counts()
type_count


# In[45]:


fig2=px.pie(
    values=type_count.values,
    names=type_count.index,
    title='App type distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)

fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(l=10,r=10,t=30,b=10)
)


save_plot_as_html(fig2,"Apptypes.html","Most of the apps in google play store are free")

fig2.show()


# In[46]:


fig3=px.histogram(
    data_googleplaystore,
    x='Rating',
    nbins=20,
    
    title='Rating distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)

fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)


save_plot_as_html(fig3,"Rating distribution.html","Ratings are skewed towardshigher values,suggesting that most appsarerated favourably by users")

fig3.show()


# In[47]:


sentiment_counts=data_userreviews['Sentiment_Score'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment Score','y':'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)

fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)



save_plot_as_html(fig4,"Sentiment analysis.html","Sentiments in reviews show a mix ofpositive andnegative feedback,with a slight lean towards positive")
fig4.show()


# In[ ]:





# In[ ]:





# In[48]:


install_counts=data_googleplaystore.groupby('Category')['Installs'].sum()
fig5=px.bar(
    x=install_counts.index,
    y=install_counts.values,
    orientation='h',
    labels={'x':'installs','y':'Category'},
    title='Installs by Category',
    color=install_counts.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)

fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)



save_plot_as_html(fig5,"Installations By Category.html","The categories with most installs are socail and communication apps")
fig5.show()


# In[49]:


updates_per_year=data_googleplaystore['Last Updated'].dt.year.value_counts().sort_index()
fig6=px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x':'Year','y':'No.of Updates'},
    title='Number of updates over the year',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)

fig6.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)



save_plot_as_html(fig6,"Updates.html","Updates have been increasing over the years")
fig6.show()


# In[50]:


revenue_by_category=data_googleplaystore.groupby('Category')['Revenue'].sum()
fig7=px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x':'Category','y':'Revenue'},
    title='Revenue by Category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)

fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)



save_plot_as_html(fig7,"Revenue By Category.html","Categoriessuch as Business and productivity has higher revenues")
fig7.show()


# In[51]:


genre_counts=data_googleplaystore['Genres'].str.split(";",expand=True).stack().value_counts().nlargest(10)
fig8=px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x':'Genre','y':'Count'},
    title='Genere Counts',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)

fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)



save_plot_as_html(fig8,"Genre Count.html","Actual and Casualgames are most common")
fig8.show()


# In[52]:


fig9=px.scatter(
    data_googleplaystore,
    x='Last Updated',
    y='Rating',
    title='Impact of Last Update on Rating',
    color='Type',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    
    width=400,
    height=300
)

fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)


save_plot_as_html(fig9,"Latest update impact on rating.html","Weak Corelation between Last Updated and Rating")

fig9.show()


# In[53]:


fig10=px.box(
    data_googleplaystore,
    x='Type',
    y='Rating',
    title='Rating of Free Vs Paid Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    color='Type',
    width=400,
    height=300
)

fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)


save_plot_as_html(fig10,"Rating of Free Vs Paid Apps.html","Paid appsgenerally have higher rating than free apps")

fig10.show()


# In[54]:


plot_containers_split=plot_containers.split('</div>')


# In[55]:


if len(plot_containers_split) > 1:
    final_plot=plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers


# In[56]:


dashboard_html= """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name=viewport" content="width=device-width,initial-scale-1.0">
    <title> Google Play Store Review Analytics</title>
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
            background-color: #444
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify_content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0,0,0,0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container: hover .insights {{
            display: block;
        }}
        </style>
        <script>
            function openPlot(filename) {{
                window.open(filename, '_blank');
                }}
        </script>
    </head>
    <body>
        <div class= "header">
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


# In[57]:


final_html=dashboard_html.format(plots=plot_containers,plot_width=plot_width,plot_height=plot_height)


# In[58]:


dashboard_path=os.path.join(html_files_path,"web page.html")


# In[59]:


with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)


# In[60]:


import webbrowser
webbrowser.open('file://'+os.path.realpath(dashboard_path))


# In[ ]:




