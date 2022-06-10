"""
@author: gabeshires
custom functions to accompany PRAC1.ipynb
"""

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from textblob import TextBlob
from nrclex import NRCLex
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as npfrom sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

def preprocessing_text(x):
    """
    function to preprocess description type columns, ready for nlp analysis
    """
    words = str(x)
    words = word_tokenize(words.lower())
    sw = stopwords.words('english')
    words = [w for w in words if w not in sw]
    words = [w for w in words if w.isnumeric() != True]
    words = [w for w in words if w not in string.punctuation]
    lem = nltk.stem.WordNetLemmatizer()
    words = [(lem.lemmatize(w)) for w in words]
    return ' '.join(words)

def sentiment_and_emotions_analysis(x):
    """
    function that will analysis text and return values for sentiment and emotions
    """
    try:
        return TextBlob(x).sentiment, NRCLex(x).affect_frequencies
    except:
        return None
    
def convert_to_list(col):
    """
    function to convert a dictionary type column to a list
    """
    col = "".join(col)
    col = col.replace('{','')
    col = col.replace('}','')
    col = col.replace('"','')
    col = col.split(',')
    return col

def to_1d(series):
    """
    function to convert a list to a flat series
    """
    return pd.Series([x for _list in series for x in _list])

def price_to_int(col):
    """
    function to convert a price in dollars to integer
    """
    col = col[1:-3]
    col = int(col.replace(",",""))
    return col

def categorical_plot(df, col, aggregation='median'):
    """
    function to plot categorical columns using their value counts and categories
    """
    if aggregation == 'median':
        temp = df.groupby(col)[['price','availability_90']].median().reset_index()
    elif aggregation == 'mean':
        temp = df.groupby(col)[['price','availability_90']].mean().reset_index()
    else:
        print('unavailable aggregation')
    
    fig = make_subplots(rows=4, cols=1,shared_xaxes=True,
                    specs=[[{"rowspan": 1}],
                           [{"rowspan": 3}],
                           [None],
                          [None]],
                    print_grid=True,
                   subplot_titles=(f"value_counts of {col}",
                                   f"{aggregation} price ($) of {col} with 'availability_90' as colour"))

    fig.add_trace(go.Bar(y=df[col].value_counts().sort_index()/len(df),
                     name="(1,1)",text=df[col].value_counts().sort_index()),
                  row=1, col=1)
              
    fig.add_trace(go.Bar(x=temp[col],y=temp['price'],
                     name="(2,1)",text=round(temp['price']),
                    marker=dict(color=temp['availability_90'],coloraxis="coloraxis")),
                  row=2, col=1)

    fig.update_layout(coloraxis=dict(colorscale='ylgn'), showlegend=False)
    fig.update_coloraxes(colorbar_title='availability_90')
    fig.show()

def boolean_plot(df, col):
    """
    function to plot boolean columns counts and how it affects median price
    """
    counts = df.groupby(col).size().reset_index().rename({0:'count'},axis=1).replace({0:'False',1:'True'})
    prices = df.groupby(col)['price'].median().reset_index().replace({0:'False',1:'True'})
                                                                     
    fig = make_subplots(rows=1, cols=2,subplot_titles=("category count", "median price ($)"))
    fig.add_trace(
        go.Bar(x=counts.iloc[:,0], y=counts.iloc[:,1],
               marker=dict(color=counts.iloc[:,1])),
        row=1, col=1)
    fig.add_trace(
        go.Bar(x=prices.iloc[:,0], y=prices.iloc[:,1],
               marker=dict(color=counts.iloc[:,1])),
        row=1, col=2)
    fig.update_layout(height=400, width=1000, title_x=0.5,
                      title_text=f"{col} plots",showlegend=False)
    fig.show()
    
def correlation_heatmap(df, width=1500, height=1500):
    """
    function to plot a correlation heatmap of all categories in df
    """
    corr = df.corr()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = corr.columns,
            y = corr.index,
            z = np.array(corr),
            colorscale='Geyser'
        )
    )
    fig.update_layout(
        width=width,
        height=height)
    fig.show()
    
def scores_and_residuals(y_test, y_pred, mean_position='top right', median_position='top left'):
    """
    function to print metrics for performance of ML algorithm, as well as plotting residuals
    """
    residuals = y_test - y_pred
    print("the r2 value for this model is:",round(r2_score(y_test, y_pred),4))
    print("the MAE value for this model is:",round(mean_absolute_error(y_test, y_pred),4))
    print("the MSE value for this model is:",round(mean_squared_error(y_test, y_pred),4))
    fig = px.histogram(residuals)
    fig.add_vline(x=residuals.mean(), line_width=3, line_dash="dash", line_color="green",
             annotation_text=f"mean:{round(residuals.mean(),2)}",
                  annotation_position=mean_position)
    fig.add_vline(x=residuals.median(), line_width=3, line_dash="dash", line_color="red",
             annotation_text=f"median:{round(residuals.median(),2)}",
                  annotation_position=median_position)
    fig.update_layout(height=400,width=900,showlegend=False)
    fig.show()
    
def print_changes_csv():
    """
    function to get df that was manually created when cleaning 'amenities' column
    """
    expr = ['Washer|Dryer|Microwave|Dishwasher|Oven|Refrigerator',
 'TV',
 'Wifi|Laptop friendly workspace',
 'Family/kid friendly|Children',
 'Elevator',
 'Long term stays allowed',
 'Coffee maker|Espresso machine',
 'Host greets you',
 'Free parking on premises|Free street parking',
 'Paid parking off premises|Paid parking on premises',
 'No stairs or steps to enter|Wheelchair accessible',
 'Bed linens|Extra pillows and blankets',
 'Self check-in',
 'Luggage dropoff allowed',
 'Breakfast',
 'Patio or balcony',
 'Garden or backyard',
 'Buzzer/wireless intercom',
 'Lockbox|Safety card|Lock on bedroom door|Fire extinguisher',
 'Private entrance']
    col = ['white_goods',
 'tv',
 'can_wfh',
 'child_friendly',
 'elevator',
 'long_term_stays',
 'coffee_maker',
 'host_greeting',
 'free_parking',
 'paid_parking',
 'accessible',
 'bed_extras',
 'self_check_in',
 'luggage_dropoff',
 'breakfast_provided',
 'balcony',
 'outdoor_space',
 'intercom',
 'extra_safety',
 'private_entrance']
    return pd.DataFrame(list(zip(expr,col)),columns=['expr','col'] )

