#%% Introduction
"""
Author: Rushabh Barbhaya
Guided by: Dr. Carlo Lipizzi 
	profile -> https://web.stevens.edu/facultyprofile/?id=2186
Subject: EM624
	Informatics for Engineering Management

Project Topic: Movie Recommendation System
Dataset: The Movies Dataset
	link -> https://www.kaggle.com/rounakbanik/the-movies-dataset
	Dataset size -> 990mb (7 csv files / 5 used)
		csv1:
		Name: movies_metadata.csv
		entries: 45466 x 24 data entries
		csv2:
		Name: credits.csv
		entries: 25476 x 3 data entries
		csv3:
		Name: keywords.csv
		entires: 46419 x 2 data entries
		csv4:
		Name: links_small.csv
		entries: 9112 entries
		csv5:
		Name: ratings.csv
		entries: 100004 x 4 entries

Features:
	User Based Search Recommendation
	Provides chart toppers for the movie entered
	Provides data story
	Describes the movie selected

---------------------------------------------------------------------
Requirements: Microsoft Visual Studio v14+
Warnings: Computational Heavy Code!
---------------------------------------------------------------------

Limitations:
	Couldn't run the code on whole dataset due to lack of computational power.
"""

#%% Initializing section
git remote add origin https://github.com/rhbarbhaya/User-Based-Movie-Recommendation-System.git
git push -u origin master
import os # Importing OS library to invoke installation IDE
try:
	import time # Import time library
except ImportError:
    os.system("python -m pip install time")
try:
	import pandas as pd # Importing Pandas library
except ImportError: #If import error occurs. Install that library
	os.system("python -m pip install pandas")
try:
	from ast import literal_eval # Importing literal_eval from ast
except ImportError: #If import error occurs. Install that library
	os.system("python -m pip install ast")
try:
	import numpy as np # Importing Numerical Python library
except ImportError: #If import error occurs. Install that library
	os.system("python -m pip install numpy")
try:
	from wordcloud import WordCloud, STOPWORDS
except ImportError:
	os.system("python -m pip install wordcloud")
try:
	import matplotlib.pyplot as plt
except ImportError:
	os.system("python -m pip install matplotlib")
try:
	import plotly
	import plotly.offline as py
	py.init_notebook_mode(connected=True)
	import plotly.graph_objs as go
	import plotly.tools as tls
except ImportError:
	os.system("python -m pip install plotly")
try:
	import warnings
except ImportError:
	os.system("python -m pip install warnings")
try:
	from sklearn.feature_extraction.text import TfidVectorizer, CountVectorizer
	from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
except ImportError:
	os.system("python -m pip install sklearn")
try:
	from nltk.stem.snowball import SnowballStemmer
	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import wordnet
except ImportError:
	os.system("python -m pip install nltk")
try:
	from surprise import Reader, Dataset, SVD, evaluate
except ImportError:
	os.system("python -m pip install scikit-surprise")


import time
import pandas as pd
from ast import literal_eval
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import warnings
from IPython.display import Image, HTML
sns.set_style('whitegrid')
sns.set(font_scale=1.25)
pd.set_option('display.max_colwidth', 50)
warnings.filterwarnings("ignore")
plotly.tools.set_credentials_file(username='rounakbanik', api_key='xTLaHBy9MVv5szF4Pwan')

print ("\n\n")
print ("-- Welcome to the Movie Recommendation System --") #Welcome statement
time.sleep(1) # Halt the execution for 1 second
print ("-- This is Recommendation Enginer v3.2 --") #Version Number
time.sleep(1) # Halt the execution for 1 second

try:
	import datetime #Importing datetime library
except ImportError:
	os.system("python -m pip install datetime")

now = datetime.datetime.now() # Invoking Datetime settings 
print ("\nCurrent Date:", now.year, "-", now.month, "-", now.day) # Printing current data in yyyy-mm-dd format
time.sleep(1)
print ("Time:", now.hour, ":", now.minute) # Printing current date-time in 24hour hh:mm format
time.sleep(1)

# all the datasets
dataset = pd.read_csv("movies_metadata.csv") # Importing the dataset
dataset = dataset.drop([19730, 29503, 35587]) # Dropping error rows
credit = pd.read_csv("credits.csv") # Importing Cast and Directors Dataset
keywords = pd.read_csv("keywords.csv") # Importing keywords dataset. They related to the movie database
links = pd.read_csv("links_small.csv") # Smaller dataset as compared to links.csv
ratings = pd.read_csv("ratings_small.csv")

#%% Intro to dataset

dataset["title"] = dataset["title"].astype("str")
dataset["overview"] = dataset["overview"].astype("str")
dataset["revenue"] = pd.to_numeric(dataset["revenue"], errors = "coerce")
dataset["revenue"] = dataset["revenue"].replace(0, np.nan) # Cleaning revenue column of the dataset
#dataset["revenue"] = dataset["revenue"].astype("int")
dataset["budget"] = pd.to_numeric(dataset["budget"], errors = "coerce") # Converting the budget column to numeric values
dataset["budget"] = dataset["budget"].replace(0, np.nan) # Cleaning the budget column of the dataset
dataset["return"] = dataset["revenue"] / dataset["budget"] # Return value on movies
dataset["year"] = pd.to_datetime(dataset["release_date"], errors = "coerce").apply(lambda x: str(x).split("-")[0] if x != np.nan else np.nan) # Creating a year column to check the year of release
poster_url = "http://image.tmdb.org/t/p/w185" # Heading for movie Poster
dataset["poster_path"] = "<img src='" + poster_url + dataset["poster_path"] + "'style='height:100px;'>" # Adding Path for movie poster - Viz works best in Jupyter Notebook
links = links[links["tmdbId"].notnull()]["tmdbId"].astype("int")


# For Title Wordcloud
print ("\n-- Generating Wordcloud of Movie Titles --")
title_wc = " ".join(dataset["title"])
title_wordcloud = WordCloud(stopwords = STOPWORDS, background_color = "white", height = 2000, width = 4000).generate(title_wc)
plt.figure(figsize = (16,9))
plt.imshow(title_wordcloud)
plt.axis("off")
plt.title("Wordcloud of Movie Titles\n\n")
plt.show()

# Country-wise movies
	# Cleaning the production countries column
dataset["production_countries"] = dataset["production_countries"].fillna("[]").apply(literal_eval)
dataset["production_countries"] = dataset["production_countries"].apply(lambda x: [i["name"] for i in x] if isinstance(x,list) else [])
	# Genreating countries chart
countries_chart = dataset.apply(lambda x: pd.Series(x["production_countries"]), axis = 1).stack().reset_index(level = 1, drop = True)
countries_chart.name = "countries"
country_dataset = dataset.drop("production_countries", axis = 1).join(countries_chart)
country_dataset = pd.DataFrame(country_dataset["countries"].value_counts())
country_dataset["country"] = country_dataset.index
country_dataset.columns = ["Total Movies", "Country"]
country_dataset = country_dataset.reset_index().drop("index", axis = 1)
print ("\nTop 10 movie production countries according to the dataset are:\n", country_dataset.head(10))
time.sleep(5)
	# Map of movie producing countries
data = [dict(type = 'choropleth',
        locations = country_dataset['Country'],
        locationmode = 'country names',
        z = country_dataset['Total Movies'],
        text = country_dataset['Country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(0, 0, 255)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]

layout = dict(
    title = 'Production Countries for the MovieLens Movies',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
plt.figure()
py.iplot( fig, validate=False, filename='d3-world-map' )
plt.show()
time.sleep(3)

	# Collection Data
dataset = dataset[dataset['belongs_to_collection'].notnull()]
dataset['belongs_to_collection'] = dataset['belongs_to_collection'].apply(literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)
dataset = dataset[dataset['belongs_to_collection'].notnull()]
#creating pivot table to check
#1. Most successful movie franchise by total collection
franchise_pivot = dataset.pivot_table(index='belongs_to_collection', values='revenue', aggfunc={'revenue': ['mean', 'sum', 'count']}).reset_index()
print ("\nTop 10 Movie Franchise Collections\n", franchise_pivot.sort_values('sum', ascending=False).head(10))
time.sleep(5)
#2. Average collection per movie
print ("\nTop 10 Movie Franchise (Average)\n", franchise_pivot.sort_values("mean", ascending = False).head(10))
time.sleep(5)
#3. Total movie under a perticular franchise
print ("\nTop 10 Longest Running Franchise\n", franchise_pivot.sort_values("count", ascending = False).head(10))
time.sleep(5)

    # Cleaning Spoken Languages
dataset["spoken_languages"] = dataset["spoken_languages"].fillna("[]").apply(literal_eval)
dataset["spoken_languages"] = dataset["spoken_languages"].apply(lambda x: [i["name"] for i in x] if isinstance(x,list) else [])

	# Biggest Production Companies
dataset["production_companies"] = dataset["production_companies"].fillna("[]").apply(literal_eval)
dataset["production_companies"] = dataset["production_companies"].apply(lambda x: [i["name"] for i in x] if isinstance(x,list) else [])
companies = dataset.apply(lambda x: pd.Series(x["production_companies"]), axis = 1).stack().reset_index(level = 1, drop = True)
companies.name = "companies"
companies_dataset = dataset.drop("production_companies", axis = 1).join(companies)
companies_sum = pd.DataFrame(companies_dataset.groupby("companies")["revenue"].sum().sort_values(ascending = False))
companies_sum.columns = ["Total"]
companies_average = pd.DataFrame(companies_dataset.groupby("companies")["revenue"].mean().sort_values(ascending = False))
companies_average.columns = ["Avergae"]
companies_count = pd.DataFrame(companies_dataset.groupby("companies")["revenue"].count().sort_values(ascending = False))
companies_count.columns = ["Count"]
companies_chart = pd.concat((companies_sum, companies_average, companies_count), axis = 1)
print ("\nTop 10 production companies are:\n", companies_chart.sort_values("Total", ascending = False).head(10))
time.sleep(5)

	# Overall Expensive Movies
print ("\nTop 10 Most Expensive Movies are:\n", dataset[dataset["budget"].notnull()][["title", "budget", "revenue", "return", "year"]].sort_values("budget", ascending = False).head(10))

	# Top Grosssing Movies
top_grossing = dataset[["poster_path", "title", "budget", "revenue", "year"]].sort_values("revenue", ascending = False)
print (top_grossing.head(10))
#pd.set_option("dislay.max_colwidth", 100)
HTML(top_grossing.to_html(escape = False))
#pd.set_option('display.max_colwidth', 50)

#%% Movie Selection

# All the objects
def build_chart(genre, percentile = 0.85):
	df = genre_database[genre_database["genre"] == genre]
	vote_counts = df[df["vote_count"].notnull()]["vote_count"].astype("int")
	vote_averages = df[df["vote_average"].notnull()]["vote_average"].astype("int")
	C = vote_averages.mean()
	m = vote_counts.quantile(percentile)
	qualified = df[(df["vote_count"] >= m) & (df["vote_count"].notnull()) & (df["vote_average"].notnull())][["title", "year", "vote_count", "vote_average", "popularity"]]
	qualified["vote_count"] = qualified["vote_count"].astype("int")
	qualified["vote_average"] = qualified["vote_average"].astype("int")
	qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
	qualified = qualified.sort_values("wr", ascending = False).head(250)
	return qualified

def get_director(x):
	for i in x:
		if i["job"] == "Director":
			return i["name"]
	return np.nan

def filter_keywords(x):
	words = []
	for i in x:
		if i in keywords_dataset:
			words.append(i)
	return words

def convert_int(x):
	try:
		return int(x)
	except:
		return np.nan

def get_recommendations(title):
    idx = indices[title]
    simulate_scores = list(enumerate(cosine_sim[idx]))
    simulate_scores = sorted(simulate_scores, key=lambda x: x[1], reverse=True)
    simulate_scores = simulate_scores[1:31]
    movie_indices = [i[0] for i in simulate_scores]
    return titles.iloc[movie_indices]

def hybrid(username, SELECTED):
	idx = indices[SELECTED]
	tmdbId = id_map.loc[SELECTED]["id"]
	movie_id = id_map.loc[SELECTED]["movieId"]
	simulate_scores = list(enumerate(cosine_sim[int(idx)]))
	simulate_scores = sorted(simulate_scores, key = lambda x: x[1], reverse = True)
	simulate_scores = simulate_scores[1:26]
	movies = linked_dataset.iloc[movie_indices][["title", "vote_count", "vote_average", "year", "id"]]
	movies["estimate"] = movies["id"].apply(lambda x: svd.predict(username, indices_map.loc[x]["movieId"]).est)
	movies = movies.sort_values("estimate", ascending = False)
	return movies.head(10)

dataset["genres"] = dataset["genres"].fillna("[]").apply(literal_eval).apply(lambda x: [i["name"] for i in x] if isinstance(x, list) else []) # Cleaning the genre column

"""
We are going to use IMDB weighted average formula to score average ratings for a movie
	link -> https://help.imdb.com/article/imdb/track-movies-tv/faq-for-imdb-ratings/G67Y87TFYYP6TWAV#
	Formula: Weighted Rating (WR) = ((v/v+m)R) + ((m/v+m)C)
	v = number of votes for movies
	m = minimum votes required
	R = Average movie rating
	C = Overall Average vote
"""

print ("\n")
print ("We are going to use IMDB's weighted average formula: Weighted Rating (WR) = ((v/v+m)R) + ((m/v+m)C)\nv = number of votes for movies\nm = minimum votes required\nR = Average movie rating\nC = Overall Average vote")
time.sleep(2)
print ("\nWe are using this for getting average user ratings")
time.sleep(1)
genre_chart = dataset.apply(lambda x: pd.Series(x["genres"]), axis = 1).stack().reset_index(level = 1, drop = True)
genre_chart.name = "genre"
genre_database = dataset.drop("genres", axis = 1).join(genre_chart)

keywords["id"] = keywords["id"].astype("int") # Making id column as int
credit["id"] = credit["id"].astype("int") # Making id column as int
dataset["id"] = dataset["id"].astype("int") # Making id column as int
dataset = dataset.merge(credit, on = "id") # Merging credit dataset with original dataset
dataset = dataset.merge(keywords, on = "id") # Merfing keywords dataset with original dataset
linked_dataset = dataset[dataset["id"].isin(links)] # Creating a new dataframe
linked_dataset["cast"] = linked_dataset["cast"].apply(literal_eval) # Cleaning cast
linked_dataset["crew"] = linked_dataset["crew"].apply(literal_eval) # Cleaning crew
linked_dataset["keywords"] = linked_dataset["keywords"].apply(literal_eval) # Cleaning keywords
linked_dataset["cast_size"] = linked_dataset["cast"].apply(lambda x: len(x)) # Getting length on crew
linked_dataset["crew_size"] = linked_dataset["crew"].apply(lambda x: len(x)) # Getting length of cast
linked_dataset["director"] = linked_dataset["crew"].apply(get_director) # getting directors name
linked_dataset["cast"] = linked_dataset["cast"].apply(lambda x: [i["name"] for i in x] if isinstance(x,list) else []) # Getting other cast members
linked_dataset["cast"] = linked_dataset["cast"].apply(lambda x: x[:3] if len(x) >= 3 else x) # Limiting cast to 3 main people
linked_dataset["keywords"] = linked_dataset["keywords"].apply(lambda x: [i["name"] for i in x] if isinstance(x,list) else []) 
# Printing the details of the selected Movie
# Generating a search directory
searchable = linked_dataset[["id", "title"]]
while True:
    userselection = input("\n\n>>  Enter the name of the movie you want the Recommendations on: ")
    search = searchable[searchable["title"].str.contains(userselection, case = False, na = False, regex = True)]
    print (search[:10])
    correct_search = str(input("\n >> Correct Search? [y]/n: "))
    if correct_search == "n" or correct_search == "N":
    	continue
    elif correct_search == "y" or correct_search == "Y":
    	break
    elif ValueError:
    	print ("Invalid Input")
    	continue
    else:
        print ("Something seems wrong")
        continue

selected_movie = linked_dataset.set_index(linked_dataset["id"], inplace = False)
selection = int(input("\n>>  Enter the 'ID' of the movie selected: "))
selection_data = selected_movie.loc[selection]
SELECTED = selection_data["title"]
GENRE = selection_data["genres"]
GENRE = GENRE[0]

print ("\n\n\n\nShowing the top 10 in that genre:\n")
print ((build_chart(GENRE)).head(10))
print ("\nComputational heavy process going on. Please wait...\n")
print ("----------------------------------------------")
pd.set_option('display.max_colwidth', 100)
print (selection_data["poster_path"])
print ("\n\n\nMovie Selected: ", selection_data["title"])
print ("\nOriginal Title: ", selection_data["original_title"])
print ("\nDirector: ", selection_data["director"])
print ("\nActors: ", selection_data["cast"])
print ("\nGenre: ", selection_data["genres"])
print ("Release: ", selection_data["release_date"])
print ("\nTagline: ", selection_data["tagline"])
print ("\nDescription: \n", selection_data["overview"])
print ("\nRun-Time: ", selection_data["runtime"], "min")
print ("Budget: ", selection_data["budget"])
print ("Spoken Language(s): ", selection_data["spoken_languages"])
print ("Language: ", selection_data["original_language"])
print ("-----------------------------------------------")
pd.set_option('display.max_colwidth', 50)
linked_dataset["cast"] =  linked_dataset["cast"].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x]) # Lowering case and removing spaces
linked_dataset["director"] = linked_dataset["director"].apply(lambda x: [x,x,x])
# creatig a directory of keywords
keywords_dataset = linked_dataset.apply(lambda x: pd.Series(x["keywords"]), axis = 1).stack().reset_index(level = 1, drop = True)
keywords_dataset.name = "keyword"
keywords_dataset = keywords_dataset.value_counts()
# we have no use for keywords which occur only once
keywords_dataset = keywords_dataset[keywords_dataset > 1]
stemmer = SnowballStemmer('english') # Setting stemmer languge to english; it converts plural to singular
linked_dataset["keywords"] = linked_dataset["keywords"].apply(filter_keywords)
linked_dataset["keywords"] = linked_dataset["keywords"].apply(lambda x: [stemmer.stem(i) for i in x])
linked_dataset["keywords"] = linked_dataset["keywords"].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# Creating a mixture for recommendation
linked_dataset["combine"] = linked_dataset["keywords"] + linked_dataset["cast"] + linked_dataset["director"] + linked_dataset["genres"]
linked_dataset["combine"] = linked_dataset["combine"].apply(lambda x: " ".join(str(i) for i in x))

# Vectorizing
count = CountVectorizer(analyzer = "word", ngram_range = (1,2), min_df = 0, stop_words = "english")
count_matrix = count.fit_transform(linked_dataset["combine"])
print ("\n\nGenerating cosine similarity for better recommendation")
print ("This process is computationally heavy, hence chose a smaller dataset for this purpose\nWon't affect the code")
time.sleep(5)
cosine_sim = cosine_similarity(count_matrix, count_matrix)

linked_dataset = linked_dataset.reset_index()
titles = linked_dataset["title"]
indices = pd.Series(linked_dataset.index, index = linked_dataset["title"])

#%%
# User Based Movie Recommendations

reader = Reader() 
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader) 
data.split(n_folds = 5) 
svd = SVD()
print ("\nCalculating RMSE for the training dataset\nThis process is also requires computation energy\nHence, chose a smaller dataset\n")
evaluate(svd, data, measures = ["RMSE", "MAE"])
print ("\nTraining the dataset")
trainset = data.build_full_trainset()
svd.train(trainset)

usernames = {"Carlo": 10, "Andy": 20, "Rushabh": 30, "Alkim": 40, "Aashna": 50, "Anurag": 60, "Nikhil": 70, "Nishit": 80, "Prayash": 90}
username = input("Enter Username: \n")
username = username.title()
if username in usernames.keys():
    print ("Username = ", username, "\nUnique No", usernames[username])
else:
    UniqueID = len(usernames) + 10
    usernames[username] = UniqueID
    print ("New User: ")
    print ("Username: ", username, "\nUser Number: ", UniqueID)

id_map = pd.read_csv("links_small.csv")[["movieId", "tmdbId"]]
id_map["tmdbId"] = id_map["tmdbId"].apply(convert_int)
id_map.columns = ["movieId", "id"]
id_map = id_map.merge(linked_dataset[["title", "id"]], on = "id").set_index("title")
indices_map = id_map.set_index("id")

idp = indices[SELECTED]
sim_scores = list(enumerate(cosine_sim[idp]))
sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
sim_scores = sim_scores[1:26]
movie_indices = [i[0] for i in sim_scores]
movies = linked_dataset.iloc[movie_indices][["title", "vote_count", "vote_average", "year"]]
print ("\nRecommendations for ", username, ":")
print (hybrid(username, SELECTED))
time.sleep(5)
print ("\n\n\n-- End of simulation --")
print ("Author: Rushabh Barbhaya")
print ("\n-------------x-------------")