# 1. read tabular data file intp pandas
	import pandas as pd
	orders = pd.read_table('http://bit.ly/chiporders')

	user_cols = ['user_id', 'age', 'gender','occupation', 'zip_code'] #Corrections for Bad Example
	users = pd.read_table('http://bit.ly/movieusers',sep='|', header=None, names=user_cols)
	users.head()

# 2 Select a Panda Series from a Dataframe
	ufo['City'] #bracket notation
	type(ufo['City'])
	
	ufo.City #dot notation; find other attributes by hitting tab on ufo.
	
	# Creating a new Series DataFrame by combining existing columns
	# Must use bracket notation when assigning new series in a dataframe
	ufo['Location'] = ufo.City + ', ' + ufo.State 
	
# 3 Pandas Method and Attributes
	movies = pd.read_csv('http://bit.ly/imdbratings')
	movies.head() #method
	movies.describe() #method
	drinks.describe(include='all')
	drinks.describe(include=['object','float64'])

	movies.shape #attribute
	movies.dtypes #attribute
	ufo.columns
	ufo.shape

	movies.describe(include=['object']) #TIP
	
# 4 Rename Columns
	ufo = pd.read_csv('http://bit.ly/uforeports')
	ufo.head()
	ufo.columns
	# Overwriting Selected Columns
	ufo.rename(columns = {'Colors Reported' : 'Colors_Reported', 'Shape Reported' : 'Shape_Reported'}, inplace=True)
	ufo.columns

	# Overwritting All columns
	ufo_cols = ['city','colors reported','shape report','state','time']
	ufo.columns = ufo_cols

	# Overwriting when importing
	ufo = pd.read_csv('http://bit.ly/uforeports',names=ufo_cols,header=0)

	# Modifying Columns w/ str
	ufo.columns = ufo.columns.str.replace(' ', '_') #Replacing space with _
	
# 5 Removing Data; Drop rows and columns

	# Dropping Columns
	ufo.drop('Colors Reported',axis=1,inplace=True) # drop one
	ufo.drop(['City','State'],axis=1,inplace=True) # drop multiple

	# Dropping Rows
	ufo.drop([0,1],axis=0,inplace=True) 
		# axis=0 is default 
		# best pratice, always define axis=0 or 1
		
# 6 Sorting Pandas DataFrame or a Series

	# Series
	movies.title.sort_values() 
	movies['title'].sort_values()
	movies.title.sort_values(ascending=False)
	
	# Sorting DataFrame by a Series
	movies.sort_values('title')
	movies.sort_values(['genre','duration'], ascending=False)
	
# 7 Filtering
	# filtering and booleans
	booleans = []
	for length in movies.duration:
		if length >= 200:
			booleans.append(True)
		else:
			booleans.append(False)
	is_long = pd.Series(booleans)
	movies[is_long]
	
	is_long = movies.duration >= 200
	is_long.head() # Series of booleans
	
	movies[movies.duration >= 200]
	movies[movies.duration >= 200]['genre']
	
	# using loc
	movies.loc[movies.duration >= 200, 'genre'] #loc
	
# 8 Filtering Multiple Criteria
	movies[movies.duration >= 200]
	
	# and
	movies[(movies.duration >= 200) & (movies.genre == 'Drama')]
	
	# or
	movies[(movies.duration >= 200) | (movies.genre == 'Drama')] 
	
	(movies.duration >= 200) | (movies.genre == 'Drama') # This will result a boolean
	
	# .isin()
	movies.genre.isin(['Crime','Drama','Action']) #boolean
	movies[movies.genre.isin(['Crime','Drama','Action'])]
	
# 9 selecting while loading csv: usecols and looping
	# Load csv and select columns
	ufo = pd.read_csv('http://bit.ly/uforeports',usecols=['City','State']) #usecols=[0,4]
	ufo.columns

	ufo = pd.read_csv('http://bit.ly/uforeports',nrows=3)
	ufo
		
	# interation in Series and DataFrame
	# Series
	for c in ufo.City:
		print(c)
		
	# DataFrame
	for index, row in ufo.iterrows():
		print(index, row.City, row.State)
		
	# dropping every non-numeric column from a DataFrame
	drinks = pd.read_csv('http://bit.ly/drinksbycountry')
	drinks.dtypes
	import numpy as np
	drinks.select_dtypes(include=[np.number]).dtypes

	# describe with selecting datatype
	drinks.describe(include='all')
	drinks.describe(include=['object','float64'])
	
# 10 Axis
	drinks = pd.read_csv('http://bit.ly/drinksbycountry')
	drinks.drop('continent', axis=1).head() # dropping column
	drinks.drop(2, axis=0).head() # dropping row

	# Operation on rows; mean of each column
	drinks.mean()
	drinks.mean(axis=0)
	drinks.mean(axis='index') # axis=0

	# Operation on columns; mean of each row
	drinks.mean(axis=1) # moving "left/right"
	drinks.mean(axis='columns') # axis=1
	
# 11 Pandas Str Method (https://pandas.pydata.org/pandas-docs/stable/reference/index.html)
	orders = pd.read_table('http://bit.ly/chiporders')
	orders.item_name.str.upper()
	orders.item_name.str.contains('Chicken')

	#Chaining Str Method
	orders.choice_description.str.replace('[','').str.replace(']','')
	orders.choice_description.str.replace('[\[\]]','')

# 12 Changing Data Type
	drinks = pd.read_csv('http://bit.ly/drinksbycountry')
	drinks.dtypes
	drinks['beer_servings'] = drinks.beer_servings.astype(float)
	drinks.dtypes
	
	# Define Data Type before Loading csv
	drinks = pd.read_csv('http://bit.ly/drinksbycountry',dtype={'beer_servings':float})
	drinks.dtypes
	
	# str and astype chain
	orders.item_price.str.replace('$','').astype(float).mean()
	
	# boolean to 0 and 1
	orders.item_name.str.contains('Chicken').head()
	orders.item_name.str.contains('Chicken').astype(int).head()
	