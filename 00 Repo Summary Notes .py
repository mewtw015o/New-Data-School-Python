# 10. read tabular data file into pandas
# 11 Select a Panda Series from a Dataframe
# 12 Pandas Method and Attributes
# 13 Rename Columns
# 14 Removing Data; Drop rows and columns
# 15 Sorting Pandas DataFrame or a Series
# 16 Filtering
# 17 Filtering Multiple Criteria
# 18 selecting while loading csv: usecols and looping
# 19 Axis
# 20 Pandas Str Method 
# 21 Changing Data Type
# 22 Groupby
	# Chart/Graph
# 23 Exploring a Panda Series
# 24 Handle Missing Values
# 25 Index and Indices
# 25_2 Index - Index Aligment and Concat
# 26 Selecting Multiple rows and columns (loc and iloc)
# 27 inplace
# 28 DataFrame Efficency with Categorical Dtype
# 29 Kaggle, scikit learn, to_pickle, read_pickle
# 30 ufo.isnull() vs pd.isnull(ufo), ufo.sample for sampling
# 31 Creating Dummy Variables and Map
# 32 Dates and Times in Pandas
# 33 Find and Remove Duplicate rows in pandas
# 34 Settingwithcopywarning and using copy()
# 35 Display Options
# 36 Creating DataFrame
# 37 apply, map, and applymap
# 38 Multiindex
# 38 Merge
# 40
	# Create a datetime column from a DataFrame
	# Create a category column during file reading
	# Convert the data type of multiple columns at once
	# Apple Multiple Aggregations on a Series or Dataframe
# 41
	# ix replacement
	# aliases have been added for isnull and notnull (isna)
	# drop now accepts "index" and "columns" keywords
	# Ordered Categories must be specified indepedent of the data


# 10. read tabular data file intp pandas
	import pandas as pd
	orders = pd.read_table('http://bit.ly/chiporders')

	user_cols = ['user_id', 'age', 'gender','occupation', 'zip_code'] #Corrections for Bad Example
	users = pd.read_table('http://bit.ly/movieusers',sep='|', header=None, names=user_cols)
	users.head()
	
	user_cols = ['user_id', 'age', 'gender','occupation', 'zip_code'] #Corrections for Bad Example
	users = pd.read_table('http://bit.ly/movieusers',sep='|', header=None, names=user_cols, index_col='user_id')

# 11 Select a Panda Series from a Dataframe
	ufo['City'] #bracket notation
	type(ufo['City'])
	
	ufo.City #dot notation; find other attributes by hitting tab on ufo.
	
	# Creating a new Series DataFrame by combining existing columns
	# Must use bracket notation when assigning new series in a dataframe
	ufo['Location'] = ufo.City + ', ' + ufo.State 
	
# 12 Pandas Method and Attributes
	movies = pd.read_csv('http://bit.ly/imdbratings')
	movies.head() #method
	movies.describe() #method
	drinks.describe(include='all')
	drinks.describe(include=['object','float64'])
	drinks.continent.head()

	movies.shape #attribute
	movies.dtypes #attribute
	ufo.columns
	ufo.shape
	drinks.index

	movies.describe(include=['object']) #TIP
	
# 13 Rename Columns
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
	
	df.add_prefix('X_')
	df.add_suffix('_Y')
	
# 14 Removing Data; Drop rows and columns

	# Dropping Columns
	ufo.drop('Colors Reported',axis=1,inplace=True) # drop one
	ufo.drop(['City','State'],axis=1,inplace=True) # drop multiple

	# Dropping Rows
	ufo.drop([0,1],axis=0,inplace=True) 
		# axis=0 is default 
		# best pratice, always define axis=0 or 1
		
# 15 Sorting Pandas DataFrame or a Series

	# Series
	movies.title.sort_values() 
	movies['title'].sort_values()
	movies.title.sort_values(ascending=False)
	
	# Sorting DataFrame by a Series
	movies.sort_values('title')
	movies.sort_values(['genre','duration'], ascending=False)
	
# 16 Filtering
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
	
# 17 Filtering Multiple Criteria
	movies[movies.duration >= 200]
	
	# and
	movies[(movies.duration >= 200) & (movies.genre == 'Drama')]
	
	# or
	movies[(movies.duration >= 200) | (movies.genre == 'Drama')] 
	
	(movies.duration >= 200) | (movies.genre == 'Drama') # This will result a boolean
	
	# .isin()
	movies.genre.isin(['Crime','Drama','Action']) #boolean
	movies[movies.genre.isin(['Crime','Drama','Action'])]
	
# 18 selecting while loading csv: usecols and looping
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
	
# 19 Axis
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
	
# 20 Pandas Str Method (https://pandas.pydata.org/pandas-docs/stable/reference/index.html)
	orders = pd.read_table('http://bit.ly/chiporders')
	orders.item_name.str.upper()
	orders.item_name.str.contains('Chicken')

	#Chaining Str Method
	orders.choice_description.str.replace('[','').str.replace(']','')
	orders.choice_description.str.replace('[\[\]]','')

# 21 Changing Data Type
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
	
# 22 Groupby
	drinks.beer_servings.mean()
	drinks.groupby('continent').beer_servings.mean()
	drinks[drinks.continent=='Africa'].beer_servings.mean()

	drinks.groupby('continent').beer_servings.max()
	drinks.groupby('continent').beer_servings.min()

	drinks.groupby('continent').beer_servings.agg(['count','min','max','mean'])

	# Means of all columns
	drinks.groupby('continent').mean() #For all columns! 

	# Chart/Graph
	%matplotlib inline
	drinks.groupby('continent').mean().plot(kind='bar')
	
# 23 Exploring a Panda Series
	movies.dtypes
	movies.genre.describe()
	movies.genre.value_counts() # counts of unique values
	movies.genre.value_counts(normalize=True)
	movies.genre.value_counts().head()
	movies.genre.unique()
	movies.genre.nunique() # Number of unique

	# Cross Tab
	pd.crosstab(movies.genre, movies.content_rating) # Selecting Two Columns

	# Column Functions
	movies.duration.describe() #describe for numeric values type column
	movies.duration.mean()
	movies.duration.value_counts() #Might not be most useful for numeric values; Used for 0 and 1 or Catergries Data

	# # Chart/Graph
	%matplotlib inline
	movies.duration.plot(kind='hist')

	movies.genre.value_counts().plot(kind='bar')
	
# 24 Handle Missing Values
	ufo.tail() #NaN - Not a Number; Missing Value

	ufo.isnull().tail()
	ufo.notnull().tail()


	# Numbers of missing value in each columns; 
	# axis=0 by default: summing rows;operating the sum across the rows
	ufo.isnull().sum()

	# axis = 1; operating sum across columns; sum of each rows
	ufo.isnull().sum(axis=1)

	pd.Series([True,False,True]).sum() #True = 1 and False = 0. 2xTrue = 2

	# Filtering Series by Null/Nan in a Panda Frame
	ufo[ufo.City.isnull()]

	# What to do with null
	# Drop
	# drop a row if any of its value is missing
	ufo.dropna(how='any').shape 
	ufo.dropna(subset=['City','Shape Reported'], how='any').shape

	# drop a row if all of its value is missing
	ufo.dropna(how='all').shape 
	ufo.dropna(subset=['City','Shape Reported'], how='all').shape

	# filling Missing Value
	ufo['Shape Reported'].value_counts() #Missing Value is excluded
	ufo['Shape Reported'].value_counts(dropna=False) # Including Missing Value; NaN 2644

	ufo['Shape Reported'].fillna(value='VARIOUS', inplace=True)
	ufo['Shape Reported'].value_counts() #VARIOUS 2977 (2644+333)
		
	# fillna (inplace=False by default)
	ufo.fillna(method='bfill').tail()
	ufo.fillna(method='ffill').tail()
		
# 25 Index and Indices
	drinks.index
	drinks.columns
	drinks.shape

	drinks[drinks.continent=='South America'] #Notice the first row index = 6; Index stays with the row

	# loc method
	drinks.loc[23,'beer_servings']

	# set index
	drinks.set_index('country',inplace=True)
	drainks.head()
	drinks.index #Notice name='country'; index name

	drinks.loc['Brazil','beer_servings']

	drinks.index.name = None #Removing index name 'country'
	drinks.head()

	# reset the index from country to values
	drinks.index.name='country' #index name would become a column name in the reset!
	drinks.reset_index(inplace=True)
	drinks.head()

	# Describe
	drinks.describe() #describe is a dataframe
	drinks.describe().index
	drinks.describe().columns
	drinks.describe().loc['25%','beer_servings']
	
# 25_2 Index - Index Aligment and Concat
	drinks.head()
	drinks.continent.head()
	drinks.set_index('country',inplace=True)
	drinks.head()
	drinks.continent.head()

	# Using value_counts() as a Series
	drinks.continent.value_counts()
	drinks.continent.value_counts().index
	drinks.continent.value_counts().values
	drinks.continent.value_counts()['Africa']

	# Sorting the index
	drinks.continent.value_counts().sort_index()

	# Calculation
	people = pd.Series([3000000,85000],index=['Albania','Andorra'],name='population')
	drinks.beer_servings * people # Nan if country is not in the 'people Series'

	# Concat People Series to the drinks Dataframe (concat)
	pd.concat([drinks, people], axis=1) # axis = 1; two objects side by side
	
# 26 Selecting Multiple rows and columns (loc and iloc)
	# loc (Inclusive)
		ufo.loc[0,:]
		ufo.loc[[0,1,2],:]
		ufo.loc[0:2,:] #loc is inclusive

		#also, returning all columns; not preferred, use ufo.loc[0:2,:] instead!
		ufo.loc[0:2,:]
		#vs
		ufo.loc[0:2] 

		ufo.loc[:, 'City']
		ufo.loc[:,['City', 'State']]
		ufo.loc[:,'City':'State']

		ufo.loc[0:2, 'City':'State'] #ufo.head(3).drop('Time', axis=1) would return the same results
		ufo.head(3).drop('Time', axis=1)

		ufo[ufo.City=='Oakland'].head()
		ufo.loc[ufo.City=='Oakland', :].head()

		ufo.loc[ufo.City=='Oakland', 'State'].head() # Chain
		ufo[ufo.City=='Oakland'].State #DO NOT WORK

	# iloc (Exclusive)
		ufo.iloc[:,[0,3]].head()
		ufo.iloc[:,0:4].head() # Exclusive
		ufo.iloc[0:3,:] # Exclusive; Did not include row 3
		
# 27 inplace
	ufo.drop('City',axis=1).head()
	ufo.head() #drop by default inplace=False; original Dataframe not changed

	ufo.drop('City',axis=1, inplace=True) #nothing displayed when running
	ufo.head()

	ufo.dropna(how='any').shape # Inplace = False; For Experiment
	ufo.shape

	# Assignment statement method vs inplace; 
	# no differences in effiency in inplace
	ufo = ufo.set_index('Time') 

	# fillna (inplace=False by default)
	ufo.fillna(method='bfill').tail()
	ufo.fillna(method='ffill').tail()
	
# 28 DataFrame Efficency with Categorical Dtype
	drinks.info() # notice the memory usage
	drinks.info(memory_usage='deep') #real memory
	drinks.memory_usage(deep=True) #real column/Series Memory
	drinks.memory_usage(deep=True).sum()

	# Converting Strings to integer type; Use Catergory Type
	sorted(drinks.continent.unique()) # Length=6
	drinks['continent'] = drinks.continent.astype('category')
	drinks.dtypes # Check
	drinks.continent.head() # Check
	drinks.continent.cat.codes.head()
	drinks.memory_usage(deep=True) #size of continent is smaller

	# Bad Case
	drinks['country'] = drinks.country.astype('category')
	drinks.country.cat.categories # length=193

	# Ordered Categories
	df = pd.DataFrame({'ID' : [100,101,102,103], 'quality' : ['good','very good', 'good', 'excellent']})
	df.sort_values('quality') # Sorting by str order

	from pandas.api.types import CategoricalDtype # New Code
	cats=['good','very good','excellent']
	cat_type = CategoricalDtype(categories=cats, ordered=True)
	df['quality'] = df.quality.astype(cat_type)

	df.quality
	df.sort_values('quality')

	df.loc[df.quality > 'good',:]
	
# 29 Kaggle, scikit learn, to_pickle, read_pickle
	train = pd.read_csv('http://bit.ly/kaggletrain')
	feature_cols = ['Pclass','Parch']
	x = train.loc[:,feature_cols] # x.shape (891,2)
	y = train.Survived # y.shape (891,)

	from sklearn.linear_model import LogisticRegression
	logreg = LogisticRegression()
	logreg.fit(x,y)

	test = pd.read_csv('http://bit.ly/kaggletest')
	test.head()
	x_new = test.loc[:,feature_cols]
	x_new.shape
	new_pred_class = logreg.predict(x_new)

	pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').to_csv('29_out.csv')

	train.to_pickle('train.pkl')
	pd.read_pickle('train.pkl')
	
# 30 ufo.isnull() vs pd.isnull(ufo), ufo.sample for sampling
	ufo = pd.read_csv('http://bit.ly/uforeports')
	pd.isnull(ufo).head()
	ufo.isnull().head()

	# Random Sampling
	ufo.sample(n=3)
	ufo.sample(n=3,random_state=42)
	ufo.sample(frac=0.1, random_state=99)
	ufo.shape
	
	# frac
	train = ufo.sample(frac=0.75, random_state=99) # sampling 75% of the data
	test = ufo.loc[~ufo.index.isin(train.index),:] # notice ~ ; Getting the non overlapping 25% into test
	test.shape
	train.shape
	
# 31 Creating Dummy Variables and Map
	train = pd.read_csv('http://bit.ly/kaggletrain')
	train.head()

	train['Sex_male'] = train.Sex.map({'female':0, 'male':1})
	train.head()

	# Creating one column for each possible values
	pd.get_dummies(train.Sex) 

	train.Embarked.value_counts()
	embarked_dummies = pd.get_dummies(train.Embarked, prefix='Embarked').iloc[:,1:]
	pd.get_dummies(train.Sex, prefix='Sex').iloc[:,1:] # Numbers of dummy variables = k - 1
	train = pd.concat([train, embarked_dummies], axis=1)

	# Bonus
	train = pd.read_csv('http://bit.ly/kaggletrain')
	train.head()
	pd.get_dummies(train, columns=['Sex', 'Embarked']).head() # Dataframe into get_dummies
	pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True).head() # Dataframe into get_dummies with k-1 variables
	
# 32 Dates and Times in Pandas
	ufo = pd.read_csv('http://bit.ly/uforeports')
	ufo.dtypes # Time object
	ufo.Time.str.slice(-5,-3).astype(int).head()
	ufo['Time'] = pd.to_datetime(ufo.Time)
	ufo.dtypes # Time datetime64[ns]

	ufo.Time.dt.hour
	ufo.Time.dt.weekday # To find more: Datetime Properties: Series.dt.data

	# Define a time limit and apply to dataframe
	ts = pd.to_datetime('1/1/1999')
	ufo.loc[ufo.Time >= ts, :].head()

	ufo.Time.max() - ufo.Time.min() # Timedelta('25781 days 01:59:00')
	(ufo.Time.max() - ufo.Time.min()).days

	# Graph/Plot
	%matplotlib inline
	ufo['Year'] = ufo.Time.dt.year
	ufo.Year.value_counts().sort_index().plot()
	
# 33 Find and Remove Duplicate rows in pandas
	user_cols = ['user_id', 'age', 'gender','occupation', 'zip_code'] #Corrections for Bad Example
	users = pd.read_table('http://bit.ly/movieusers',sep='|', header=None, names=user_cols, index_col='user_id')

	# duplicates in Series
	users.zip_code.duplicated()
	users.zip_code.duplicated().sum()

	# duplicates in DataFrame
	users.duplicated() # If an 'Entire' row are the same as any rows above it

	# Numbers of duplicates
	users.duplicated().sum()

	# Marking duplicates
	users.loc[users.duplicated(),:]
	users.loc[users.duplicated(keep='first'),:] #first default: 1st record is kept while its other duplicated records are marked
	users.loc[users.duplicated(keep='last'),:]
	users.loc[users.duplicated(keep=False),:] # Marking all duplicates

	# dropping duplicates
	users.drop_duplicates(keep='first').shape #inplace=False by default
	users.drop_duplicates(keep='last').shape
	users.drop_duplicates(keep=False).shape

	users.duplicated(subset=['age','zip_code']).sum()
	users.drop_duplicates(subset=['age','zip_code']).shape
	
# 34 Settingwithcopywarning and using copy()
	movies = pd.read_csv('http://bit.ly/imdbratings')
	movies.content_rating.isnull().sum() # Counting Null
	movies[movies.content_rating.isnull()] # Displaying Null
	movies.content_rating.value_counts()
	movies[movies.content_rating=='NOT RATED'].content_rating

	# Example 1: NOT RATED to Nan
	import numpy as np
	movies[movies.content_rating=='NOT RATED'].content_rating = np.nan # Warning: SettingWithCopyWarning
	movies.content_rating.isnull().sum() # Remaining 3 counts; above code didn't work
	# Correction
	movies.loc[movies.content_rating=='NOT RATED', 'content_rating'] = np.nan
	movies.content_rating.isnull().sum() # Updated to 68

	# Example 2 w/ Copy
	top_movies = movies.loc[movies.star_rating >=9, :]
	top_movies
	top_movies.loc[0,'duration'] = 150 # Using Loc and still has warning!
	top_movies # 150 was updated, however; Reasons of the warning being updating 'top_movies', a copy of movies
	#Correction
	top_movies = movies.loc[movies.star_rating >=9, :].copy() #.copy()
	top_movies.loc[0,'duration'] = 150
	movies.head() # original movies not updated # Original movies not updated
	
# 35 Display Options
	pd.get_option('display.max_rows')
	pd.set_option('display.max_rows', 200) #pd.set_option('display.max_rows', None)
	pd.reset_option('display.max_rows')

	pd.get_option('display.max_columns')

	pd.get_option('display.max_colwidth')
	pd.set_option('display.max_colwidth',1000)

	# Decimals
	pd.get_option('display.precision')
	pd.set_option('display.precision',2)

	drinks['x'] = drinks.wine_servings * 1000
	drinks['y'] = drinks.total_litres_of_pure_alcohol * 1000
	pd.set_option('display.float_format','{:,}'.format)
	drinks.head() # Only affected y cuz y is float while x is int
	drinks.dtypes

	# Bonus
	pd.describe_option()
	pd.describe_option('excel')
	pd.reset_option('all') #Warning is ok and can be ignored
	
# 36 Creating DataFrame
	# By Dictionary
	pd.DataFrame({'id':[100,101,102], 'color':['red','blue','red']})
	pd.DataFrame({'id':[100,101,102], 'color':['red','blue','red']}, columns=['id','color'])
	df = pd.DataFrame({'id':[100,101,102], 'color':['red','blue','red']}, columns=['id','color'], index=['a','b','c'])

	# By list of list
	pd.DataFrame([[100,'red'],[101,'blue'],[102,'red']], columns=['id','color'])

	# By Numpy
	import numpy as np
	arr = np.random.rand(4,2)
	arr
	pd.DataFrame(arr, columns=['one','two'])

	# By Arrange
	pd.DataFrame({'student': np.arange(100,110,1), 'test': np.random.randint(60,101,10)})
	pd.DataFrame({'student': np.arange(100,110,1), 'test': np.random.randint(60,101,10)}).set_index('student')

	# Combining Series and DataFrame w/ concat
	s = pd.Series(['round','square'], index=['c','b'], name='shape')
	pd.concat([df,s],axis=1) # aligned by the index
	
# 37 apply, map, and applymap
	# .map
	train['Sex_num'] = train.Sex.map({'female':0,'male':1})
	train.loc[0:4,['Sex','Sex_num']]

	# .apply
		# as Series Method
			train['Name_Length'] = train.Name.apply(len) #pass on the name of the function
			train.loc[0:4,['Name','Name_Length']]

			# with numpy
			import numpy as np
			train['Fare_ceil'] = train.Fare.apply(np.ceil)
			train.loc[0:4, ['Fare','Fare_ceil']]

			# with a function
			train.Name.str.split(',').head()
			def get_element(my_list, position):
				return my_list[position]
			train.Name.str.split(',').apply(get_element, position=0).head()

			# with lambda
			train.Name.str.split(',').apply(lambda x : x[0]).head()

		# as DataFrame
			drinks = pd.read_csv('http://bit.ly/drinksbycountry')
			drinks.head()

			drinks.loc[:,'beer_servings' : 'wine_servings'].apply(max, axis=0) #apply max in the down directions
			drinks.loc[:,'beer_servings' : 'wine_servings'].apply(np.argmax, axis=1) #apply max in the left to right directions

	# .applymap
	drinks.loc[:,'beer_servings' : 'wine_servings'].applymap(float)
	drinks.loc[:,'beer_servings' : 'wine_servings'] = drinks.loc[:,'beer_servings' : 'wine_servings'].applymap(float)

# 38 Multiindex
	stocks = pd.read_csv('http://bit.ly/smallstocks')
	stocks.index # RangeIndex(start=0, stop=9, step=1)

	stocks.groupby('Symbol').Close.mean()
	ser = stocks.groupby(['Symbol','Date']).Close.mean()
	ser.index # ser.index

	ser.unstack()

	# Pivot Table
	df=stocks.pivot_table(values='Close',index='Symbol',columns='Date') # By Default, pivot_table is by mean
	df

	# Selecting from a MultiIndex Series
	ser
	ser.loc['AAPL']
	ser.loc['AAPL','2016-10-03'] # if dataframe, df.loc['AAPL','2016-10-03']
	ser.loc[:,'2016-10-03'] # if dataframe, df.loc[:,'2016-10-03']

	# selecting from a multiIndex DataFrame
	stocks.set_index(['Symbol','Date'], inplace=True)
	stocks.sort_index(inplace=True)
	stocks.loc['AAPL']
	stocks.loc[('AAPL','2016-10-03'),:]
	stocks.loc[('AAPL','2016-10-03'),'Close']
	stocks.loc[(['AAPL','MSFT'],'2016-10-03'),:]
	stocks.loc[(['AAPL','MSFT'],'2016-10-03'),'Close']
	stocks.loc[('MSFT',['2016-10-03','2016-10-04']),'Close']
	stocks.loc[(slice(None),['2016-10-03','2016-10-04']),:] # slice(None) means selecting all outter indices

	# Merger to Concat
	volume = pd.read_csv('http://bit.ly/smallstocks', usecols=[0,2,3], index_col=['Symbol','Date'])
	volume.sort_index()
	close = pd.read_csv('http://bit.ly/smallstocks', usecols=[0,1,3], index_col=['Symbol','Date'])
	close.sort_index()
	both = pd.merge(close, volume, left_index=True, right_index=True)
	both
	
# 38 Merge
	# Movie
	movie_cols = ['movie_id', 'title']
	movies = pd.read_table('data/u.item', sep='|', header=None, names=movie_cols, usecols=[0,1])
	movies.head()
	movies.shape # (1682, 2)
	movies.movie_id_unique() # array([   1,    2,    3, ..., 1680, 1681, 1682], dtype=int64)

	# Ratings
	rating_cols = ['user_id', 'movie_id','rating','timestamp']
	ratings = pd.read_table('data/u.data', sep='\t', header=None, names=rating_cols)
	ratings.head()
	ratings.shape # (100000, 4)
	ratings.movie_id.nunique() # Number of unique
	ratings.loc[ratings.movie_id == 1, :].head()

	# Merging Ratings to Movies
	movies.columns 
		# Index(['movie_id', 'title'], dtype='object')
	ratings.columns 
		# Index(['user_id', 'movie_id', 'rating', 'timestamp'], dtype='object')
	movie_ratings = pd.merge(movies, ratings)
	movie_ratings.columns # By default, Matching columns with the same name
		# Index(['movie_id', 'title', 'user_id', 'rating', 'timestamp'], dtype='object')
		
	movie_ratings.shape # 100000 rows because ratings has 100000 rows; ratings.shape (100000, 4)
		# (100000, 5)
		
	print(movies.shape) # (1682, 2)
	print(ratings.shape) # (100000, 4)
	print(movie_ratings.shape) # (100000, 5)

	# What if the columns you want to join on don't have the same name?
	movies.columns = ['m_id', 'title']
	movies.columns
	ratings.columns
	pd.merge(movies, ratings, left_on='m_id', right_on='movie_id').head()

	# what if you want to join on one index?
	movies = movies.set_index('m_id')
	movies.head()

	pd.merge(movies, ratings, left_index=True, right_on='movie_id')

	# Two Indexes
	ratings = ratings.set_index('movie_id')
	ratings.head()
	pd.merge(movies, ratings, left_index=True, right_index=True).head()

	# Four Types of Joins
	A = pd.DataFrame({'color' : ['green','yellow','red'], 'num' : [1,2,3]})
	A

	B = pd.DataFrame({'color' : ['green','yellow','pink'], 'size' :['S','M','L']})
	B

	# Inner Join
	pd.merge(A, B, how='inner')

	# Outer Join
	pd.merge(A,B,how='outer')

	# Left Join
	pd.merge(A,B,how='left')

	# Right Join
	pd.merge(A,B,how='right')
	
# 40
	# Version
	pd.__version__
	
	# Create a datetime column from a DataFrame
		df = pd.DataFrame([[12,25,2017,10],[1,15,2018,11]],columns=['month','day','year','hour'])
		df

		pd.to_datetime(df)
		# create a datetime column from a subset of columns
		pd.to_datetime(df[['month','day','year']])
		# Application of this - overwrite the index
		df.index = pd.to_datetime(df[['month','day','year']])
	
	# Create a category column during file reading
		drinks = pd.read_csv('http://bit.ly/drinksbycountry')
		drinks.head()

		drinks.dtypes

		# old way
		drinks['continent'] = drinks.continent.astype('category')
		drinks.dtypes

		# new way
		drinks = pd.read_csv('http://bit.ly/drinksbycountry',dtype={'continent':'category'})
		drinks.dtypes
		
	# Convert the data type of multiple columns at once
		drinks = pd.read_csv('http://bit.ly/drinksbycountry')
		drinks.dtypes

		# old way to convert data types (one at a time)
		drinks['beer_servings'] = drinks.beer_servings.astype('float')
		drinks['spirit_servings'] = drinks.spirit_servings.astype('float')
		drinks.dtypes

		# new way
		drinks = pd.read_csv('http://bit.ly/drinksbycountry')
		drinks = drinks.astype({'beer_servings':'float','spirit_servings':'float'})
		drinks.dtypes
		
	# Apple Multiple Aggregations on a Series or Dataframe
		drinks.groupby('continent').beer_servings.mean()

		# old multiple aggreation
		drinks.groupby('continent').beer_servings.agg(['mean','min','max'])

		# new: apply the same aggregation to a Series
		drinks.beer_servings.agg(['mean','min','max'])

		# new: apply the same aggregation to a dataframe
		drinks.agg(['mean','min','max'])

		drinks.describe() # less flexible, compared to .agg; in .agg, you can define the measures
		
# 41
	# ix replacement
		# ix Example 1
		drinks.ix['angola',1]

			# alternativ: use loc
			drinks.loc['Angola',drinks.columns[1]]
			
			# alternative: use iloc
			drinks.iloc[drinks.index.get_loc('Angola'),1]
		
		# ix Example 2: accesses by label or position
		drinks.ix[4,'spirit_servings']
			
			# alternative: use loc
			drinks.loc[drinks.index[4],'spirit_servings']
			
			# alternative: use iloc
			drinks.iloc[4, drinks.columns.get_loc('spirit_servings')]
		
	# aliases have been added for isnull and notnull (isna)
		ufo = pd.read_csv('http://bit.ly/uforeports')
		ufo.isnull().head()
		# check which values are not missing
		ufo.notnull().head()
		# drop rows with missing values
		ufo.dropna().head()
		# fill in missing values
		ufo.fillna(value='UNKOWN').head()
		# new alias for isnull
		ufo.isna().head()
		
	# drop now accepts "index" and "columns" keywords
		ufo = pd.read_csv('http://bit.ly/uforeports')
		ufo.head()

		# Rows
			# old way to drop rows
			ufo.drop([0,1],axis=0).head()
			ufo.drop([0,1],axis='index').head()

			# new way to drop rows
			ufo.drop(index=[0,1]).head()
			
		# Columns
			# old way to drop columns
			ufo.drop(['City','State'],axis=1).head()
			ufo.drop(['City','State'],axis='columns').head()
			
			# new way to drop columns: specify columns
			ufo.drop(columns=['City','State']).head()
			
		# rename and reindex now accept "axis" keyword
		# old way to rename columns: specify columns
			ufo.rename(columns = {'City':'CITY','State':'STATE'}).head()
		# new way to rename columns: specify mapper and axis
			ufo.rename({'City':'CITY','State':'STATE'}, axis='columns').head()
		# note: mapper can be a function
			ufo.rename(str.upper, axis='columns').head()
		
	# Ordered Categories must be specified indepedent of the data

		df = pd.DataFrame({'ID': [100,101,102,103],'quality':['good','very good','good','excellent']})
		df

		# old way to create an ordered category
		df.quality.astype('category',categories=['good','very good','excellent'], ordered=True)

		from pandas.api.types import CategoricalDtype
		quality_cat = CategoricalDtype(['good','very good','excellent'], ordered=True)
		df['quality'] = df.quality.astype(quality_cat)
		df.quality