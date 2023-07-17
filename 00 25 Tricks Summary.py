# ### 1. Show Installed Versions
# ### 2. Create an example DataFrame
# ### 3. Rename Columns
# ### 4. Reverse row order
# ### 5. Reverse column order
# ### 6. select columns by data type
# ### 7. Convert strings to numbers
# ### 8. Reduce DataFrame Size
# ### 9. Build a DataFrame from multiple files (row-wise)
# ### 10. Build a DataFrame from multiple files (column-wise)
# ### 11. Create a DataFrame from the clipboard
# ### 12. Split a DataFrame into two random subsets
# ### 13. Filter a DataFrame by multiple categories
# ### 14. Filter a DataFrame by largest categories
# ### 15. Handle missing values
# ### 16. Split a string into multiple columns
# ### 17. Expand a Series of lists into a DataFrame
# ### 18. Aggregate by multiple functions
# ### 19. Combine the output of an aggregation with a DataFrame
# ### 20. Select a slice of rows and columns
# ### 21. Reshape a MultiIndexed Series (.unstack())
# ### 22. Create a pivot table (df.pivot_table)
# ### 23. Convert continupous data into categorical data (.cut)
# ### 24 Change display options
# ### 25 Style a DataFrame (by columns)

drinks = pd.read_csv('http://bit.ly/drinksbycountry')
movies = pd.read_csv('http://bit.ly/imdbratings')
orders = pd.read_csv('http://bit.ly/chiporders', sep='\t')
orders['item_price'] = orders.item_price.str.replace('$', '').astype('float')
stocks = pd.read_csv('http://bit.ly/smallstocks', parse_dates=['Date'])
titanic = pd.read_csv('http://bit.ly/kaggletrain')
ufo = pd.read_csv('http://bit.ly/uforeports', parse_dates=['Time'])


# ### 1. Show Installed Versions
	pd.__version__
	pd.show_versions()

# ### 2. Create an example DataFrame

	df = pd.DataFrame({'col one':[100, 200], 'col two':[300, 400]})
	df

	pd.DataFrame(np.random.rand(4, 8))

	pd.DataFrame(np.random.rand(4, 8), columns=list('abcdefgh'))

# ### 3. Rename Columns

	df = df.rename({'col one':'col_one', 'col two':'col_two'}, axis='columns')
	df.columns = ['col_one', 'col_two'] #Renaming all columns
	df.columns = df.columns.str.replace(' ', '_')
	df.add_prefix('X_')
	df.add_suffix('_Y')

# ### 4.Reverse row order

	drinks.loc[::-1].head()
	drinks.loc[::-1].reset_index(drop=True).head()

# ### 5. Reverse column order

	drinks.loc[:,::-1].head()

# ### 6. select columns by data type

	drinks.dtypes
	drinks.select_dtypes(include='number').head()
	drinks.select_dtypes(include='object').head()
	drinks.select_dtypes(include=['number', 'object', 'category', 'datetime']).head()
	drinks.select_dtypes(exclude='number').head()


# ### 7. Convert strings to numbers

	df = pd.DataFrame({'col_one':['1.1', '2.2', '3.3'],
					   'col_two':['4.4', '5.5', '6.6'],
					   'col_three':['7.7', '8.8', '-']})
	df.dtypes #object = string
	df.astype({'col_one':'float', 'col_two':'float'}).dtypes # col_three     object
	pd.to_numeric(df.col_three, errors='coerce') # "-" becomes Nan
	pd.to_numeric(df.col_three, errors='coerce').fillna(0) #"-" becomes 0
	df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
	df
	df.dtypes

# ### 8. Reduce DataFrame Size

	drinks.info(memory_usage='deep')
	cols = ['beer_servings', 'continent'] # Only Reading the columns we need
	small_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols)
	small_drinks.info(memory_usage='deep')

	dtypes = {'continent':'category'} # Use category datatype
	smaller_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols, dtype=dtypes)
	smaller_drinks.info(memory_usage='deep')


# ### 9. Build a DataFrame from multiple files (row-wise)

	from glob import glob
	stock_files = sorted(glob('data/stocks*.csv'))
	stock_files
	pd.concat((pd.read_csv(file) for file in stock_files))

	# duplicated value in the index
	pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)

# ### 10. Build a DataFrame from multiple files (column-wise)

	pd.read_csv('data/drinks1.csv').head()
	pd.read_csv('data/drinks2.csv').head()
	drink_files = sorted(glob('data/drinks*.csv'))
	pd.concat((pd.read_csv(file) for file in drink_files), axis='columns').head()


# ### 11. Create a DataFrame from the clipboard

	df = pd.read_clipboard() # from Excel
	df
	#Just like the read_csv() function, read_clipboard() automatically detects the correct data type for each column:
	df.dtypes
	df = pd.read_clipboard() # https://youtu.be/RlIiVeig3hc?t=685
	df 
	df.index

# ### 12. Split a DataFrame into two random subsets

	len(movies)
	movies_1 = movies.sample(frac=0.75, random_state=1234)
	movies_2 = movies.drop(movies_1.index)
	len(movies_1) + len(movies_2)
	movies_1.index.sort_values()
	movies_2.index.sort_values()


# ### 13. Filter a DataFrame by multiple categories

	movies.head()
	movies.genre.unique()

	# Long way
	movies[(movies.genre == 'Action') |
		   (movies.genre == 'Drama') |
		   (movies.genre == 'Western')].head()
	# Alternative
	movies[movies.genre.isin(['Action', 'Drama', 'Western'])].head()

	movies[~movies.genre.isin(['Action', 'Drama', 'Western'])].head() # tilda is the NOT Operator


# ### 14. Filter a DataFrame by largest categories

	counts = movies.genre.value_counts()
	counts
	counts.nlargest(3)
	counts.nlargest(3).index
	movies[movies.genre.isin(counts.nlargest(3).index)].head()


# ### 15. Handle missing values

	ufo.head()
	ufo.isna().sum()
	ufo.isna().mean() # % of missing value in the columns 
	ufo.dropna(axis='columns').head()
	ufo.dropna(thresh=len(ufo)*0.9, axis='columns').head()
	# 1. len(ufo) returns the total number of rows, 
	# 2. then we multiply that by 0.9 to tell pandas to only keep columns in which at least 90% of the values are not missing.

# ### 16. Split a string into multiple columns

	df = pd.DataFrame({'name':['John Arthur Doe', 'Jane Ann Smith'],
					   'location':['Los Angeles, CA', 'Washington, DC']})

	df.name.str.split(' ', expand=True)
	df[['first', 'middle', 'last']] = df.name.str.split(' ', expand=True)

	df.location.str.split(', ', expand=True)
	df['city'] = df.location.str.split(', ', expand=True)[0]

# ### 17. Expand a Series of lists into a DataFrame

	df = pd.DataFrame({
	'col_one':['a', 'b', 'c'], 
	'col_two':[[10, 40], [20, 50], [30, 60]]
	})

	df_new = df.col_two.apply(pd.Series)
	df_new

	pd.concat([df, df_new], axis='columns')

# ### 18. Aggregate by multiple functions

	orders[orders.order_id == 1].item_price.sum()
	orders.groupby('order_id').item_price.sum().head()
	orders.groupby('order_id').item_price.agg(['sum', 'count']).head()


# ### 19. Combine the output of an aggregation with a DataFrame

	orders.groupby('order_id').item_price.sum().head()
	len(orders.groupby('order_id').item_price.sum()) 
		# 1834 output is less the input
	len(orders.item_price)
		# 4622
	total_price = orders.groupby('order_id').item_price.transform('sum')
	len(total_price)
		# 4622
	orders['total_price'] = total_price
	orders['percent_of_total'] = orders.item_price / orders.total_price

# ### 20. Select a slice of rows and columns

	titanic.head()
	titanic.describe()
	titanic.describe().loc['min':'max']
	titanic.describe().loc['min':'max', 'Pclass':'Parch']


# ### 21. Reshape a MultiIndexed Series (.unstack())

	titanic.Survived.mean()
	titanic.groupby('Sex').Survived.mean()
	titanic.groupby(['Sex', 'Pclass']).Survived.mean()
	titanic.groupby(['Sex', 'Pclass']).Survived.mean().unstack()


# ### 22. Create a pivot table (df.pivot_table)

	titanic.pivot_table(
		index='Sex', 
		columns='Pclass', 
		values='Survived', 
		aggfunc='mean')

	titanic.pivot_table(
		index='Sex', 
		columns='Pclass', 
		values='Survived', 
		aggfunc='mean',
		margins=True) # margins to show total

	titanic.pivot_table(
		index='Sex', 
		columns='Pclass', 
		values='Survived', 
		aggfunc='count',
		margins=True)


# ### 23. Convert continupous data into categorical data (.cut)

	titanic.Age.head(10)

	pd.cut(
		titanic.Age, 
		bins=[0, 18, 25, 99], 
		labels=['child', 'young adult', 'adult']
	).head(10)
		# Notice the dtype is now dtype: category

# ### 24 Change display options

	titanic.head()
	pd.set_option('display.float_format', '{:.2f}'.format)
	titanic.head()
		# No change to the underlying data; only the display
	pd.reset_option('display.float_format')


# ### 25 Style a DataFrame (by columns)

	stocks
	format_dict = {'Date':'{:%m/%d/%y}', 'Close':'${:.2f}', 'Volume':'{:,}'}

	(stocks.style.format(format_dict)
	 .hide_index()
	 .highlight_min('Close', color='red')
	 .highlight_max('Close', color='lightgreen')
	)

	(stocks.style.format(format_dict)
	 .hide_index()
	 .background_gradient(subset='Volume', cmap='Blues')
	)

	(stocks.style.format(format_dict)
	 .hide_index()
	 .bar('Volume', color='lightblue', align='zero')
	 .set_caption('Stock Prices from October 2016')
	)


	# ### Bonus
	# 

	# In[ ]:


	import pandas_profiling # Need to install
	pandas_profiling.ProfileReport(titanic)

