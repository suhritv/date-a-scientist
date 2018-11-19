import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Reading the CSV File. Had to give the entire path for it to work
df=pd.read_csv(r"/Users/chug1985/Desktop/Python/DataScienceLibraries/Assignmnent/capstone_starter/profiles.csv")
#print(df.sign.value_counts())
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)
smokes_mapping={"no":1,"sometimes":2,"when drinking":3,"yes":4,"trying to quit":5}
df["smokes_code"]=df.smokes.map(smokes_mapping)
drugs_mapping={"never":1, "sometimes":2, "often":3}
df["drugs_code"]=df.drugs.map(drugs_mapping)

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
#print(all_essays)

# Finding the length of essay column

df["essay_len"] = all_essays.apply(lambda x: len(x))

df["all_essays"]=all_essays

# Finding the total words per row and the average word length
df["totalwords"] = [len(x.split()) for x in df["all_essays"].tolist()]
df["average_word_length"]=df["essay_len"]/df["totalwords"]

#print(df.head())

# Finding the frequency of I in the essay column
df["frequency"]=df.all_essays.str.contains(r'I').sum()

# Scatter plot Age Vs Total words. Trying to find if age makes a difference in a person typing more words in profiles?

from sklearn.linear_model import LinearRegression
regr=LinearRegression()
X=df["age"]
X=X.values.reshape(-1,1)
y=df["totalwords"]
regr.fit(X,y)
y_predict=regr.predict(X)
plt.plot(X,y_predict)
plt.scatter(X,y)
plt.xlabel("Age of a Person")
plt.ylabel("Words on their profile")
plt.title("Age as an Online Dating Criteria?")
plt.xlim(18,75)
plt.show()

# Trying to find a correlation between income and essay length. Trying to answer the question if income makes a person
# more of less interested in creating a profile?

from sklearn.linear_model import LinearRegression
regr=LinearRegression()
X=df["income"]
X=X.values.reshape(-1,1)
Y=df["essay_len"]
regr.fit(X,y)
y_predict=regr.predict(X)
plt.plot(X,y_predict)
plt.scatter(X,y)
plt.xlabel("Income of a person")
plt.ylabel("Length of descriptions on their profile")
plt.title("Income of a person determining factor?")
plt.xlim(20000,100000)
plt.show()
