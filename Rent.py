import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

sns.set()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df1 = pd.read_csv('Population_growth.csv')
df2 = pd.read_csv('Economic_Growth.csv')
df3 = pd.read_csv('abbreviations.csv')

data = pd.merge(df1, df2, on = 'state', how = 'left')




data.drop(columns = ['growthSince2010', 'gDPGrowth2020To21', 'gDPGrowth2019To20', 'gDPGrowth2018To19'], inplace = True)

new_columns = {
    'state' : 'State',
    'growthRate' : 'Pop_Growth',
    'pop2023' : 'Population',
    'gDPGrowth2021To22' : 'GDP_Growth'
}

data.rename(columns = new_columns, inplace = True)

data.at[40,'GDP_Growth'] = 0.73
data.at[17, 'GDP_Growth'] = 2.5
data.at[20, 'GDP_Growth'] = 1.7
data = pd.merge(data, df3, on = 'State', how = 'left' )
data['abbreviation'] = data['abbreviation'].astype('str')


#Making a plot of Population Growth vs Economic Growth
max_population = max(data.Population)
bubble_sizes = [pop / max_population * 800 for pop in data.Population]
#plt.figure(figsize = (20,10))
colors = np.random.rand(len(data), 3)
plt.scatter(x = data.Pop_Growth, y = data.GDP_Growth, s = bubble_sizes, alpha = 0.5, c = colors)
plt.title('Economic Growth vs Population Growth')
plt.xlabel('Population Growth')
plt.ylabel('Economic Growth')

x,y = data['Pop_Growth'], data['GDP_Growth']
for i, txt in enumerate(data['abbreviation']):
    plt.annotate(txt, (x[i], y[i]))
data['Econ&Pop'] = data['GDP_Growth'] + data['Pop_Growth']
#print(data.nlargest(10, 'Econ&Pop').State)

plt.show()
plt.clf()

#https://worldpopulationreview.com/state-rankings/fastest-growing-states

# Making a plot of Average residential price and average rental income
df4 = pd.read_csv('Res_Price.csv')
df5 = pd.read_csv('Rent_Income.csv')

df4 = df4.drop(columns = ['medianHomePriceYearOnYearGrowth', 'dataUpdated'])
df5 = df5.drop(columns = ['averageRentZillow'])

data = pd.merge(data, df4, on = 'State', how = 'left' )
data = pd.merge(data, df5, on = 'State', how = 'left' )

data.at[4,'MedianHomePrice'] = 296096
data.at[5, 'MedianHomePrice'] = 390000
data.at[26, 'MedianHomePrice'] = 249529
null_values = data.isna()

data['1% Rule'] = data['MedianRent'] / data['MedianHomePrice']

data.to_csv('data.csv', index = False)


plt.figure(figsize = (20,10))
colors = np.random.rand(len(data),3)
plt.scatter(x = data.MedianHomePrice, y = data.MedianRent, s = data['1% Rule'] * 10000, alpha = 0.5, c = colors)
plt.title('Median Home prices vs Median Rent')
plt.xlabel('Median Home Price')
plt.ylabel('Median Rent / Median Rental Income')
x,y = data['MedianHomePrice'], data['MedianRent']
for i, txt in enumerate(data['abbreviation']):
    plt.annotate(txt, (x[i], y[i]))

#print((data.nlargest(15, '1% Rule')).State)
plt.show()
plt.clf()

# Plot of Crime Rate vs Cost of Living - Livable factor
df6 = pd.read_csv('Crime_Rate.csv')
df7 = pd.read_csv('COL.csv')

df6 = df6.drop(columns = ['pop2020', 'reported', 'violent', 'nonViolent', 'violentRate', 'nonViolentRate'])
df7 = df7.drop(columns  = ['GroceryCostsIndex', 'HealthCostsIndex', 'HousingCostsIndex','MiscCostsIndex' , 'TransportationCostsIndex', 'UtilityCostsIndex'])

data = pd.merge(data, df6, on = 'State', how = 'left')
data = pd.merge(data, df7, on = 'State', how = 'left')
data = data.rename(columns = {'2023' : 'COL', 'rate' : 'CrimeRate_100k'})

data.to_csv('data.csv', index = False)


plt.figure(figsize = (20,10))
colors = np.random.rand(len(data),3)
plt.scatter(x = data.COL, y = data.CrimeRate_100k, s = bubble_sizes, alpha = 0.5, c = colors)
plt.title('Cost of Living vs Crime Rate')
plt.xlabel('Cost of Living Index (Based on Purchasing Power of average 100 US dollars)')
plt.ylabel('Crime Rate per 100k People')
x,y = data['COL'], data['CrimeRate_100k']
for i, txt in enumerate(data['abbreviation']):
    plt.annotate(txt, (x[i], y[i]))
plt.show()
plt.clf()

# - Finalizing the top 10 in this category

data['COL&Crime'] = data['COL'] + data['CrimeRate_100k']

#print((data.nsmallest(15, 'COL&Crime')).State)


# Vacancy Rate source = https://www.census.gov/housing/hvs/data/rates.html (fist link)
df8 = pd.read_csv('Rent_Vacancy.csv')

df8['State'] = df8['State'].str.replace('.','')

df8 = df8.drop(columns = [ 'Margin of Error1', 'Margin of Error1.1', 'Margin of Error1.2', 'Margin of Error1.3'])
df8['Average_Vacancy'] = df8[['First         Quarter       2022', 'Second         Quarter       2022', 'Third         Quarter       2022', 'Fourth         Quarter       2022']].mean(axis = 1)
df8 = df8.sort_values(by='Average_Vacancy', ascending=False)

sns.barplot(x = 'Average_Vacancy', y = 'State', data = df8)
plt.xlabel('Vacancy Rate 2022')
plt.ylabel('State')
plt.title('Average Vacancy by State')
plt.show()
plt.clf()


#Annual Landlord Insurance - in order to account for natural disasters 
df9 = pd.read_csv('Landlord_Insurance_Annual.csv')
df9 = df9.sort_values(by = 'Landlord_Insurance')




sns.barplot(x = 'Landlord_Insurance', y = 'State', data = df9)
plt.xlabel('Average Annual Landlord Insurance')
plt.ylabel('State')
plt.title('Average Landlord Insurance by State')
plt.show()
plt.clf()

#Annual Property Taxes 
df10 = pd.read_csv('Property Tax Rate.csv')
data = pd.merge(data, df10, on = 'State', how = 'left' )
data['Median Property Taxes'] = data.apply(lambda row: row['MedianHomePrice'] * row['propertyTaxRate'], axis =1 )


sns.barplot(x = 'Median Property Taxes', y = 'State', data = data)
plt.xlabel('Median Property Taxes')
plt.ylabel('State')
plt.title('Median Property Taxes')
plt.show()
plt.clf()


#Finding out the Best State
data = pd.merge(data, df8, on = 'State', how = 'left' )
data = pd.merge(data, df9, on = 'State', how = 'left' )


Ranking = pd.DataFrame()
Ranking['State'] = data.State
Ranking['PropertyTaxes_Rank'] = data['Median Property Taxes'].rank(ascending=True)
Ranking['Landlord_Insurance_Rank'] = data['Landlord_Insurance'].rank(ascending=True)
Ranking['Average_Vacancy_Rank'] = data['Average_Vacancy'].rank(ascending=True)
Ranking['COL&Crime_Rank'] = data['COL&Crime'].rank(ascending=True)
Ranking['1% Rule_Rank'] = data['1% Rule'].rank(ascending=False) 
Ranking['GDP+Pop_Growth_Rank'] = data['Econ&Pop'].rank(ascending=False)  

Ranking['Final Rank'] = Ranking['PropertyTaxes_Rank'] + Ranking['Landlord_Insurance_Rank'] + Ranking['Average_Vacancy_Rank'] + Ranking['COL&Crime_Rank'] + Ranking['1% Rule_Rank'] + Ranking['GDP+Pop_Growth_Rank']
Ranking = Ranking.sort_values(by='Final Rank')


Ranking.to_csv('Ranking.csv', index = False)
print(Ranking.State)