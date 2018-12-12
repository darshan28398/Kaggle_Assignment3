import pandas as pd                 # for working with data in Python
import numpy as np
import matplotlib.pyplot as plt     # for visualization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("1 \n")


print("Train data shape:", train.shape)
print("Test data shape:", test.shape)



print("2 \n")


print(train.head())

#to do some plotting
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)



print("3 \n")

print (train.SalePrice.describe())

print("4 \n")


print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

print("5 \n")

target = np.log(train.SalePrice)
print ("\n Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


#Numeric Features
print("6 \n")

numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)

print("7 \n")

corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

print("8 \n")

"""
#to get the unique values that a particular column has.
#train.OverallQual.unique()
print(train.OverallQual.unique())
"""
print("9 \n")
"""
#investigate the relationship between OverallQual and SalePrice.
#We set index='OverallQual' and values='SalePrice'. We chose to look at the median here.
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(quality_pivot)
"""
print("10 \n")
"""
#visualize this pivot table more easily, we can create a bar plot
#Notice that the median sales price strictly increases as Overall Quality increases.
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
"""
print("11 \n")
"""
#to generate some scatter plots and visualize the relationship between the Ground Living Area(GrLivArea) and SalePrice
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()
"""
print("12 \n")

# do the same for GarageArea.
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


print("13 \n")

train = train[train['GarageArea'] < 1200]

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600)     # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()





print("14 \n")

# create a DataFrame to view the top null columns and return the counts of the null values in each column
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
#nulls
print(nulls)

print("15 \n")
"""
#to return a list of the unique values
print ("Unique values are:", train.MiscFeature.unique())
"""



print("16 \n")

categoricals = train.select_dtypes(exclude=[np.number])
print(categoricals.describe())




print("17 \n")




#Eg:
print ("Original: \n")
print (train.Street.value_counts(), "\n")

print("18 \n")


train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)

print ('Encoded: \n')
print (train.enc_street.value_counts())  # Pave and Grvl values converted into 1 and 0

print("19 \n")

condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

print("20 \n")

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

print("21 \n")

print(sum(data.isnull().sum() != 0))

print("22 \n")



# separate the features and the target variable for modeling.
# We will assign the features to X and the target variable(Sales Price)to y.

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
# exclude ID from features since Id is just an index with no relationship to SalePrice.


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)





lr = linear_model.LinearRegression()


model = lr.fit(X_train, y_train)

print("23 \n")


print("R^2 is: \n", model.score(X_test, y_test))

predictions = model.predict(X_test)

print("24 \n")
print('RMSE is: \n', mean_squared_error(y_test, predictions))

print("25 \n")

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b')  # alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()




print("26 \n")

for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()



print("27 \n")
print("R^2 is: \n", model.score(X_test, y_test))




submission = pd.DataFrame()
submission['Id'] = test.Id


feats = test.select_dtypes(
    include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = model.predict(feats)


final_predictions = np.exp(predictions)

print("28 \n")


print("Original predictions are: \n", predictions[:10], "\n")
print("Final predictions are: \n", final_predictions[:10])

print("29 \n")

submission['SalePrice'] = final_predictions

print(submission.head())


submission.to_csv('submission1.csv', index=False)


print("\n Finish")
