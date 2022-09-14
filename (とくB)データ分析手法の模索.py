from multiprocessing import Value
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


teams = pd.read_excel("C:\\Users\\MyPC\\Desktop\\python\\filemmm.xlsx",parse_dates=['施策'],index_col='施策')
teams = teams.replace(np.nan,0)
teams = teams.replace('-',0)
teams = teams.rename(columns={"ファインド":"Find"})
train, test = train_test_split(teams, test_size=0.01, random_state=50)
predictors = ["YDA","GDN","Find","Trueview","Facebook","Twitter","Criteo","LINE広告","Logicad","docomoAD","nend"]
# predictors = ["YDA","GDN","ファインド","Trueview","Facebook","Twitter","LINE広告","Logicad","docomoAD","nend"]
target = 'taget'


X = train[predictors].copy()
y = train[[target]].copy()
X = X.loc[:,(X != 0).any(axis=0)]
print(X)
# X["ファインド"].to_excel("C:\\Users\\MyPC\\Desktop\\python\\test.xlsx")
x_mean = X.mean()
x_std = X.std()
X = (X - x_mean) / x_std
alpha = 0.1
test_X = test[predictors]
test_X = test_X.loc[:,(test_X != 0).any(axis=0)]
test_X = (test_X - x_mean) / x_std
y_test = test[target]

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=alpha)
ridge.fit(X, y)
pre = ridge.predict(test_X)
print(r2_score(y_test, pre))

# print(ridge.coef_)
weights = pd.Series(
        ridge.coef_[0],
        index=test_X.columns
    )


data = {
    'Media' : list(test_X.columns),
    'Value' : ridge.coef_[0].tolist()
}
num = pd.DataFrame(
    data = data
    
)
# print(num)
reg = ridge.intercept_
#draw result
# f, ax = plt.subplots(figsize=(8, 5))
sns.set_color_codes("pastel")

sns.barplot(x='Value', y='Media', orient = 'h',data = num,
            order=num.sort_values('Value',ascending = False).Media)
plt.show()
scores = cross_val_score(ridge, test_X, y_test,scoring='r2',cv=3)

print(scores.mean(),scores.std())
sklearn_predictions = ridge.predict(test_X)


#draw chart for each media

# y_ = teams[[target]]
# print(X.columns[1])
# row = 4
# col = int((len(X.columns))/row) +1
# fig, axes = plt.subplots(col, row)
# for i,ax in zip(range(len(X.columns)),axes.flatten()):
#     # sns.histplot(teams.iloc[:,i:i+1], ax =ax )
#     sns.scatterplot(x="taget",
#                     y = X.columns[i],
#                     data = teams,
#                      ax =ax 
#                     ).set(title=X.columns[i])
   
    
# plt.tight_layout()
    
# plt.show()
