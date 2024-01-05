import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data/data.csv')

print(type(df))
print(df.head)
print(df.describe().round())
print(df.dropna)
#print(df.head(10))


#headers = ['SEQN','age_group','RIDAGEYR','RIAGENDR','PAQ605','BMXBMI','LBXGLU','DIQ010','LBXGLT','LBXIN']
#print("headers\n", headers)
#df.columns = headers

print(df.replace("?", np.nan, inplace=True))
missing_data = df.isnull()
print(missing_data.head(5))

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


avg_BMXBMI = df['BMXBMI'].astype('float').mean(axis=0)
print("Average BMXBMI:", avg_BMXBMI)
df['BMXBMI'].replace(np.nan, avg_BMXBMI, inplace=True)


print(df['DIQ010'].value_counts().idxmax())

df = df.astype(
    {
        'age_group': 'category',
    }
)
df.age_group = df.age_group.cat.codes

print(df.dtypes)

sns.displot(df, x="BMXBMI", discrete=True)
plt.savefig("BMXBMI_displot.png")
plt.close()


df["BMXBMI"]=df["BMXBMI"].astype(int, copy=True)
bins = np.linspace(min(df["BMXBMI"]), max(df["BMXBMI"]), 4)
print(bins)
group_names = ['Low', 'Medium', 'High']
df['BMXBMI-binned'] = pd.cut(df['BMXBMI'], bins, labels=group_names, include_lowest=True )
df[['BMXBMI','BMXBMI-binned']].head(20)
df["BMXBMI-binned"].value_counts()


sns.displot(df, x="BMXBMI-binned", bins=3, discrete=True)
plt.show()
plt.savefig("BMXBMI-binned_displot.png")
plt.close()

df.columns
dummy_variable_1 = pd.get_dummies(df["age_group"])
dummy_variable_1.head()
dummy_variable_1.rename(columns={'Adult':'age_group-Adult', 'Senior':'age_group-Senior'}, inplace=True)
dummy_variable_1.head()
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("age_group", axis = 1, inplace=True)
print(df.head())


sns.relplot(data=df.sample(500), x="RIDAGEYR", y="LBXGLU")
plt.ylim(0,)
plt.savefig("LBXGLU_RIDAGEYR_regplot.png")
plt.close()

print(df[["LBXGLU", "RIDAGEYR"]].corr())

sns.catplot(
    data=df,
    x="RIAGENDR", y="BMXBMI",
    kind="violin"
)
plt.savefig("RIDAGEYR_LBXGLU_catplot.png")
plt.close()

df['PAQ605'].unique()
df_group_one = df[['PAQ605','RIAGENDR','LBXIN']].sample(n=100, random_state=1)
print(df_group_one)

df_gptest = df[['PAQ605','RIAGENDR','LBXIN']]
grouped_test1 = df_gptest.groupby(['PAQ605','RIAGENDR'],as_index=False).mean()
print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='PAQ605',columns='RIAGENDR')
print(grouped_pivot)

pearson_coef, p_value = stats.pearsonr(df['BMXBMI'], df['LBXGLU'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#ANOVA: Analysis of Variance

lm = LinearRegression()
print(lm)
X = df[['LBXIN']]
Y = df['LBXGLU']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5] 
print(lm.intercept_)
print(lm.coef_)

Z = df[['LBXGLT','RIDAGEYR','RIAGENDR','LBXIN']]
lm.fit(Z, df['LBXGLU'])
print(lm.intercept_)
print(lm.coef_)

width = 12
height = 10
plt.figure(figsize=(width, height))
sample_df = df.sample(n=500, random_state=1)
sns.regplot(x="LBXGLT", y="LBXGLU", data=sample_df)
plt.show()
plt.savefig("regplot_plot.png")
plt.close()

plt.figure(figsize=(width, height))
sample_df = df.sample(n=500, random_state=1)
sns.residplot(x="BMXBMI", y="LBXGLU", data=sample_df)
plt.show()
plt.savefig("resid_plot.png")
plt.close()


Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['LBXGLU'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for LBXGLU')
plt.xlabel('LBXGLU ( in mg)')
plt.ylabel('Proportion of LBXGLU')
plt.savefig("distplot.png")
plt.show()
plt.close()


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for LBXGLT ~ Count')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('LBXGLT ')
    plt.savefig("PlotPolly.png")
    plt.show()
    plt.close()

x = df['BMXBMI']
y = df['LBXGLT']

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'BMXBMI')
np.polyfit(x, y, 3)

pr=PolynomialFeatures(degree=2)
print(pr)
Z_pr=pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
print(pipe)
Z = Z.astype(float)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
print(ypipe[0:4])

lm.fit(X, Y)
print('The R-square is: ', lm.score(X, Y))
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

mse = mean_squared_error(df['LBXGLT'], Yhat)
print('The mean square error of LBXGLT and predicted value is: ', mse)

lm.fit(Z, df['LBXGLT'])
print('The R-square is: ', lm.score(Z, df['LBXGLT']))

Y_predict_multifit = lm.predict(Z)
print('The mean square error of LBXGLT and predicted value using multifit is: ', \
      mean_squared_error(df['LBXGLT'], Y_predict_multifit))

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

mean_squared_error(df['LBXGLT'], p(x))

new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
print(lm)
yhat=lm.predict(new_input)
yhat[0:5]
#plt.plot(new_input, yhat)
#plt.savefig("Plot.png")
#plt.show()
#plt.close()

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('LBXGLU (in mg)')
    plt.ylabel('Proportion of LBXGLU ')
    plt.savefig("DistributionPlot.png")
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('LBXGLT')
    plt.legend()

y_data = df['LBXGLU']
x_data=df.drop('LBXGLU',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lre=LinearRegression()
print( lre.fit(x_train[['RIDAGEYR']], y_train))
print(lre.score(x_test[['RIDAGEYR']], y_test))
print(lre.score(x_train[['RIDAGEYR']], y_train))

Rcross = cross_val_score(lre, x_data[['RIDAGEYR']], y_data, cv=4)
print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
-1 * cross_val_score(lre,x_data[['RIDAGEYR']], y_data,cv=4,scoring='neg_mean_squared_error')

yhat = cross_val_predict(lre,x_data[['RIDAGEYR']], y_data,cv=4)
print(yhat[0:5])

lr = LinearRegression()
lr.fit(x_train[['RIDAGEYR','LBXGLT','LBXIN','BMXBMI']], y_train)
yhat_train = lr.predict(x_train[['RIDAGEYR','LBXGLT','LBXIN','BMXBMI']])
print(yhat_train[0:5])
yhat_test = lr.predict(x_test[['RIDAGEYR','LBXGLT','LBXIN','BMXBMI']])
print(yhat_test[0:5])

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['RIDAGEYR']])
x_test_pr = pr.fit_transform(x_test[['RIDAGEYR']])
print(pr)

poly = LinearRegression()
poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)
print(yhat[0:5])
print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)
PollyPlot(x_train[['RIDAGEYR']], x_test[['RIDAGEYR']], y_train, y_test, poly,pr)

poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[['RIDAGEYR']])

    x_test_pr = pr.fit_transform(x_test[['RIDAGEYR']])

    lr.fit(x_train_pr, y_train)

    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['RIDAGEYR']])
    x_test_pr = pr.fit_transform(x_test[['RIDAGEYR']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['RIDAGEYR']], x_test[['RIDAGEYR']], y_train,y_test, poly, pr)

interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['RIDAGEYR','RIAGENDR','PAQ605','BMXBMI']])
x_test_pr=pr.fit_transform(x_test[['RIDAGEYR','RIAGENDR','PAQ605','BMXBMI']])

RigeModel=Ridge(alpha=1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)

    print(pbar.set_postfix({"Test Score": test_score, "Train Score": train_score}))

    print(Rsqu_test.append(test_score))
    print(Rsqu_train.append(train_score))


parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
print(parameters1)
RR=Ridge()
print(RR)

Grid1 = GridSearchCV(RR, parameters1,cv=4)
Grid1.fit(x_data[['RIDAGEYR','RIAGENDR','PAQ605','BMXBMI']], y_data)
BestRR=Grid1.best_estimator_
print(BestRR)
BestRR.score(x_test[['RIDAGEYR','RIAGENDR','PAQ605','BMXBMI']], y_test)

