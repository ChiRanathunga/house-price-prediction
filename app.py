import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,root_mean_squared_error


df=pd.read_csv('Housing.csv')

'''print(df.head())

df.info()
print(df.describe())
print(df.isnull().sum())
'''
#Data Preprocessing

df=pd.get_dummies(df,dtype=int,drop_first=True)

#Feature Selection
X = df.drop('price',axis = 1)

y = df['price']

#Test/Train Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Train Linear Regression
LR_model = LinearRegression()

LR_model.fit(X_train,y_train)

#Predictions
 
LR_pred =LR_model.predict(X_test)


#Prediction Visualization
plt.figure(figsize=(6,6))

plt.scatter(y_test,LR_pred,alpha=0.7)

plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],
         'r--'
         )

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title('Actual vs Predicted House Prices')

fmt = mticker.FuncFormatter(lambda x , _:f'${x:,.0f}')
plt.gca().xaxis.set_major_formatter(fmt)
plt.gca().yaxis.set_major_formatter(fmt)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()



#Evaluations
mae=mean_absolute_error(y_test,LR_pred)
print(f"Mean Absolute Error: ${mae:.2f}")

rmse = root_mean_squared_error(y_test, LR_pred)
print(f"RMSE: ${rmse:.2f}")

r2=r2_score(y_test,LR_pred)
print(f"R2 Score: {r2:.2f}")

print("\nModel evalution completed successfully.")

