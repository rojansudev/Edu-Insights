from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import time
import os, re
import operator
from scipy import stats


app = Flask(__name__)


data = pd.read_csv("Final.csv");
columns=data.columns.values.tolist()
columns.sort()




def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def getCorrelation(xCol,yCol,alpha=0.05):
	
	#drop missing
	df2=data.dropna(subset=[xCol,yCol])

	r=df2[xCol].corr(method="spearman",other=df2[yCol])

	r_z = np.arctanh(r)
	se = 1/np.sqrt(df2[xCol].shape[0]-3)
	z = stats.norm.ppf(1-alpha/2)
	lo_z, hi_z = r_z-z*se, r_z+z*se
	lo, hi = np.tanh((lo_z, hi_z))
	return r,lo, hi


def linearReg(n,xCol,yCol,inpFeat,deg):
	
	colList=xCol.copy()
	colList.append(yCol)

	#drop missing values
	df=data.dropna(subset=colList)

	#extract columns for training
	X=df[xCol].values.reshape(-1,n)
	y=df[yCol].values.reshape(-1,1)

	#split data 80% training 20% test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


	polynomial_features= PolynomialFeatures(degree=deg)
	X_train_poly = polynomial_features.fit_transform(X_train)
	X_test_poly = polynomial_features.fit_transform(X_test)

	#train model and predict
	regressor = LinearRegression()  
	regressor.fit(X_train_poly, y_train) 

	y_pred = regressor.predict(X_test_poly)



	#plot actual vs prediction
	if n==1:
		plt.figure()
		plt.scatter(X_test, y_test,  color='blue')
		# plt.plot(X_test, y_pred, color='red', linewidth=1)
		# plt.title("Actual vs Prediction")
		# t=time.time()
		# purge('static/images/',"regression*")
		# path="static/images/regression"+str(t)+".png"
		# plt.savefig(path)
		# sort the values of x before line plot
		t=time.time()
		purge('static/images/',"regression*")
		path="static/images/regression"+str(t)+".png"
		sort_axis = operator.itemgetter(0)
		sorted_zip = sorted(zip(X_test,y_pred), key=sort_axis)
		X_test, y_pred = zip(*sorted_zip)
		plt.plot(X_test, y_pred, color='red',linewidth=1)
		plt.xlabel(xCol[0])  
		plt.ylabel(yCol) 
		plt.savefig(path)

	
	rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))  

	#prediction

	if n!=1:
		return rmse,regressor.predict( polynomial_features.fit_transform (np.array([inpFeat])))[0][0]
	return 	path,rmse,regressor.predict( polynomial_features.fit_transform (np.array([inpFeat])))[0][0]

@app.route('/home')
def home():
    return render_template('home.html')

@app.route("/home/option", methods = ["POST"] )
def process_form():
	
	if request.method == 'POST' and request.form['submit'] == 'Submit!':
			if request.form["option"]=="lreg":
				return render_template('lreg.html',columns=columns)	

			elif request.form["option"]=="corr":
				return render_template('corr.html',columns=columns) 				
				

@app.route("/home/option/corr", methods = ["POST"] )
def process_corr():
	if request.method == 'POST' and request.form['submit'] == 'Submit!':
		xCol=request.form["col1"]
		yCol=request.form["col2"]
		corr,lo,hi=getCorrelation(str(xCol),str(yCol))
		print(lo,hi)

		return render_template('corr.html',corr=corr,columns=columns,sc1=xCol,sc2=yCol,lo=lo,hi=hi)


@app.route("/home/option/lreg", methods = ["POST"] )
def process_lreg():
	if request.method == 'POST' and request.form['Predict'] == 'Predict':
		n=int(request.form["nattr"])
		col=[]
		print(request.form);
		for i in range(n):
			col.append(request.form["mySelect"+str(i+1)])


		val=[]
		for i in range(n):
			val.append(int(request.form["inpcol"+str(i+1)]))

		outcol=request.form["out"]
		deg=int(request.form["deg"])

		try:
			path=''
			rmse=0.0
			predout=0.0
			


			if n!=1:
				rmse,predout=linearReg(n,col,outcol,val,deg)
				return render_template('lreg.html',columns=columns,n=n,col=col,val=val,outcol=outcol,predout=predout,rmse=rmse,deg=deg)
			else:
				path,rmse,predout=linearReg(n,col,outcol,val,deg)
				return render_template('lreg.html',columns=columns,n=n,col=col,val=val,outcol=outcol,predout=predout,rmse=rmse,deg=deg,graphs=path)
		except Exception as e:
			print(e)
			return render_template('lreg.html',err=1)		


@app.route("/graphs", methods = ["POST"] )
def graphs():
	return render_template('graphs.html', name = 'Graphs', url =request.args.get('url'))



if __name__ == '__main__':
    app.run(debug=True)