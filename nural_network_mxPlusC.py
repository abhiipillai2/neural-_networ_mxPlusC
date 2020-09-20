import pandas as  pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_set = pd.read_csv("student_scores2.csv")

# data_set.plot.scatter(x ='Hours',y = 'IQ',c = 'Pass',colormap ='bwr')
# plt.show()

x = data_set.drop(["Scores","Pass"],axis = 1).values
y = data_set["Pass"].values.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def sigmoid(s):
	
	return 1/(1 + np.exp(-s))

def sigmoid_prime(s):
		return s * (1 - s)

# nural netwrok object
class mxPlusC(object):

	def __init__(self,input_size,output_size):

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = 3

		self.w1 = np.zeros((self.input_size,self.hidden_size))
		self.w2 = np.zeros((self.hidden_size,self.output_size))

	def forward(self,X_train):

		self.output_1 = sigmoid(np.dot(X_train,self.w1))
		self.output_2 =sigmoid(np.dot(self.output_1,self.w2))
			
		return self.output_2


	def backword(self,X_train,y_train):

		delta_w2 = (self.output_2 - y_train) * sigmoid_prime(self.output_2)

		delta_w1 = delta_w2.dot(self.w2.T) * sigmoid_prime(self.output_1)

		self.w2 = self.w2 + self.output_1.T.dot(delta_w2) * -1
		self.w1 = self.w1 + X_train.T.dot(delta_w1) * -1
			

	def fit(self,X_train,y_train):

		self.forward(X_train)
		self.backword(X_train,y_train)



	def predict(self,X_test,y_test):

		prediction = self.forward(X_test)
		self.backword(X_test,y_test)

		return prediction


	def loss(self,x,y_actual):

		prediction = self.forward(x)

		return np.mean(np.square(y_actual - prediction))

		
nn = mxPlusC(input_size = 2,output_size = 1)

train_loss = []
test_loss = []

for i in range(10000):
	nn.fit(X_train,y_train)		
	train_loss.append(nn.loss(X_train,y_train))
	test_loss.append(nn.loss(X_test,y_test))



plt.plot(train_loss,'r--')
plt.plot(test_loss,'g')
plt.show()


		