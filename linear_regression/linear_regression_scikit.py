import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self):
        noise = 0.1
        self.x = np.random.rand(100, 1)
        self.y = 2*self.x + noise*np.random.rand(100,1)

    def linreg(self):
        linreg = LinearRegression()
        linreg.fit(self.x,self.y)
        self.xnew = np.array([[0], [1], [2], [3]])
        self.ypredict = linreg.predict(self.xnew)
        self.ypredict2 = linreg.predict(self.x)
        """
        Note that we have ypredict and ypredict2.
        ypredict is for visualization of the line, while ypredict2 is for finding the
        error, since then we use all the x points.
        """

        print("The interference alpha: \n", linreg.intercept_)
        print("The coefficient beta: \n", linreg.coef_)
        print('Mean square error: %.2f' %mean_squared_error(self.y, self.ypredict2))
        #R2 score equal to 1 is a perfect prediction.
        print("Variance score: %.4f" %r2_score(self.y, self.ypredict2))



    def error(self):
        self.error = np.abs(self.ypredict2 - self.y) / np.abs(self.y)

    def visualization(self):
        plt.plot(self.xnew, self.ypredict, "r-")
        plt.plot(self.x, self.y ,'ro')
        plt.axis([0,1.0,0, 5.0])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.grid(1)
        plt.title(r'Simple Linear Regression')
        plt.show()

        plt.plot(self.x, self.error, "ro")
        plt.show()

if __name__ == '__main__':
    obj = LinearRegression()
    obj.linreg()
    obj.error()
    #obj.visualization()
