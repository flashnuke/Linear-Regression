import numpy as np
import matplotlib.pyplot as plt


def load_data(path: str) -> np.array:
    dataset = open(f"{path}", "r")
    dataset_list = []
    for example in dataset:
        feats = eval(f"[{example}]")
        dataset_list.append(feats)
    return np.array([i for i in dataset_list], dtype=float)  # important to use float


class DataSet:  # pass a matrix of (dimensions = [examples*features])
    def __init__(self, data_set: np.array, theta: list = None, normalization: bool = False,
                 regularization: bool = False, lmbda: float = 0.3):
        if not isinstance(data_set, np.ndarray):
            raise TypeError("Wrong data set format - please use a numpy array")

        if data_set.dtype != float:
            raise TypeError("Matrix elements must be in float format")

        self.m = data_set.shape[0]  # number of EXAMPLES
        self.x = data_set[:, :-1]  # excluding output from features

        self._norm_mode = False
        if normalization:
            self._norm_mode = True
            r, c = self.x.shape
            self.means = []
            self.stds = []
            for i in range(c):

                self.means.append(np.mean(data_set[:, i]))
                self.x[:, i] = self.x[:, i] - np.mean(data_set[:, i])

                if np.std(data_set[:, i]) != 0.0:
                    self.stds.append(np.std(data_set[:, i]))
                    self.x[:, i] = self.x[:, i] / np.std(data_set[:, i])

                else:
                    self.stds.append(1)

        self._regmode = False
        if regularization:
            self._regmode = True
            self.lmbda = lmbda
        self.x = np.c_[np.ones([self.m, 1]), self.x]  # adding bias unit
        self.n = self.x.shape[1]  # number of features

        self.y = np.array(data_set[:, -1])  # output vector
        self.y = self.y.reshape(len(self.y), 1)  # making vertical -> vectorization

        if isinstance(theta, np.ndarray):
            raise Exception(f"You are trying to pass the weights as an array. Please pass as a list")
        elif theta:
            if len(theta) != self.n:
                raise Exception(f"Mismatched dimensions: {len(theta)} weights versus {self.n} features")
            elif not isinstance(theta, list):
                raise TypeError(f"Weights should be passed in list format")
            else:
                self.theta = np.array([weight for weight in theta])
        else:
            self.theta = self._initial_theta()

    def _initial_theta(self):
        return np.array([theta for theta in np.random.normal(1, 0.3, self.n)])


class Model(DataSet):
    def __init__(self, data_set: np.array, theta: list = None, normalization: bool = False,
                 regularization: bool = False, lmbda: float = 0.3, alpha: float = 0.01,
                 iters: int = 10000, convergence: float = 1.000000001):
        super().__init__(data_set, theta=theta, normalization=normalization, regularization=regularization, lmbda=lmbda)
        self.mse = 0
        self.opt_theta = self.theta  # this value will be changed with every iteration
        self.alpha = alpha
        self.iters = iters
        self.convergence = convergence
        self.jlist = []  # list of errors
        self.iter_num = 0  # number of iterations for completion
        self._build_model()
        self.r2 = self._calc_r2()
        self.scaledtheta = None
        if self._norm_mode:
            self._convert_thetas()
        self.intercept = self.opt_theta[0, 0]
        self.hypothesis = self._generate_hypothesis()

    def __repr__(self):
        return self.hypothesis

    def _training_predictions(self):
        if self.opt_theta.shape[0] != self.x.T.shape[1]:
            self.opt_theta = self.opt_theta.reshape(1, len(self.opt_theta))
        h = self.opt_theta.dot(self.x.T)
        return h.reshape(self.y.shape[0], 1)  # returns prediction for each sample

    def _costfunction(self, error):
        error = error**2
        j = 0.5*(1/self.m)*error  # returns a vector of predictions
        if self._regmode:
            regterm = np.sum(self.opt_theta**2)*self.lmbda
            j += regterm
        self.jlist.append(float(sum(j)))
        return j

    def _gradient_descent(self):
        error = self._training_predictions() - self.y
        self._costfunction(error)

        if self.opt_theta.shape[1] != 1:
            self.opt_theta = self.opt_theta.reshape(len(self.theta), 1)

        if self._regmode:
            regterm = (self.lmbda / self.m) * self.opt_theta
            regterm[:, 0] = 0
            self.opt_theta -= self.alpha * (((1 / self.m) * self.x.T.dot(error)) + regterm)
        else:
            self.opt_theta -= (self.alpha / self.m) * (self.x.T.dot(error))

    def _build_model(self):
        for i in range(self.iters):
            self._gradient_descent()
            self.iter_num = i + 1
            if i > 2:
                if self.jlist[-2] / self.jlist[-1] < self.convergence:
                    self.convergence = True
                    break

        if self._regmode:
            self.mse = 2*float(self.jlist[-1] - np.sum(self.opt_theta**2)*self.lmbda)
        self.mse = 2*float(self.jlist[-1])

    def predict(self, features: np.array, withbias: bool = False) -> np.array:   # pass features without bias unit
        if features.dtype != float:
            raise TypeError("Prediction features must be passed as float type")

        if not withbias:
            if features.ndim == 1 or features.ndim == 0:
                features = np.c_[1, features]  # adding bias unit
            else:
                r, c = features.shape
                features = np.c_[np.ones([r, 1]), features]  # adding bias unit

        predictions = features.dot(self.opt_theta)
        return predictions

    def _convert_thetas(self):
        self.scaledtheta = self.opt_theta
        feats = self.x[0, :]  # removing x0=1 as (theta0*x0 = theta0) anyway
        prediction = self.predict(feats, withbias=True)
        feats = self.x[0, 1:]
        feats = (feats*self.stds)+self.means  # reverting scale from features

        for index, std in enumerate(self.stds):
            self.opt_theta[index+1, :] = self.opt_theta[index+1, :]/std  # remove scale from all other thetas

        theta0 = prediction - (self.opt_theta[1, :]*feats)  # calculating theta0
        self.opt_theta[0, :] = theta0[0]  # inserting theta0 into theta matrix

    def _calc_r2(self):
        error = abs(self.y) - abs((self.x.dot(self.opt_theta)))  # note: this must use old thetas!
        modelvar = np.var(error)
        meanvar = np.mean(self.y)

        return (meanvar-modelvar)/meanvar

    def visualize(self, save: bool = False):  # only for n=2 (with intercept)
        if self.n > 2:
            raise Exception(f"Visualization of n > 2 models is currently unavailable")

        originalfeatures = (self.x*self.stds)+self.means if self._norm_mode else self.x
        x = np.linspace(np.min(self.x[:, 1]), np.max(self.x[:, 1]), 1000*np.min(self.x[:, 1]))
        y = [float(self.predict(pred)) for pred in x]
        plt.plot(x, y, '-r')
        plt.scatter(originalfeatures[:, 1], self.y)
        plt.xlabel("x1 feature")
        plt.ylabel("y prediction")
        plt.title(self.hypothesis)
        plt.grid()
        if save:
            plt.savefig("linear-reg-plot.png")
        plt.show()

    def _generate_hypothesis(self):
        xt = str(f"({self.intercept})") + str([f"+({self.opt_theta[i, :]})*x{i}"
                                               for i in range(1, len(self.opt_theta))])
        xt = xt.replace("[", "")
        xt = xt.replace(",", " + ")
        xt = xt.replace("'", "")
        xt = xt.replace("]", "")
        xt = xt.replace(" ", "")
        return f"h(x) = {xt}"


if __name__ == "__main__":
    andrewsdataset = load_data("ex1data1.txt")
    dataset_array2 = np.array([[2, 5],
                               [3, 7],
                               [4, 9],
                               [5, 11]], dtype=float)

    mod = Model(dataset_array2, normalization=False, regularization=False, lmbda=0.03)

    print(mod.r2)
    mod.visualize(save=True)
    print(mod.opt_theta)
    # print(mod)
    # print(mod.hypothesis)
    # print(mod.stds)
    # print(mod.jlist)
    print(mod.mse)
    # print(mod.iter_num)
    #
    featuersss = np.array([[10.0],
                           [11.0]], dtype=float)
    result = mod.predict(featuersss)
    print(result)
    # print(mod.x)
