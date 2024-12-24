import os
import numpy as np
import cv2

class LM_NN_Classifier:
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 mu=1e-2, mu_max=1e4, mu_min=1e-4, mu_decay=0.9, 
                 tolerance=1e-4, max_iters=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = self.init_params()
        self.mu = mu
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.mu_decay = mu_decay
        self.tolerance = tolerance
        self.max_iters = max_iters

    def init_params(self):
        W1 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0/(self.input_dim+self.hidden_dim))
        b1 = np.zeros(self.hidden_dim)
        W2 = np.random.randn(self.output_dim, self.hidden_dim) * np.sqrt(2.0/(self.hidden_dim+self.output_dim))
        b2 = np.zeros(self.output_dim)
        return np.concatenate([W1.flatten(), b1, W2.flatten(), b2])

    def unpack_params(self, params):
        w1_size = self.hidden_dim * self.input_dim
        W1 = params[:w1_size].reshape(self.hidden_dim, self.input_dim)
        b1 = params[w1_size:w1_size+self.hidden_dim]
        w2_size = self.output_dim * self.hidden_dim
        W2 = params[w1_size+self.hidden_dim:w1_size+self.hidden_dim+w2_size].reshape(self.output_dim, self.hidden_dim)
        b2 = params[-self.output_dim:]
        return W1, b1, W2, b2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, params):
        W1, b1, W2, b2 = self.unpack_params(params)
        z1 = np.dot(W1, x) + b1
        h1 = self.sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        exp_z2 = np.exp(z2 - np.max(z2))
        out = exp_z2 / np.sum(exp_z2)
        return h1, out

    def loss(self, params, X, Y):
        residuals = []
        for i in range(X.shape[0]):
            _, out = self.forward(X[i], params)
            residuals.append(out - Y[i])
        return np.array(residuals)

    def jacobian(self, params, X, Y):
        W1, b1, W2, b2 = self.unpack_params(params)
        N = X.shape[0]
        P = len(params)
        J = np.zeros((N, self.output_dim, P))

        for i in range(N):
            x = X[i]
            z1 = np.dot(W1, x) + b1
            h1 = self.sigmoid(z1)
            z2 = np.dot(W2, h1) + b2
            exp_z2 = np.exp(z2 - np.max(z2))
            out = exp_z2 / np.sum(exp_z2)

            dOut_dz2 = np.diag(out) - np.outer(out, out)
            dh1_dz1 = h1*(1 - h1)

            for k_ in range(self.output_dim):
                d_residual_dz2 = dOut_dz2[k_]

                for j_ in range(self.output_dim):
                    for i_ in range(self.hidden_dim):
                        w2_start = self.hidden_dim*self.input_dim + self.hidden_dim
                        w2_index = w2_start + j_*self.hidden_dim + i_
                        J[i, k_, w2_index] = d_residual_dz2[j_]*h1[i_]
                for j_ in range(self.output_dim):
                    b2_index = self.hidden_dim*self.input_dim + self.hidden_dim + self.output_dim*self.hidden_dim + j_
                    J[i, k_, b2_index] = d_residual_dz2[j_]

                for a_ in range(self.hidden_dim):
                    for b_ in range(self.input_dim):
                        val = 0
                        for j_ in range(self.output_dim):
                            val += d_residual_dz2[j_]*W2[j_, a_]*dh1_dz1[a_]*x[b_]
                        w1_index = a_*self.input_dim + b_
                        J[i, k_, w1_index] = val

                for a_ in range(self.hidden_dim):
                    val = 0
                    for j_ in range(self.output_dim):
                        val += d_residual_dz2[j_]*W2[j_, a_]*dh1_dz1[a_]
                    b1_index = self.hidden_dim*self.input_dim + a_
                    J[i, k_, b1_index] = val

        J = J.reshape(N*self.output_dim, P)
        return J

    def one_hot(self, labels):
        num_classes = len(np.unique(labels))
        oh = np.zeros((len(labels), num_classes))
        for i, l in enumerate(labels):
            oh[i, l-1] = 1
        return oh

    def train(self, X_train, y_train):
        Y_train = self.one_hot(y_train)
        params = self.params
        for iteration in range(self.max_iters):
            residuals = self.loss(params, X_train, Y_train)
            R = residuals.reshape(-1)
            J = self.jacobian(params, X_train, Y_train)

            H = J.T @ J
            g = J.T @ R

            I = np.eye(len(params))
            update = np.linalg.solve(H + self.mu * I, -g)

            new_params = params + update
            new_residuals = self.loss(new_params, X_train, Y_train).reshape(-1)
            new_cost = np.sum(new_residuals**2)
            old_cost = np.sum(R**2)

            if new_cost < old_cost:
                params = new_params
                self.mu = max(self.mu * self.mu_decay, self.mu_min)
                print(f"Iteration {iteration}, Cost: {new_cost}")
                if np.abs(old_cost - new_cost) < self.tolerance:
                    print("Convergence reached.")
                    break
            else:
                self.mu = min(self.mu * 10, self.mu_max)
        self.params = params

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            _, out = self.forward(X[i], self.params)
            preds.append(np.argmax(out) + 1)
        return np.array(preds)



people = 40
train_data = []
train_labels = []
test_data = []
test_labels = []

for k in range(1, people+1):
    for m in range(1, 10+1):
        path_string = os.path.join('ORL3232', str(k), f'{m}.bmp')
        image_data = cv2.imread(path_string, cv2.IMREAD_GRAYSCALE)
        image_data = image_data.astype(np.float64)
        if k == 1 and m == 1:
            row, col = image_data.shape
        flat_img = image_data.flatten()
        if m % 2 == 1:
            train_data.append(flat_img)
            train_labels.append(k)
        else:
            test_data.append(flat_img)
            test_labels.append(k)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

def pca(X, n_components=50):
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvecs = eigvecs[:, :n_components]
    return eigvecs, mean_vec

eigvecs, train_mean = pca(train_data, n_components=20)
train_data_pca = (train_data - train_mean) @ eigvecs
test_data_pca = (test_data - train_mean) @ eigvecs

input_dim = train_data_pca.shape[1]
hidden_dim = 50
output_dim = len(np.unique(train_labels))

# Create a new instance of the model (or use the same one, but typically you'd re-initialize):
model_new = LM_NN_Classifier(input_dim, hidden_dim, output_dim, max_iters=50)

# Load the saved parameters
loaded_params = np.load("lm_nn_params.npy")
model_new.params = loaded_params
print("Parameters loaded from file.")

# Now model_new has the same parameters as the old model
predictions = model_new.predict(test_data_pca)
accuracy = np.mean(predictions == test_labels)
print(f"Classification Accuracy (loaded params): {accuracy * 100:.2f}%")
