"""A reference implementation of spectral map."""

# Reference implementation from plumID:24.005

import numpy as np
import torch
from sklearn.model_selection import train_test_split

class SpectralMap(torch.nn.Module):

    def __init__(
        self,
        *,
        n_eigen=50,
        n_gap=2,
        n_neigh=0.5,
        batch_size=1000,
        alpha=0.5,
        layers=None,
    ):
        super(SpectralMap, self).__init__()

        self.n_eigen = n_eigen
        self.n_gap = n_gap
        self.n_neigh = n_neigh
        self.alpha = alpha
        self.batch_size = batch_size
        self.coeffs = torch.nn.Parameter(torch.rand(self.batch_size))
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append((torch.nn.Linear(layers[i], layers[i + 1], bias=True)))
            self.layers.append(torch.nn.ELU())
        self.map = torch.nn.Sequential(*self.layers[:-1])
        print(self.map)
        self.n_cv = layers[-1]

    def diffusion_kernel(self, X):
        pdist = torch.cdist(X, X)
            
        k = int(max(2, np.round(pdist.shape[0] * self.n_neigh)))
        epsilon = pdist.sort(dim=1)[0][:,k]
        epsilon = epsilon.outer(epsilon)
        epsilon = epsilon.clamp_min(1e-7)
        kernel = torch.exp(-pdist * pdist / epsilon)
        norm = torch.pow(torch.sum(kernel, axis=1), -self.alpha)
        kernel_diff = torch.outer(norm, norm) * kernel
        sqrt_norm = torch.sqrt(kernel_diff.sum(1))
        kernel_diff = kernel_diff / sqrt_norm.outer(sqrt_norm)
        return kernel_diff

    def eigendecomposition(self, X):
        kernel = self.diffusion_kernel(X)
        eigval, eigvec = torch.linalg.eigh(kernel)
        idx = eigval.argsort(descending=True)
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        eigvec[:, eigvec.sum(axis=0) < 1.0] *= -1
        eigval = eigval[:self.n_eigen]
        eigvec = eigvec[:, :self.n_eigen]
        spectral_gap = eigval[self.n_gap - 1] - eigval[self.n_gap]
        self.spectral_gap = spectral_gap
        self.eigval = eigval
        self.eigvec = eigvec
        return spectral_gap, eigval, eigvec

    def eigendecomposition_split(self, X, n_splits=1, n_eigen=20, n_size=0.9):
        spectrum = np.zeros(shape=(n_splits, n_eigen))
        for n in range(n_splits):
            x, _ = train_test_split(X, train_size=n_size)
            _, eigvals, _ = self.eigendecomposition(x)
            spectrum[n] = eigvals[:n_eigen].detach().numpy()
        return spectrum

    def forward(self, X):
        X = self.map(X)
        return X

    def fit(self, X_train, X_valid, n_epochs, learning_rate, n_stride, batch_size):

        n_train_samples = X_train.shape[0]
        n_train_batch = n_train_samples / batch_size

        n_valid_samples = X_valid.shape[0]
        n_valid_batch = n_valid_samples / batch_size

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, maximize=True)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=50)

        train_score_l = []
        valid_score_l = []

        for epoch in range(n_epochs):

            train_perm_idx = torch.randperm(n_train_samples)
            X_train = X_train[train_perm_idx]
            train_score = 0.0
            i = 0
            while i < n_train_samples:
                batch_slice = torch.arange(i, i + batch_size)
                x_batch = X_train[batch_slice]
                y_batch = self(x_batch)
                spectral_gap, eigval, eigvec = self.eigendecomposition(y_batch)
                optimizer.zero_grad()
                gap = spectral_gap / n_train_batch
                gap.backward()
                optimizer.step()
                train_score += gap
                i += batch_size
            scheduler.step()

            valid_perm_idx = torch.randperm(n_valid_samples)
            X_valid = X_valid[valid_perm_idx]
            valid_score = 0.0
            i = 0
            while i < n_valid_samples:
                batch_slice = torch.arange(i, i + batch_size)
                x_batch = X_valid[batch_slice]
                y_batch = self(x_batch)
                spectral_gap, eigval, eigvec = self.eigendecomposition(y_batch)
                gap = spectral_gap / n_valid_batch
                valid_score += gap
                i += batch_size

            train_score_l.append(train_score.detach())
            valid_score_l.append(valid_score.detach())

            if epoch % n_stride == 0:
              print('Epoch: ', epoch + 1, '| train spectral gap: %.4f ' % train_score.detach())
              print('Epoch: ', epoch + 1, '| valid spectral gap: %.4f ' % valid_score.detach())

        self.train_score = np.array(train_score_l)
        self.valid_score = np.array(valid_score_l)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, required=True, help="data set")
    parser.add_argument("--stride", type=int, default=1, help="training stride")
    parser.add_argument("--n_eigen", type=int, default=50, help="number of eigenvalues")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--n_gap", type=int, default=2, help="position of spectral gap")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="diffusion kernel normalization alpha")
    parser.add_argument("--n_neigh", type=float, default=None, help="fraction of neighborhood size")
    parser.add_argument("--o", type=str, required=True, help="output")
    parser.add_argument("--n_feature", type=int, default=None, help="number of feature from data set")
    parser.add_argument("--valid_size", type=float, default=None, help="validation size")
    
    args = parser.parse_args()

    if args.n_feature is None:
        X = np.load(args.npy)
    else:
        X = np.load(args.npy)[:, args.n_feature]
        X = X.reshape(-1, 1)

    X = np.array(X, dtype='float32')
    X_train, X_valid = None, None
    if args.valid_size is not None:
        X_train, X_valid = train_test_split(X, test_size=args.valid_size)
    else:
        X_train = X
    
    W_train, W_valid = X_train[:: args.stride], X_valid[:: args.stride]
    print("Train set", W_train.shape)
    print("Valid set", W_valid.shape)

    n_train_batches = W_train.shape[0] // args.batch_size
    n_train_end = n_train_batches * args.batch_size

    n_valid_batches = W_valid.shape[0] // args.batch_size
    n_valid_end = n_valid_batches * args.batch_size

    print("Number of train batches:", W_train.shape[0] / args.batch_size)
    print("Number of valid batches:", W_valid.shape[0] / args.batch_size)
    
    Z_train = torch.from_numpy(W_train)
    Z_valid = torch.from_numpy(W_valid)

    embedding = SpectralMap(
        n_eigen=args.n_eigen,
        n_gap=args.n_gap,
        n_neigh=args.n_neigh,
        batch_size=args.batch_size,
        alpha=args.alpha,
        layers=[W_train.shape[1], 200, 100, 50, 2]
    )

    embedding.fit(
        Z_train,
        Z_valid,
        n_epochs=args.n_epochs, 
        learning_rate=0.001,
        n_stride=1, 
        batch_size=args.batch_size,
    )

    Y = embedding(torch.from_numpy(X))

    result = {
        'train_score': embedding.train_score,
        'valid_score': embedding.valid_score,
        'map': Y.detach().numpy(),
    }

    np.savez(args.o, **result)

if __name__ == "__main__":
    import sys
    import argparse

    main(sys.argv[1:])
