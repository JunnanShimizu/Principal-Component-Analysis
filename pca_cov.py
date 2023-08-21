'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Junnan Shimizu
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        # Original data, no normalization
        self.original = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''

        means = np.mean(data, axis=0)

        cen_data = data - means

        cov_matrix = np.dot(cen_data.T, cen_data) / (cen_data.shape[0] - 1)

        return cov_matrix

        pass

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''

        prop_var = []
        tot_var = np.sum(e_vals)

        for e_val in e_vals:
            prop_var.append(e_val/tot_var)

        return prop_var

        pass

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''

        cum_var = []
        total_var = 0

        for var in prop_var:
            total_var += var
            cum_var.append(total_var)

        return cum_var

        pass

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''

        data = self.data[vars]
        self.A = data.to_numpy()
        self.normalized = normalize

        if normalize is True:
            self.original = self.A
            max_vals = np.max(self.original, axis=0)
            min_vals = np.min(self.original, axis=0)
            self.A = (self.original - min_vals) / (max_vals - min_vals)

        self.vars = vars
        self.e_vals, self.e_vecs = np.linalg.eig(self.covariance_matrix(self.A))
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

        pass

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''

        if num_pcs_to_keep is None:
            x = range(len(self.cum_var))
        else:
            x = range(num_pcs_to_keep)

        plt.scatter(x=x, y=self.cum_var[:len(x)], s=100, marker='x')
        plt.plot(x, self.cum_var[:len(x)])
        plt.xlabel('# of PCs')
        plt.ylabel('Cumulative Variance')

        # used to set Y-axis range to 0-1, increments by .05
        # plt.yticks(np.arange(0, 1.05, .05))

        # used to set Y-axis range to 0-1, increments by .1
        # plt.yticks(np.arange(0, 1.1, .1))

        pass

    def elbow_plot_range(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''

        if num_pcs_to_keep is None:
            x = range(len(self.cum_var))
        else:
            x = range(num_pcs_to_keep)

        plt.scatter(x=x, y=self.cum_var[:len(x)], s=100, marker='x')
        plt.plot(x, self.cum_var[:len(x)])
        plt.xlabel('# of PCs')
        plt.ylabel('Cumulative Variance')

        # used to set Y-axis range to 0-1, increments by .05
        # plt.yticks(np.arange(0, 1.05, .05))

        # used to set Y-axis range to 0-1, increments by .1
        plt.yticks(np.arange(0, 1.1, .1))


        pass

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''

        PCs = np.zeros(shape=[np.shape(self.e_vecs)[0], len(pcs_to_keep)])

        count = 0
        for i in pcs_to_keep:
            PCs[:, count] = self.e_vecs[:, i]
            count += 1

        self.A_proj = np.dot(self.A, PCs)

        return self.A_proj

        pass

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''

        means = np.mean(self.A, axis=0)
        self.pca_project(range(top_k))

        Ar = np.dot(self.A_proj, np.transpose(self.e_vecs[:, range(top_k)])) + means
    
        if self.normalized is True:
            max_vals = np.max(self.original, axis=0)
            min_vals = np.min(self.original, axis=0)
            Ar = Ar * (max_vals - min_vals) + min_vals

        return Ar 


        pass
