# coding=utf-8
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib import animation
from matplotlib.lines import Line2D


# from pseudopy import NonnormalAuto

def heat_map(matrix, save_name='heat_map', vmin=-1, vmax=1, plot=False):
    fig, ax = plt.subplots()
    plt_mat = ax.imshow(matrix, cmap='jet', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    n_t = matrix.shape[0]
    n_x = matrix.shape[1]

    plt.xticks([0, int(n_x / 2.0) - 1, n_x - 1], ['$0$', '$\pi$', '$2\pi$'])
    plt.yticks([0, int(n_t / 2.0) - 1, n_t - 1], ['$0$', '$0.5$', '$1$'])
    # colorbar
    cbar = plt.colorbar(plt_mat, cmap="jet")
    if plot:
        plt.savefig(save_name + ".pdf", dpi=600, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='pdf',
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
    plt.show()


def matrix_plot(matrix, ax_x=False, ax_y=False):
    fig, ax = plt.subplots()
    plt_mat = ax.imshow(matrix, cmap='RdBu', interpolation='nearest')
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if ax_x is not False:
        plt.xticks([0, matrix.shape[0] / 2, matrix.shape[0]], np.linspace(ax_x[0], ax_x[-1], 3))
    if ax_y is not False:
        plt.yticks([0, matrix.shape[1] / 2, matrix.shape[1]], np.linspace(ax_y[0], ax_y[-1], 3))
    # colorbar
    plt.colorbar(plt_mat)
    #plt.savefig('matrix_plot.png')
    plt.show()


def matrix_row_plot(matrix_list):
    numb = len(matrix_list)
    fig, ax_list = plt.subplots(ncols=numb, figsize=(10, 5))
    for matrix, ax in zip(matrix_list, ax_list):
        plt_mat = ax.imshow(matrix, cmap=plt.cm.jet, interpolation='nearest')
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # colorbar
        plt.axes(ax)
        fig.colorbar(plt_mat, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()

def eigvalue_plot_list(M_list):
    fig, ax = plt.subplots()
    for A in M_list:
        eigenvalues = sp.linalg.eigvals(A)
        real_part = np.real(eigenvalues)
        img_part = np.imag(eigenvalues)
        ax.plot(real_part, img_part, 'o')
    ax.set_xlabel("real part")
    ax.set_ylabel("img part")
    ax.set_title('eigenvalues')
    plt.show()

# gives two arrays and computes the distance to the nearest neighbor
def plot_distance_histogram(l1, l2, num_bins=50):
    min_dist = []
    for e in l1:
        min_dist.append(np.min(np.abs(l2 - e)))
    min_dist = np.asarray(min_dist)

    # the histogram of the data
    n, bins, patches = plt.hist(min_dist, num_bins, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    #     y = mlab.normpdf(bins, mu, sigma)
    #     plt.plot(bins, y, 'r--')
    plt.xlabel('Distances')
    plt.ylabel('Number of points')
    #     plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


def plot_relative_distance_histogram(l1, l2, num_bins=50):
    min_dist = []
    for e in l1:
        min_dist.append(np.min(np.abs(l2 - e) / np.abs(e)))
    min_dist = np.asarray(min_dist)

    # the histogram of the data
    n, bins, patches = plt.hist(min_dist, num_bins, normed=1, facecolor='green', alpha=0.5)
    plt.xlabel('Relative distances')
    plt.ylabel('Number of points')
    plt.subplots_adjust(left=0.15)
    plt.show()


def eigvalue_plot_nested_list(M_list, return_plt=False):
    fig, ax = plt.subplots()
    symbols = [u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for B_list, i in zip(M_list, range(len(M_list))):
        s = symbols[i % 13]
        for A, j in zip(B_list, range(len(B_list))):
            c = colors[j % 7]
            eigenvalues = sp.linalg.eigvals(A)
            real_part = np.real(eigenvalues)
            img_part = np.imag(eigenvalues)
            ax.plot(real_part, img_part, c + s)
    ax.set_xlabel("real part")
    ax.set_ylabel("img part")
    ax.set_title('eigenvalues')
    if return_plt:
        return plt
    else:
        plt.show()


def analyse_norm_differences(norm_tuples):
    relative_difference_high_cfl = map(lambda x: np.abs(x[0] - x[1]) / np.max(x), norm_tuples)
    plt.plot(norm_tuples)
    plt.show()
    plt.loglog(relative_difference_high_cfl)
    plt.show()


def hermitian_part(A):
    return 0.5 * (A + np.transpose(np.conjugate(A)))


def inner_approx_field_of_values(A, n):
    thetas = np.linspace(2 * np.pi / (n), 2 * np.pi, n)
    p_thetas = []
    N = A.shape[0]
    for th in thetas:
        H = hermitian_part(np.exp(1.0j * th) * A)
        eig_theta, eigvec_theta = la.eigh(H, eigvals=(N - 1, N - 1))
        # print "Theta:\t\t",th
        # print "eigvals:\n",eig_theta
        # print "eigvec_theta:\n",eigvec_theta,"\n"
        # print "x* x:",np.dot(np.transpose(np.conjugate(eigvec_theta)),eigvec_theta)
        p_thetas.append(np.dot(np.conjugate(np.transpose(eigvec_theta)), np.dot(A, eigvec_theta)))
        # print p_thetas[-1]
    p = np.asarray(p_thetas).flatten()
    real_part = np.real(p)
    print(np.min(real_part))
    img_part = np.imag(p)
    fig, ax = plt.subplots()
    ax.plot(real_part, img_part, 'ro')
    ax.set_xlabel("real part")
    ax.set_ylabel("img part")
    ax.set_title('field of values')
    plt.show()
    return p


# def generic_pseudo_plot(A, l=1e-5, r=1):
#     pseudo = NonnormalAuto(A, l, r)
#     pseudo.plot([10**k for k in range(-4, 0)], spectrum=sp.linalg.eigvals(A))
#     plt.show()

#
# Animations
#
class MatrixAnimation(animation.TimedAnimation):
    """ A animation class, for matrix functions. This will mainly be used to
        see how iteration matrices change with changing e.g. cfl numbers.

    """

    def __init__(self, matrix_generator, parameter_list, parameter_name, norm_type=np.inf, lim_eigs=0.3):
        """
        :param matrix_generator (function):
        :param parameter_list (np.ndarray):
        :param parameter_name (string):
        :param norm_type:
        :param lim_eigs:
        :return:
        """
        self.m_gen = matrix_generator
        self.p_list = np.asarray(parameter_list)
        self.p_name = parameter_name
        self.norm_type = norm_type

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)
        fig.tight_layout(pad=2)

        ax1.set_xlabel("real part")
        ax1.set_ylabel("img part")

        ax2.set_xlabel(self.p_name)
        ax2.set_ylabel("spectral radius")

        ax3.set_xlabel(self.p_name)
        ax3.set_ylabel("norm")

        self.lin1 = Line2D([], [], marker='o', linestyle='None')
        self.lin2 = Line2D([], [], color='blue', linewidth=2)
        self.lin3 = Line2D([], [], color='red', linewidth=2)

        ax1.add_line(self.lin1)
        ax2.add_line(self.lin2)
        ax3.add_line(self.lin3)

        ax1.set_xlim(-lim_eigs, lim_eigs)
        ax1.set_ylim(-lim_eigs, lim_eigs)

        ax2.set_xlim(self.p_list[0], self.p_list[-1])
        ax2.set_ylim(0, lim_eigs)
        ax2.set_xscale('log')

        ax3.set_xlim(self.p_list[0], self.p_list[-1])
        ax3.set_ylim(0, 2)
        ax3.set_xscale('log')

        self.spec_rads = np.zeros(self.p_list.size)
        self.norms = np.zeros(self.p_list.size)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata

        # get iteration matrix
        T = self.m_gen(self.p_list[i])
        eigenvalues = sp.linalg.eigvals(T)
        x = np.real(eigenvalues)
        y = np.imag(eigenvalues)
        spec_rad = np.max(np.abs(eigenvalues))
        norm = np.linalg.norm(T, self.norm_type)

        self.spec_rads[i] = spec_rad
        self.norms[i] = norm
        self.lin1.set_data(x, y)
        self.lin2.set_data(self.cfl_s[:i], self.spec_rads[:i])
        self.lin3.set_data(self.cfl_s[:i], self.norms[:i])
        self._drawn_artists = [self.lin1] + [self.lin2] + [self.lin3]

    def new_frame_seq(self):
        return iter(range(self.p_list.size))

    def _init_draw(self):
        lines = [self.lin1] + [self.lin2] + [self.lin3]
        for l in lines:
            l.set_data([], [])


class BlockMatrixAnimation(animation.TimedAnimation):
    def __init__(self, block_matrix_generator, num_blocks, parameter_list, parameter_name, norm_type=np.inf,
                 lim_eigs=0.3):
        self.m_gen = block_matrix_generator
        self.p_list = parameter_list
        self.p_name = parameter_name
        self.norm_type = norm_type

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)
        fig.tight_layout(pad=2)

        ax1.set_xlabel("real part")
        ax1.set_ylabel("img part")

        ax2.set_xlabel(self.p_name)
        ax2.set_ylabel("spectral radius")

        ax3.set_xlabel(self.p_name)
        ax3.set_ylabel("norm")

        self.lin1_list = []
        for i in range(num_blocks):
            self.lin1_list.append(Line2D([], [], color=colors[i % 7], marker='o', linestyle='None'))
            ax1.add_line(self.lin1_list[-1])

        ax2.add_line(self.lin2)
        ax3.add_line(self.lin3)

        ax1.set_xlim(-lim_eigs, lim_eigs)
        ax1.set_ylim(-lim_eigs, lim_eigs)

        ax2.set_xlim(self.p_list[0], self.p_list[-1])
        ax2.set_ylim(0, lim_eigs)
        ax2.set_xscale('log')

        ax3.set_xlim(self.p_list[0], self.p_list[-1])
        ax3.set_ylim(0, 2)
        ax3.set_xscale('log')

        self.spec_rads = np.zeros(self.p_list.size)
        self.norms = np.zeros(self.p_list.size)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        T_blocks = self.m_gen(self.p_list[i])

        max_rad = 0
        max_norm = 0

        for t_block, lin in zip(T_blocks, self.lin1_list):
            eigenvalues = sp.linalg.eigvals(t_block)
            x = np.real(eigenvalues)
            y = np.imag(eigenvalues)
            lin.set_data(x, y)
            max_rad = np.max([np.max(np.abs(eigenvalues)), max_rad])
            max_norm = np.max([np.linalg.norm(t_block, np.inf), max_norm])

        self.spec_rads[i] = max_rad
        self.norms[i] = max_norm

        self.lin2.set_data(self.p_list[:i], self.spec_rads[:i])
        self.lin3.set_data(self.p_list[:i], self.norms[:i])
        self._drawn_artists = self.lin1_list + [self.lin2] + [self.lin3]

    def new_frame_seq(self):
        return iter(range(self.p_list.size))

    def _init_draw(self):
        lines = self.lin1_list + [self.lin2] + [self.lin3]
        for l in lines:
            l.set_data([], [])
