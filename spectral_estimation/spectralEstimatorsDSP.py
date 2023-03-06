# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:30:18 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.linalg

np.random.seed(10)
### Predefined functions


def music(received_signal, num_sources, corr_mat_model_order, digital_freq_grid):
    '''corr_mat_model_order : must be strictly less than half the signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order,corr_mat_model_order)).astype('complex64')
    for ele in np.arange(corr_mat_model_order-1,signal_length):
        if ele == corr_mat_model_order-1:
            auto_corr_matrix += np.matmul(received_signal[ele::-1,:],received_signal[ele::-1,:].T.conj())
        else:
            auto_corr_matrix += np.matmul(received_signal[ele:ele-corr_mat_model_order:-1,:],received_signal[ele:ele-corr_mat_model_order:-1,:].T.conj())
    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    noise_subspace = u[:,num_sources::] # The first # number of sources eigen vectors belong to the signal subspace and the remaining eigen vectors of U belong to the noise subspace which is orthogonal to the signal subspace. Hence pick these eigen vectors
    vandermonde_matrix = np.exp(-1j*np.outer(np.arange(corr_mat_model_order),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies
    """ GhA can be computed in 2 ways:
        1. As a correlation with a vandermonde matrix
        2. Oversampled FFT of each of the noise subspace vectors
        Both 1 and 2 are essentially one and the same. But 1 is compute heavy in terms of MACS while 2 is more FFT friendly

    """
    # GhA = np.matmul(noise_subspace.T.conj(),vandermonde_matrix) #G*A essentially projects the vandermond matrix (which spans the signal subspace) on the noise subspace
    GhA = np.fft.fftshift(np.fft.fft(noise_subspace.T.conj(),n=len(digital_freq_grid),axis=1),axes=(1,)) # Method 2
    AhG = GhA.conj() # A*G
    AhGGhA = np.sum(AhG*GhA,axis=0) # A*GG*A
    pseudo_spectrum = 1/np.abs(AhGGhA) # Pseudo spectrum
    return pseudo_spectrum

def esprit(received_signal, num_sources, corr_mat_model_order):
    '''corr_mat_model_order : must be strictly less than half then signal length'''
    signal_length = len(received_signal)
    auto_corr_matrix = np.zeros((corr_mat_model_order,corr_mat_model_order)).astype('complex64')
    for ele in np.arange(corr_mat_model_order-1,signal_length):
        if ele == corr_mat_model_order-1:
            auto_corr_matrix += np.matmul(received_signal[ele::-1,:],received_signal[ele::-1,:].T.conj())
        else:
            auto_corr_matrix += np.matmul(received_signal[ele:ele-corr_mat_model_order:-1,:],received_signal[ele:ele-corr_mat_model_order:-1,:].T.conj())
    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    us = u[:,0:num_sources] # signal subspace
    us1 = us[0:corr_mat_model_order-1,:] # First N-1 rows of us
    us2 = us[1:corr_mat_model_order,:] # Last N-1 rows of us
    phi = np.matmul(np.linalg.pinv(us1), us2) # phi = pinv(us1)*us2, phi is similar to D and has same eigen vaues as D. D is a diagonal matrix with elements whose phase is the frequencies
    eig_vals = np.linalg.eigvals(phi) # compute eigen values of the phi matrix which are same as the eigen values of the D matrix since phi and D are similar matrices and hence share same eigen values
    est_freq = np.angle(eig_vals) # Angle/phase of the eigen values gives the frequencies
    return est_freq


def vtoeplitz(toprow):

    Npts= toprow.shape[1]
    Nrow= toprow.shape[0]

    ACM= np.zeros((Nrow,Npts,Npts)).astype('complex64')

    for i in range(Npts):
        ACM[:,i,i:]= toprow[:,0:Npts-i].conj()
        ACM[:,i:,i]= toprow[:,0:Npts-i]

    return ACM

def iaa_recursive(received_signal, digital_freq_grid, iterations):
    '''corr_mat_model_order : must be strictly less than half the signal length'''
    signal_length = len(received_signal)
    num_freq_grid_points = len(digital_freq_grid)
    spectrum = np.fft.fftshift(np.fft.fft(received_signal.squeeze(),num_freq_grid_points)/(signal_length),axes=(0,))
#    spectrum = np.ones(num_freq_grid_points)
    vandermonde_matrix = np.exp(1j*np.outer(np.arange(signal_length),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies. Notice the posititve sign inside the exponential
    for iter_num in np.arange(iterations):
        spectrum_without_fftshift = np.fft.fftshift(spectrum,axes=(0,))
        power_vals = np.abs(spectrum_without_fftshift)**2
        double_sided_corr_vect = np.fft.fft(power_vals,num_freq_grid_points)/(num_freq_grid_points)
        single_sided_corr_vec = double_sided_corr_vect[0:signal_length] # r0,r1,..rM-1
        auto_corr_matrix = vtoeplitz(single_sided_corr_vec[None,:])[0,:,:].T
        auto_corr_matrix_inv = np.linalg.inv(auto_corr_matrix)
        Rinv_y = np.matmul(auto_corr_matrix_inv, received_signal)
        """ Ah_Rinv_y can be computed in 2 ways:
            1. Using matrix multiplication/point wise multiplication + sum
            2. Oversampled FFT of Rinv_y followed by fft shift
        Similarly, Rinv_A can be computed in 2 ways:
            1. matrix multiplication of R_inv and Vandermonde matrix
            2. Oversampled FFT of each row of R_inv followed by FFT shift along the column dimension and
            then a flip from left to right. This flip is required because in method 1, we are simply
            correlating each row of R_inv with a sinusoid vector whose frequency is going from -pi to +pi
            from left most column to right most column. This means we are actually obtaining the frequency
            strengths from +pi to -pi(Since FFT has an implicit conjugate sign in the kernel).
            Hence to match the output from method 1, we need to do a flip from left to right.
            3. Since the frequency grid points are from -pi to pi, we can use IFFT instead of FFT+FLIPLR.
            But the scaling with the IFFT needs to be handled.

        Ah_Rinv_A HAS to be done as a point wise multiplication and sum. It cannot be cast as FFTs.
        """
        # Ah_Rinv_y = np.sum(vandermonde_matrix.conj()*Rinv_y,axis=0) # Method 1 for Ah_Rinv_y
        # Rinv_A = np.matmul(auto_corr_matrix_inv,vandermonde_matrix) # Method 1 for Rinv_A

        Ah_Rinv_y = np.fft.fftshift((np.fft.fft(Rinv_y,axis=0,n=num_freq_grid_points)),axes=(0,)).squeeze() # Method 2 for Ah_Rinv_y
        # Rinv_A = np.fliplr(np.fft.fftshift(np.fft.fft(auto_corr_matrix_inv,axis=1,n=num_freq_grid_points),axes=(1,))) # Method 2 for Rinv_A - Use FFT,FFTSHIFT,fliplr
        Rinv_A = np.fft.fftshift(np.fft.ifft(auto_corr_matrix_inv,axis=1,n=num_freq_grid_points),axes=(1,)) # Method 3 for Rinv_A - Use IFFT, FFTSHIFT

        Ah_Rinv_A = np.sum(vandermonde_matrix.conj()*Rinv_A,axis=0)

        spectrum = Ah_Rinv_y/Ah_Rinv_A
        # print(iter_num)
    return spectrum


def IAAGSFactorization1D(data, freqGridPts, NumIter):
    # This function implements non-parametric spectral estimation based on IAA
    # In particular, it implements fast algorithm as given in following paper
    # "IAA-Spectral Estimation: Fast Computation using Gohberg-Semencul Factorization" By Jian Li et.al.

    # {{------- Inputs ---------
    # [1] data: 1D input data vector whose spectrum needs to be estimated using IAA
    # [2] freqGridPts : frequency vector where spectrum needs to be estimated
    # [3] NumIter : should be greater or equal to 1, Number of Iterations for IAA, for  every iteration it gets seeded with new estimates of power of spectrum
    # }}

    # infer some variables
    eps = 1e-40
    M = data.shape[0]
    K = freqGridPts.shape[0]

    # Step1: Compute initial estimate of power spectrum using FFT
    P = np.abs(np.fft.fft(data, K))**2  # O(Klog2(K))
    mu = 1
    # for each iteration of IAA do following
    for i in range(NumIter): # loop over spectrum frequency grid points
        #mu = 1/(i+1)
        #Step2: Take FFT of P to find the first row of autocorrelation matrix
        rvec_up = np.fft.ifft(P)         # O(Klog2(K))
        rvec    = rvec_up[0:M] # first M elements are autocorrelation elements {r(0), r(1), r(2),..., r(M-1)}

        # construct Autocorrelation matrix R
        # R = [r(0) r(1) r(2) ..... r(M-1)]
        #     [r(-1) .............. r(M-2)]
        #     [...........................]
        #               ..
        #               ..
        #     [r(-M+1) r(-M+2 ....... r(0))]
        R = scipy.linalg.toeplitz(rvec)

        # Step3:{{ Compute generators t and s for R_inv using rvec
        # Here we want to solve R_{M-1}w_{M-1}=b where R_{M-1} is first principal sub-matrix of R
        # b = -[r(-1), r(-2), ....., r(-M+1)]
        # }}
        b = -R[1::,0]
        b = b[:,None] # column vec
        rvec_principalsubmat = R[1::,1]#R[1::,1] # Dim: (M-1) length vector
        wvec  = scipy.linalg.solve_toeplitz(rvec_principalsubmat, b)# dimension (M-1 length column vector)
        #Rp = R[1::,1::]
        #wvec = np.dot(scipy.linalg.inv(Rp),b)
        alpha = np.real(R[0,0] + np.dot(b.conj().T,wvec))[0,0] # scalar

        # compute generator vectors
        t = 1/np.sqrt(alpha+eps)*np.vstack((1,wvec)) # Dim: M length column vector
        s = 1/np.sqrt(alpha+eps)*np.hstack((0,np.fliplr(wvec.T).conj()[0,:])) # Dim: M length vector
        s = s[:,None]# column vector
        # Compute R_inv = R_inv1 + R_inv2 using G-S factorization
        e = np.zeros(M)
        e[1] = 1
        Z = scipy.linalg.circulant(e) # circulant matrix
        Z[0,M-1] = 0 # now Z is strictly lower triangular matrix with 1's on first sub-diagonal
        #construct Kyrlov matrices for the corresponding generators
        L_t = np.zeros((M,M)).astype(np.cfloat)#.astype(np.cdouble)
        L_s = np.zeros((M,M)).astype(np.cfloat)#.astype(np.cdouble)
        L_t[:,0] = t[:,0]
        L_s[:,0] = s[:,0]
        for j in range(1,M):
            L_t[:,j] = np.dot(np.linalg.matrix_power(Z,j),t[:,0])
            L_s[:,j] = np.dot(np.linalg.matrix_power(Z,j),s[:,0])
        # R_inv = R_inv1 + R_inv2 where R_inv1 and R_inv2 both has Toeplitz structure (G-S factorization)
        L_t_H = (L_t.T).conj()
        L_s_H = (L_s.T).conj()
        R_inv1 = np.dot(L_t,L_t_H)
        R_inv2 = -1*np.dot(L_s,L_s_H)
        R_inv = R_inv1 + R_inv2

        # Step4a: Compute the Numerator Nr = a^H*R_inv*y = a^H*(R_inv1 + R_inv2)*y
        # Now R_inv*y can be computed by using FFT operations by first converting toeplitz R into circulant matrix
        # Here we will directly multiply instead
        Nr1 = np.dot(R_inv, data)
        Nr1 = np.hstack((Nr1,np.zeros(K-M))) # stack zeros in the end to make it K length column vector
        Nr  = np.fft.fft(Nr1)

        # Step4b: Compute the Denominator
        t_tilda = np.zeros((M,1)).astype(np.cfloat)#.astype(np.cdouble)
        s_tilda = np.zeros((M,1)).astype(np.cfloat)#.astype(np.cdouble)
        for j in range(M):
            t_tilda[j] = (j+1)*t[M-j-1,0]
            s_tilda[j] = (j+1)*s[M-j-1,0]
        # construct Krylov matrix L_t_tilda and L_s_tilda
        L_t_tilda = np.zeros((M,M)).astype(np.cfloat)#.astype(np.cdouble)
        L_s_tilda = np.zeros((M,M)).astype(np.cfloat)#.astype(np.cdouble)
        L_t_tilda[:,0] = t_tilda[:,0]
        L_s_tilda[:,0] = s_tilda[:,0]
        for j in range(1,M):
            L_t_tilda[:,j] = np.dot(np.linalg.matrix_power(Z,j),t_tilda[:,0])
            L_s_tilda[:,j] = np.dot(np.linalg.matrix_power(Z,j),s_tilda[:,0])
        # find c [NOTE: Here we can exploit toeplitz matrix-vector multiplication properties, directly doing instead for now]
        c_int = np.dot(L_t_tilda,t[:,0].conj()) - np.dot(L_s_tilda,s[:,0].conj()) #[c(-M+1), c(-M+2), ...., c(-1), c(0)]
        c_int1 = np.fliplr(c_int[None,:]).conj() #[c(0), c(1), c(2), .... c(M-1)]
        c_int1 = c_int1[0,:] # row vector of length M
        c = np.hstack((c_int1, np.zeros(K - 2*M +1), c_int[0:M-1]))
        Dr = np.fft.ifft(c)

        P_prev = P
        P = P_prev + mu*(np.abs(Nr/(Dr+eps))**2 - P_prev)


    return np.fft.fftshift(P/np.amax(P))


def capon_method_marple_dev(datain,Nsen,Nfft):
    """
    capon_method_marple_dev

    capon_method_marple_dev returns the power spectrum uing Capon Marple algorithm (batch calling)

    Parameters:
    datain (2D array, complex64): Input time samples. [cases, time samples]
    Nsen (integer): No of time samples = datain.shape[1]
    Nfft (integer): No of frequency samples in pseudo - spectrum, typically 1024, 2048
    Ncas (integer): No of cases needing to be submitted to Capon = datain.shape[0]

    Returns:
    pssps_dBm (2D array, float32) : Power spectrum [cases, Nfft]

    """

    Ncas= datain.shape[0]
    pssp= np.zeros((Ncas,Nfft),dtype=np.float32)
    order = (Nsen*3)//4
    for i in range(Ncas): # this for loop needs to be handled in GPU
        psdx = stsminvar_marple(datain[i,:],order,1,Nfft)
        pssp[i,:]= psdx
    pssp_dBm= -10*np.log10(pssp)+10
    pssps_dBm= np.fft.fftshift(pssp_dBm,axes=1)

    return pssps_dBm


def stsminvar_marple(X,order,sampling,NFFT):
    """
    stsminvar_marple

    stsminvar_marple returns the power spectrum uing Capon Marple algorithm (basic engine)

    Parameters:
    X (array, complex64): Input time samples.
    order (integer): AR model order. This should be set as not exceeding 3/4 of the Nsen (no of samples)
    sampling (integer): sampling rate is set as 1
    NFFT (integer): No of frequency samples in pseudo - spectrum, typically 1024, 2048

    Returns:
    PSD (array, float32) : Power spectrum

    References:

    Super-Fast Algorithm for Minimum Variance (Capon) Spectral Estimation, S. Lawrence Marple, Majid Adeli et al.
    Asilomar 2010

    """

    gamma,err,A= hayes_burg(X, order - 1)

    Am= np.fft.fft(A,NFFT)
    Bm= np.fft.fft(A*np.arange(order),NFFT)

    Amc= np.conj(Am)
    Bmc= np.conj(Bm)

    den= order*Am*Amc - Am*Bmc - Bm*Amc

    PSD= np.real(den)  # this is energy

    return PSD


def hayes_burg(x,p):


    """
    hayes_burg

    hayes_burg returns the AR coefficients using Burg Algorithm (basic engine)

    Parameters:
    x (array, complex64): Input time samples.
    p (integer): AR model order. This should be set as not exceeding 3/4 of the Nsen (no of samples)

    Returns:
    gamma (array, float32) : Reflection Coefficients which is the parameter of the Lattice filter representation
    err (array, float32) : Error in the model for all model orers till p
    ap (array, complex64): AR model for p. That is np.dot( [1 ap1 ap2 ... app],[xn xn-1 xn-2 ... xn-p]). ap= [1 ap1 ap2 ... app]

    Only ap is used by the calling function.

    References:

    Hayes: page-319. The LD to compute AR coefficients from gamma is adopted from Orifanidis
    Optimum Signal Processing (http://www.ece.rutgers.edu/~orfanidi/osp2e). Page 195-196

    """

    N= x.size

# Initilizations

    ep= x[1:]
    em= x[0:-1]
    N +=-1

    gamma= np.zeros(p+1,dtype='complex64')
    err= np.zeros(p+1,dtype='complex64')
    A= np.eye(p+1,dtype='complex64') # A[-1,::-1] is the AR coeffecients

    for j in range(1,p+1,1):
        gamma[j]= -2*np.dot(em.conj(),ep) / (np.dot(ep.conj(),ep) + np.dot(em.conj(),em))

        temp1= ep + gamma[j]*em
        temp2= em + np.conj(gamma[j])*ep
        err[j]= np.dot(temp1.conj(),temp1) + np.dot(temp2.conj(),temp2)

        ep= temp1[1:]
        em= temp2[0:-1]

        A[j,0]= gamma[j]

        if (j>1):
            A[j,1:j]= A[j-1,0:j-1] +  gamma[j]* np.flipud(A[j-1,0:j-1].conj())

        N +=-1

    ap= A[-1,::-1]

    return gamma,err,ap


def OMP(dictionary_matrix, y_vec, threshold):
    dictionary = dictionary_matrix.copy()
    dictionary = dictionary/np.linalg.norm(dictionary,axis=0)
    col_index = []
    basis = np.zeros((dictionary.shape[0],0)).astype('complex64')
    residue_mat = np.zeros((dictionary.shape[0],1)).astype('complex64')
    residue = y_vec.copy()
    x_vec_est = np.zeros(dictionary.shape[1]).astype('complex64')[:,None]
    error_iter = []
    res_err_cond = True
    count = 1
    while res_err_cond:
        ind = np.argmax(np.abs(np.matmul(np.conj(dictionary.T), residue))) # Look for the column with maximum projection on the y/residue vector
        col_index.append(ind) # Store the column index
        basis = np.hstack((basis, dictionary[:,ind][:,None])) # Select that column which has the maximum correlation with the y/residue vector and append it to the columns obtained from previous iterations
        z_est = np.matmul(np.linalg.pinv(basis), y_vec) # compute the z_est such that z_est = pinv(basis)*y
        residue = y_vec - np.matmul(basis, z_est) # compute the residue/error as y-y^ where y is our measurement vector and y^ = basis*z_est(from previous step)
        residue_mat = np.hstack((residue_mat, residue)) # store the residue vector for each iteration(just to check how the error/residue is changing across ietrations)
        # err = np.linalg.norm(residue_mat[:,-1] - residue_mat[:,-2]) # check the error in the residue across iterations to check if the residue is changing
        err = np.linalg.norm(residue)**2
        res_err_cond =  err > threshold # check if the change in residue/error is below a particular threshold. Then stop
        error_iter.append(err)
        print('OMP iteration: ',count)
        count+=1
#    valid_col_ind = np.sort(np.array(col_index))
#    z_est_sorted = z_est[np.argsort(np.array(col_index))]
#    x_vec_est[valid_col_ind] = z_est_sorted
    x_vec_est[col_index] = z_est
    return x_vec_est, error_iter





plt.close('all')
num_samples = 32
num_sources = 2
object_snr = np.array([40,35])
noise_power_db = -40 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_variance*(10**(object_snr/10))
signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = num_sources))
complex_signal_amplitudes = weights*signal_phases
random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
fft_resol_fact = 2
resol_fact = 0.65
source_freq = np.array([random_freq, random_freq + resol_fact*2*np.pi/num_samples])
spectrumGridOSRFact = 128
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(spectrumGridOSRFact*num_samples))
source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
received_signal = source_signals + wgn_noise
corr_mat_model_order = num_samples//2-2 # must be strictly less than num_samples/2

magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
phase_spectrum_fft = np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
magnitude_spectrum_fft -= np.amax(magnitude_spectrum_fft,axis=0)[None,:]


""" MUSIC"""
pseudo_spectrum = music(received_signal, num_sources, corr_mat_model_order, digital_freq_grid)
pseudo_spectrum = pseudo_spectrum/np.amax(pseudo_spectrum)
""" ESPRIT"""
est_freq = esprit(received_signal, num_sources, corr_mat_model_order)
est_freq = -1*est_freq
print('True Digital Frequencies:', -source_freq, 'Estimated Digital Frequncies:', est_freq)

""" IAA-GS"""
IAA_spec = IAAGSFactorization1D(data=received_signal.squeeze(), freqGridPts=digital_freq_grid, NumIter=10)
IAA_spec = IAA_spec/np.amax(IAA_spec)
""" IAA Recursive """
iterations = 10
spectrum_iaa = iaa_recursive(received_signal, digital_freq_grid, iterations) # recursive IAA
#    spectrum_iaa = spec_est.iaa_recursive_levinson_temp(received_signal, digital_freq_grid, iterations) # under debug
magnitude_spectrum_iaa = np.abs(spectrum_iaa)**2
magnitude_spectrum_iaa = magnitude_spectrum_iaa/np.amax(magnitude_spectrum_iaa)


""" CAPON _BURG"""
Nfft = len(digital_freq_grid)
capon_spec = capon_method_marple_dev(received_signal.T,num_samples,Nfft)
capon_spec = capon_spec.squeeze()
capon_spec = capon_spec - np.amax(capon_spec)

plt.figure(1,figsize=(20,10))
plt.title('Magnitude Spectrum')
plt.plot(digital_freq_grid, magnitude_spectrum_fft, label = 'FFT')
plt.plot(digital_freq_grid, 10*np.log10(pseudo_spectrum), label='MUSIC')
plt.vlines(est_freq,-80,20, color='magenta', lw=6, alpha=0.3, label='Freq Est by ESPRIT')
plt.plot(digital_freq_grid, 10*np.log10(magnitude_spectrum_iaa), linewidth=6, color='lime', label='IAA')
plt.plot(digital_freq_grid, 10*np.log10(IAA_spec), linewidth=2, color='k', label='IAA-GS')
plt.plot(digital_freq_grid, capon_spec, linewidth=2, color='red', alpha=0.8, label='CAPON-BURG')
plt.vlines(-source_freq,-80,20, alpha=0.3,label = 'Ground truth')
plt.xlabel('Digital Frequencies')
plt.legend()
plt.grid(True)




""" OMP """
num_rows = 208 # 16
num_cols = 4241 # 356
dictionary = np.random.randn(num_rows, num_cols)
dictionary = dictionary/np.linalg.norm(dictionary,axis=0)
sparsity = 5
non_zero_ind = np.random.randint(num_cols, size = sparsity)
x_vec = np.zeros((num_cols,1))
x_vec[non_zero_ind,:] = 1
y_vec = np.matmul(dictionary, x_vec)
threshold = 1e-3
x_vec_est, error_iter = OMP(dictionary, y_vec, threshold)

print('True Col Ind: ', non_zero_ind,  'Estimated Col Ind: ', np.nonzero(x_vec_est)[0])

plt.figure(2,figsize=(20,10))
plt.title('Orthogonal Matching Pursuit')
plt.plot(x_vec, 'o-',lw=8,alpha=0.3)
plt.plot(np.abs(x_vec_est), '*-')
plt.grid(True)
plt.xlabel('Column Index')
plt.legend(['Ground Truth', 'Estimated from OMP'])










