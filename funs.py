from joblib import Parallel, delayed
from multiprocessing import Pool
from time import time, sleep
import mne
import numpy as np
import os
from scipy.sparse import csr_matrix

def predict_sources_2(solver_dicts, fwd, info, x_sample, sim_info, n_sources=None):
	stc_dict = dict()
	proc_time_make = dict()
	proc_time_apply = dict()
	start_make = 0
	end_make = 0

	if n_sources is None:
		n_sources = sim_info.n_sources

	for solver_dict in solver_dicts:
		solver_name = solver_dict["display_name"]
		make_args = solver_dict["make_args"]
		apply_args = solver_dict["apply_args"]
		recompute_make = solver_dict["recompute_make"]
		solver = solver_dict["solver"]

		evoked = mne.EvokedArray(x_sample.T, info, tmin=0)
		if not solver.made_inverse_operator or recompute_make:
			start_make = time()
			if "n" in make_args and "k" in make_args:
				solver.make_inverse_operator(fwd, evoked, **make_args)
			else:
				solver.make_inverse_operator(fwd, evoked, n=n_sources, k=n_sources, **make_args)
			end_make = time()
		
		start_apply = time()
		stc = solver.apply_inverse_operator(evoked, **apply_args)
		end_apply = time()
		
		stc_dict[solver_name] = csr_matrix(stc.data)
		proc_time_make[solver_name] = end_make - start_make
		proc_time_apply[solver_name] =  end_apply - start_apply
    
	return stc_dict, proc_time_make, proc_time_apply

def predict_sources_parallel3(solver_dicts, fwd, info, x_test, sim_info, n_jobs=-1, n_sources=None):
    # prepare solvers
	if type(x_test) == list:
		x_test_temp = x_test[0][0].T
	else:
		x_test_temp = x_test[0].T
	
	evoked = mne.EvokedArray(x_test_temp, info, tmin=0)
	for solver_dict in solver_dicts:
		solver_dict["solver"].make_inverse_operator(fwd, evoked, **solver_dict["make_args"])

	# This worked only if x_test is numpy.ndarray, not if its a list
	# res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(predict_sources_2)(solver_dicts, fwd, info, x_sample, sim_info_row, n_sources=n_sources) for x_sample, (_, sim_info_row) in zip(x_test, sim_info.iterrows()))

	# Do this instead
	if type(x_test) == list:
		x_test_batch = []
		for x1 in x_test:
			for x2 in x1:
				x_test_batch.append(x2)
	else:
		x_test_batch = x_test
	
	res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(predict_sources_2)(solver_dicts, fwd, info, x_sample, sim_info_row, n_sources=n_sources) for x_sample, (_, sim_info_row) in zip(x_test_batch, sim_info.iterrows()))
	return res

def predict_sources(solver_dict, fwd, info, x_test, sim_info):
	
	solver_name = solver_dict["name"]
	make_args = solver_dict["make_args"]
	apply_args = solver_dict["apply_args"]
	solver = solver_dict["solver"]
	stc_dict = {solver_name: []}
	proc_time_make = {solver_name: []}
	proc_time_apply = {solver_name: []}
	print("lets go ", solver_name)
	
	for i_sample, x_sample in enumerate(x_test):
		evoked = mne.EvokedArray(x_sample.T, info, tmin=0)
		if solver_dict["recompute_make"] or i_sample == 0:
			start_make = time()
			solver.make_inverse_operator(fwd, evoked, alpha="auto", n=sim_info.n_sources.values[i_sample], k=sim_info.n_sources.values[i_sample], **make_args)
			end_make = time()
        
		start_apply = time()
		stc = solver.apply_inverse_operator(evoked, **apply_args)
		end_apply = time()
		
		stc_dict[solver_name].append(stc)
		proc_time_make[solver_name].append(end_make - start_make)
		proc_time_apply[solver_name].append(end_apply - start_apply)
    
	print("done with ", solver_name)
        
	return stc_dict, proc_time_make, proc_time_apply

def predict_sources_parallel2(solver_dicts, fwd, info, x_test, sim_info, n_jobs=-1):
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(predict_sources)(solver_dict, fwd, info, x_test, sim_info) for solver_dict in solver_dicts)
    return res

def predict_sources_wrapper(args):
	return predict_sources(*args)

def predict_sources_parallel(solver_dicts, fwd, info, x_test, sim_info, processes=4):
    input_args = [(solver_dict, fwd, info, x_test, sim_info) for solver_dict in solver_dicts]
    pool = Pool(processes=processes)
    
    results = pool.map(predict_sources_wrapper, input_args)
    pool.close()
    pool.join()
    
    return results

def test_par(processes=-1):
    vals = [10, 10, 10, 10]
    # with Pool(processes=processes) as pool:
    pool = Pool(processes=processes)
    start = time()
    results = pool.map(sleeper, vals)
    end = time()
    pool.close()
    pool.join()
    print(end-start)
    return results

def sleeper(s):
    print(s)
    sleep(s)
    return s

def get_varexp(M_true: np.ndarray, M_est:np.ndarray):
	''' Get variance explained by M_est concerning M_true.

	Parameters
	----------
	M_true : np.ndarray
		The true measurements (M/EEG).
	M_est : np.ndarray
		The fitted data.
	
	Return
	------
	varexp : np.ndarray

	'''
	from scipy.stats import pearsonr

	if M_true.ndim == 1:
		M_true = M_true[:, np.newaxis]
	if M_est.ndim == 1:
		M_est = M_est[:, np.newaxis]
	
	# Tests
	assert M_true.ndim == 2, "M_true should be of dimension 2"
	assert M_est.ndim == 2, "M_est should be of dimension 2"
	assert M_true.shape == M_est.shape, "Dimension mismatch between M_true and M_est"
	
	n_channels, n_timepoints = M_true.shape
	varexp = np.zeros((n_timepoints,))

	for i in range(n_timepoints):
		r, _ = pearsonr(M_true[:, i], M_est[:, i])
		varexp[i] = r**2
		
	return varexp

class SurfaceCalculator:
	''' Class to facilitate the computation of active cortical surface area.
	
	Parameters
	----------
	fwd : mne.Forward
		The forward model object.

	Example
	-------
	get_surface = SurfaceCalculator(fwd)
	area_true = get_surface(J)  # J is your 1D source vector

	'''
	def __init__(self, fwd):
		self.fwd = fwd
		self.n_channels, self.n_dipoles = fwd["sol"]["data"].shape
		self.pos_from_forward()
		self.prepare_tris()
		self.calculate_triangle_areas()
		

	def prepare_tris(self,):
		''' Get a correct list of all triangles in the source space.'''
		self.fwd

		source_model = self.fwd["src"]
		vertno = source_model[0]["vertno"]
		tri_left = source_model[0]["use_tris"]
		tri_left_new = np.zeros(tri_left.shape)
		for idx, tri in enumerate(tri_left):
			tri_left_new[idx, :] = np.array([np.where(vertno==i)[0][0] for i in tri])

		tri_right = source_model[1]["use_tris"]
		tri_right_new = np.zeros(tri_right.shape)
		for idx, tri in enumerate(tri_right):
			tri_right_new[idx, :] = np.array([np.where(vertno==i)[0][0] for i in tri])

		self.tris = np.concatenate([tri_left_new, tri_right_new + self.n_dipoles//2], axis=0).astype(int)

	def calculate_triangle_areas(self,):
		'''Calculate the area of all triangles in the source space.'''
		tris_areas = np.zeros(self.tris.shape[0])
		for i, tri in enumerate(self.tris):
			xyz1, xyz2, xyz3 = [self.pos[idx] for idx in tri]
			area = self.calculate_triangle_area(xyz1, xyz2, xyz3)
			tris_areas[i] = area

		self.tris_areas = tris_areas
	
	@staticmethod
	def calculate_triangle_area(A, B, C):
		'''Calculates the area of a triangle defined by three coordinates.'''
		# Calculate the vectors AB and AC
		AB = [B[i] - A[i] for i in range(3)]
		AC = [C[i] - A[i] for i in range(3)]

		# Compute the cross product of AB and AC
		cross_product = [
			AB[1]*AC[2] - AB[2]*AC[1],
			AB[2]*AC[0] - AB[0]*AC[2],
			AB[0]*AC[1] - AB[1]*AC[0]
		]

		# Calculate the magnitude of the cross product
		magnitude = sum([i**2 for i in cross_product]) ** 0.5

		# Return half of the magnitude, which is the area of the triangle
		return 0.5 * magnitude
	
	def pos_from_forward(self, verbose=0):
		''' Get vertex/dipole positions from mne.Forward model

		Parameters
		----------
		
		Return
		------
		pos : numpy.ndarray
			A 2D matrix containing the MNI coordinates of the vertices/ dipoles

		Note
		----
		self.fwd must contain some subject id in self.fwd["src"][0]["subject_his_id"]
		in order to work.
		'''
		# Get Subjects ID
		subject_his_id = self.fwd["src"][0]["subject_his_id"]
		src = self.fwd["src"]

		# Extract vertex positions from left and right source space
		pos_left = mne.vertex_to_mni(src[0]["vertno"], 0, subject_his_id, 
									verbose=verbose)
		pos_right = mne.vertex_to_mni(src[1]["vertno"], 1, subject_his_id, 
									verbose=verbose)

		# concatenate coordinates from both hemispheres
		pos = np.concatenate([pos_left, pos_right], axis=0)

		self.pos = pos

	def __call__(self, J, threshold=None):
		'''Compute surface area of source vector J. The main method of this class to compute the surface area of some source vector J.

		Parameters
		----------
		J : numpy.ndarray (1D)
			The sparse source vector
		threshold : None/ float

		
		Return
		------
		area: float
			The triangle area.
		'''
		if threshold is not None and type(threshold) == float:
			jmax = abs(J).max()
			J[J<jmax*threshold] = 0
		area = 0.0
		active_idc = set(np.nonzero(J)[0])
		n_active = len(active_idc)
		if n_active <= 1:
			return area
		
		members_found = set()
		for tri_idx, tri in enumerate(self.tris):
			tri_set = set(tri)
			if tri_set.issubset(active_idc):
				area += self.tris_areas[tri_idx]
				members_found.update(tri_set)
			if len(members_found) == n_active:
				break
			
		return area
	
def get_whitener(C_noise):
	'''
	Code adopted from Matlab code kindly provided by Dr Amita Giri.
	'''
	# scaler = np.sqrt(np.trace(C_noise**2)) / C_noise.shape[0]  # Normalization
	# C_noise /= scaler

	Sn, Un = np.linalg.eigh(C_noise)

	Sn = np.real(Sn)
	Un = np.real(Un)

	Sn = Sn[::-1]
	Un = Un[:, ::-1]

	tol = len(Sn) * np.finfo(np.float32).eps * np.float32(Sn[0])
	Rank_Noise = np.sum(Sn > tol)
	# Trim eigenvectors and -values
	Un = Un[:, :Rank_Noise]
	Sn = Sn[:Rank_Noise]

	# Rescale
	# Sn *= scaler

	# Rebuild the noise covariance matrix with just the non-zero components
	# C_noise = Un @ np.diag(Sn**2) @ Un.T
	C_noise = Un @ np.diag(Sn) @ Un.T


	# Regularization method: none (this is inverse noise covariance (thus, whitener matrix))
	iW_noise = Un @ np.diag(1./np.sqrt(Sn)) @ Un.T

	return iW_noise, C_noise

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def whiten_evoked_fwd(epochs, fwd, noise_cov=None, tmin=None, tmax=0, method="empirical", rank=None, verbose=0):
	''' Whiten mne.Evoked and mne.Forward objects.

	Parameters
	----------
	epochs : mne.Epochs
	fwd : mne.Forward

	Return
	------
	evoked : mne.Evoked
		The non-whitened data.
	evoked_w : mne.Evoked
		The whitened data.
	fwd_w : mne.Forward
		The whitened forward matrix.
	'''	
	get_idc = lambda long_list, short_list: np.array([long_list.index(elem) for elem in short_list])

	# Ensure same channels
	common_ch_names = intersection(epochs.ch_names, fwd.ch_names)
	epochs = epochs.pick_channels(common_ch_names)
	fwd = fwd.pick_channels(common_ch_names)
	# 
	print(f"Forward channels: {len(fwd.ch_names)}")
	print(f"Epochs channels: {len(epochs.ch_names)}")
	channel_types = list(set(epochs.get_channel_types()))

	fwd_w = fwd.copy()
	evoked_w = epochs.copy().average()
	evoked = epochs.copy().average()


	for channel_type in channel_types:
		# Get channel names of current type
		channels_type = epochs.copy().pick_types(channel_type).ch_names
		print("channels: ", channel_type, len(channels_type))
		# select these channels
		epochs_channel_type = epochs.copy().pick_channels(channels_type)
		evoked_channel_type = epochs_channel_type.average()
		gain_channel_type = fwd.copy().pick_channels(channels_type)["sol"]["data"]
		if noise_cov is None:
			noise_cov_ct = mne.compute_covariance(epochs_channel_type, 
									  tmin=tmin, 
									  tmax=tmax,
									  method=method, 
									  rank=rank,
									  verbose=verbose)
		else:
			noise_cov_ct = noise_cov.copy().pick_channels(channels_type)
        
		W, _ = get_whitener(noise_cov_ct.data)
		W *= np.sqrt(evoked_channel_type.nave)

		# Whiten Forward
		gain_channel_type = W @ gain_channel_type

		# Whiten evoked
		evoked_channel_type.data = W @ evoked_channel_type.data

		channel_type_idc = get_idc(fwd.ch_names, channels_type)

		fwd_w["sol"]["data"][channel_type_idc, :] = gain_channel_type
		evoked_w.data[channel_type_idc] = evoked_channel_type.data
		
	return evoked, evoked_w, fwd_w

def get_mirrored_stc(stc, subjects_dir, subject_from, subject_to, smooth=5):
	''' Compute the mirrored source estimate.

	Parameters
	----------
	stc : mne.SourceEstimate
		The source estimate.
	subjects_dir : str
		The subjects directory containing the symmetric subject.
	subject_from : str
		The name of the subject of the source estimate.
	subject_to : str
		The name of the symmetric subject.
	smooth : int
		The amount of smoothing to apply to the source estimate.
	
	Return
	------
	stc_original : mne.SourceEstimate
		The original source estimate, mapped to fsaverage_sym.
	stc_mirror : mne.SourceEstimate
		The mirrored source estimate, mapped to fsaverage_sym.
	
	Example
	-------
	subjects_dir = "C:/Users/lukas/mne_data"
	subject_from = "MNE-spm-face/subjects/spm"
	subject_to = "MNE-sample-data/subjects/fsaverage_sym"
	smooth = 5
	stc_mirror = get_mirrored_stc(stc, subjects_dir, subject_from, subject_to, smooth)
	
	'''
	stc.subject = subject_from

	assert os.path.exists(subjects_dir), "subjects_dir does not exist"
	assert os.path.exists(os.path.join(subjects_dir, subject_from)), "subject_from does not exist"
	assert os.path.exists(os.path.join(subjects_dir, subject_to)), "subject_to does not exist"
	
	# stc_original = mne.compute_source_morph(
	# 	stc, subject_from, subject_to, smooth=smooth, warn=False, subjects_dir=subjects_dir, verbose=0
	# ).apply(stc)

	morph_to_fsaverage = mne.compute_source_morph(
		stc, subject_from, subject_to, smooth=smooth, warn=False, subjects_dir=subjects_dir, verbose=0
	)
	stc_original = morph_to_fsaverage.apply(stc)
	
	# Now we can compute the mapping from left to right
	morph_mirror = mne.compute_source_morph(
		stc_original,
		subject_to,
		subject_to,
		spacing=stc_original.vertices,
		warn=False,
		subjects_dir=subjects_dir,
		xhemi=True,
		verbose=0,
	)  # creating morph map

	stc_mirror = morph_mirror.apply(stc_original)

	return stc_original, stc_mirror