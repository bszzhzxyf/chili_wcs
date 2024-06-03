"""
Identifier:     csst_ifs_wcs/fit_wcs.py
Name:           fit_wcs.py
Description:    Triangle matching coordinates and WCS Parameter Fitting.
Author:         Yifei Xiong
Created:        2023-11-30
Modified-History:
    2023-11-30: add module header
"""

import numpy as np
from astropy.wcs import WCS as aWCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from .wcs import WCS
from collections import OrderedDict


class TriMatch():
    """
    Class for triangle matching between pixel and celestial coordinate sets.

    use triangles to match pixel & sky data.

    Parameters
    ----------
    pixel : array-like
        Array of pixel coordinates.
    radec : array-like
        Array of celestial coordinates.

    Attributes
    ----------
    pixel : numpy.ndarray
        Array of pixel coordinates.
    radec : numpy.ndarray
        Array of celestial coordinates.
    matches_array : numpy.ndarray
        Array containing matched triangles between pixel and celestial coordinates.

    Methods
    -------
    invariants(x1, x2, x3, sky=False)
        Compute invariant features for a set of three points.

    vertex(pixels, vertex_indices, sky=False)
        Return vertex_indices ordered in an (a, b, c) form.

    create_triangles(pixels, sky)
        Return an array of unique invariants derived from the array `pixels`.

    matching(pixel, radec)
        Perform triangle matching between pixel and celestial coordinate sets.
    """
    def __init__(self, pixel: np.ndarray, radec: np.ndarray):
        self.pixel = pixel
        self.radec = radec
        self.matches_array = self.matching(pixel, radec)

    def invariants(self, x1, x2, x3, sky=False):
        # if sky==True then use spherical coord
        if sky is False:
            sides = np.sort([
                np.linalg.norm(x1 - x2),
                np.linalg.norm(x2 - x3),
                np.linalg.norm(x1 - x3),
            ])
        elif sky is True:
            "Given 3stars x1, x2, x3, return the invariant features for the set."
            x1b = SkyCoord(ra=x1[0] * u.degree, dec=x1[1] * u.degree)
            x2b = SkyCoord(ra=x2[0] * u.degree, dec=x2[1] * u.degree)
            x3b = SkyCoord(ra=x3[0] * u.degree, dec=x3[1] * u.degree)
            sides = np.sort([
                x1b.separation(x2b).deg,
                x2b.separation(x3b).deg,
                x3b.separation(x1b).deg,
            ])
        return [sides[2] / sides[1], sides[1] / sides[0]]

    def vertex(self, pixels, vertex_indices, sky=False):
        """Return vertex_indices ordered in an (a, b, c) form where:
          a is the vertex defined by L1 & L2
          b is the vertex defined by L2 & L3
          c is the vertex defined by L3 & L1
        and L1 < L2 < L3 are the sides of the triangle
        defined by vertex_indices."""
        ind1, ind2, ind3 = vertex_indices
        x1, x2, x3 = pixels[vertex_indices]

        side_ind = np.array([(ind1, ind2), (ind2, ind3), (ind3, ind1)])

        if sky is False:
            side_lengths = list(
                map(np.linalg.norm, (x1 - x2, x2 - x3, x3 - x1)))
            l1_ind, l2_ind, l3_ind = np.argsort(side_lengths)
        elif sky is True:

            x1b = SkyCoord(ra=x1[0] * u.degree, dec=x1[1] * u.degree)
            x2b = SkyCoord(ra=x2[0] * u.degree, dec=x2[1] * u.degree)
            x3b = SkyCoord(ra=x3[0] * u.degree, dec=x3[1] * u.degree)
            side_lengths = list([
                x1b.separation(x2b).deg,
                x2b.separation(x3b).deg,
                x3b.separation(x1b).deg,
            ])
            l1_ind, l2_ind, l3_ind = np.argsort(side_lengths)
        # the most common vertex in the list of vertices for two sides is the
        # point at which they meet.
        from collections import Counter
        count = Counter(side_ind[[l1_ind, l2_ind]].flatten())
        a = count.most_common(1)[0][0]
        count = Counter(side_ind[[l2_ind, l3_ind]].flatten())
        b = count.most_common(1)[0][0]
        count = Counter(side_ind[[l3_ind, l1_ind]].flatten())
        c = count.most_common(1)[0][0]
        return np.array([a, b, c])

    def create_triangles(self, pixels, sky):
        """Return an array of (unique) invariants derived from the array `pixels`.
        Return an array of the indices of `pixels` that correspond to each
        invariant, arranged as described in _arrangetriplet."""
        from itertools import combinations
        from functools import partial
        arrange = partial(self.vertex, pixels=pixels, sky=sky)
        inv = []
        triang_vrtx = []
        coordtree = KDTree(pixels)
        # The number of nearest neighbors to request (to work with few pixels)
        NUM_NEAREST_NEIGHBORS = 5
        knn = min(len(pixels), NUM_NEAREST_NEIGHBORS)
        for asrc in pixels:
            __, indx = coordtree.query(asrc, knn)
            # Generate all possible triangles with the 5 indx provided, and store
            # them with the order (a, b, c) defined in _arrangetriplet
            all_asterism_triang = [
                arrange(vertex_indices=list(cmb))
                for cmb in combinations(indx, 3)
            ]
            triang_vrtx.extend(all_asterism_triang)
            inv.extend([
                self.invariants(*pixels[triplet], sky=sky)
                for triplet in all_asterism_triang
            ])
        # Remove here all possible duplicate triangles
        uniq_ind = [
            pos for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1:]
        ]
        inv_uniq = np.array(inv)[uniq_ind]
        triang_vrtx_uniq = np.array(triang_vrtx)[uniq_ind]
        return inv_uniq, triang_vrtx_uniq

    def matching(self, pixel, radec):
        pixel_controlp = np.array(pixel)  # array
        pixel_invariants, pixel_asterisms = self.create_triangles(
            pixel_controlp, sky=False)
        pixel_invariant_tree = KDTree(pixel_invariants)
        radec_controlp = np.array(radec)  # array
        radec_invariants, radec_asterisms = self.create_triangles(
            radec_controlp, sky=True)
        radec_invariant_tree = KDTree(radec_invariants)
        matches_list = pixel_invariant_tree.query_ball_tree(
            radec_invariant_tree, r=0.05)
        matches = []
        for t1, t2_list in zip(pixel_asterisms, matches_list):
            for t2 in radec_asterisms[t2_list]:
                matches.append(list(zip(t1, t2)))
        matches = np.array(matches)
        return matches


class FitParam(WCS):
    """
    Class for fitting WCS parameters based on matching pixel and celestial coordinates.

    It use RANSAC method to fit WCS parameters.

    Parameters
    ----------
    pixel : array-like
        Array of pixel coordinates.
    radec : array-like
        Array of celestial coordinates.
    matches : array-like
        Array containing matched triangles between pixel and celestial coordinates.
    method : str
        Transformation method. Options are "Normal" for Normal WCS transform,
        "MCI" for MCI WCS transform, and "IFS" for IFS WCS transform.
    fixedpara : dict
        Dictionary containing known parameters such as CRPIX.
    inipara : array-like
        Initial guess for the transformation parameters.

    Attributes
    ----------
    pixel : numpy.ndarray
        Array of pixel coordinates.
    radec : numpy.ndarray
        Array of celestial coordinates.
    matches_array : numpy.ndarray
        Array containing matched triangles between pixel and celestial coordinates.
    method : str
        Transformation method.
    fixedpara : dict
        Dictionary containing known parameters such as CRPIX.
    inipara : array-like
        Initial guess for the transformation parameters.
    bestparam : array-like
        Best-fit WCS parameters.

    Methods
    -------
    fit_resids(params, xy, radec)
        Compute residuals for the fit.

    fit_wcs(xy, radec)
        Perform the WCS fitting.

    fit(matches)
        Fit model parameters to data.

    get_error(data, approx_t)
        Calculate the error for a given set of data points.

    ransac(data, min_data_points, max_iter, thresh, min_matches)
        Perform RANSAC algorithm for robust model fitting.

    find_bestparam(pixel, radec, matches)
        Find the best-fit WCS parameters based on matching triangles between
        pixel and celestial coordinates.
    """

    def __init__(self, pixel: np.ndarray, radec: np.ndarray, matches: np.ndarray, method: str, fixedpara: dict, inipara: np.ndarray):
        """
        method="Normal":Normal WCS transform
              ="MCI"   :MCI WCS transform
              ="IFS"   :IFS WCS transform
        inipara:dict,some known parameter such as crpix
        """
        super().__init__()
        self.pixel = pixel
        self.radec = radec
        self.matches_array = matches
        self.method = method
        self.fixedpara = fixedpara
        self.inipara = inipara
        self.bestparam, self.coords = self.find_bestparam(pixel, radec, matches)

        if self.method == "Normal":
            self.wcs_fitted = OrderedDict({"CRPIX1": self.CRPIX[0], "CRPIX2": self.CRPIX[1],
                                           "CRVAL1": self.bestparam[0], "CRVAL2": self.bestparam[1],
                                           "CD1_1": self.bestparam[2], "CD1_2": self.bestparam[3],
                                           "CD2_1": self.bestparam[4], "CD2_2": self.bestparam[5]})
            w = aWCS(naxis=2)
            w.wcs.crpix = [self.wcs_fitted['CRPIX1'], self.wcs_fitted['CRPIX2']]
            w.wcs.crval = [self.wcs_fitted['CRVAL1'], self.wcs_fitted['CRVAL2']]
            w.wcs.pc[0, 0] = self.wcs_fitted['CD1_1']
            w.wcs.pc[0, 1] = self.wcs_fitted['CD1_2']
            w.wcs.pc[1, 0] = self.wcs_fitted['CD2_1']
            w.wcs.pc[1, 1] = self.wcs_fitted['CD2_2']
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            self.awcs = w
        if self.method == "MCI":
            self.mwcs_fitted = OrderedDict({"MCRPIX1": self.MCRPIX[0], "MCRPIX2": self.MCRPIX[1],
                                            "MCRVAL1": self.bestparam[0], "MCRVAL2": self.bestparam[1],
                                            "MCD1_1": self.bestparam[2], "MCD2_2": self.bestparam[3],
                                            "MLONPOLE": self.bestparam[4]})
        if self.method == "IFS":
            self.iwcs_fitted = OrderedDict({"ICRPIX1": self.ICRPIX[0], "ICRPIX2": self.ICRPIX[1],
                                            "ICRVAL1": self.bestparam[0], "ICRVAL2": self.bestparam[1],
                                            "ICD1_1": self.bestparam[2], "ICD2_2": self.bestparam[3],
                                            "ILONPOLE": self.bestparam[4]})

    def fit_resids(self, params, xy, radec):
        # params:crval1,crval2,cd11,cd12,cd21,cd22
        # Predict Value
        if self.method == "Normal":
            self.CRPIX = self.fixedpara["CRPIX"]
            self.CRVAL = np.array([params[0], params[1]])
            self.CD = np.matrix([[params[2], params[3]],
                                 [params[4], params[5]]])
            ra_c, dec_c = self.xy2sky(xy).T
        if self.method == "MCI":
            self.MCRPIX = self.fixedpara["MCRPIX"]
            self.MCRVAL = np.array([params[0], params[1]])
            self.MCD = np.matrix([[params[2], 0], [0, params[3]]])
            self.MLONPOLE = params[4]
            ra_c, dec_c = self.mci_xy2sky(xy).T
        if self.method == "IFS":
            # params:crval1,crval2,cd11,cd12,cd21,cd22
            self.ICRPIX = self.fixedpara["ICRPIX"]
            self.MCRVAL = self.fixedpara["MCRVAL"]
            self.MLONPOLE = self.fixedpara["MLONPOLE"]
            self.ICRVAL = np.array([params[0], params[1]])
            self.ICD = np.matrix([[params[2], 0], [0, params[3]]])
            self.ILONPOLE = params[4]
            ra_c, dec_c = self.ifs_xy2sky(xy).T
        # True Value
        ra, dec = radec.T
        # Residual
        ra_resids = (ra - ra_c) * 3600 * np.cos(np.radians(dec))
        # print("ra:%s,ra_c:%s" % (ra, ra_c))
        dec_resids = (dec - dec_c) * 3600
        # print("ra_resids0:%s" % ra_resids0)
        # print("ra_resids:%s" % ra_resids)
        # print(np.radians(ra))
        # resids = np.sqrt(ra_resids**2 + dec_resids**2)
        resids = np.concatenate((ra_resids, dec_resids))
        # print("ra_shape:{}".format(ra.shape))
        # print("Resi_shape:{}".format(resids.shape))
        # print("Resi:{}".format(resids))
        return resids

    def fit_wcs(self, xy, radec):
        fit = least_squares(self.fit_resids,
                            self.inipara,
                            method="lm",
                            args=(xy, radec))
        return fit.x

    def fit(self, matches):
        d1, d2, d3 = matches.shape
        s, d = matches.reshape(d1 * d2, d3).T
        approx_t = self.fit_wcs(self.pixel[s], self.radec[d])
        return approx_t

    def get_error(self, data, approx_t):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = self.fit_resids(approx_t, self.pixel[s], self.radec[d])
        ra_resid, dec_resid = np.hsplit(resid, 2)
        error = np.sqrt(ra_resid**2 + dec_resid**2).reshape(d1, d2).max(axis=1)
        return error

    def ransac(self, data, min_data_points, max_iter, thresh, min_matches):
        """fit model parameters to data using the RANSAC algorithm

        This implementation written from pseudocode found at
        http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

        Given:
            data: a set of data points
            model: a model that can be fitted to data points
            min_data_points: the minimum number of data values required to fit the
                model
            max_iter: the maximum number of iterations allowed in the algorithm
            thresh: a threshold value to determine when a data point fits a model
            min_matches: the min number of matches required to assert that a model
                fits well to data
        Return:
            bestfit: model parameters which best fit the data (or nil if no good
                      model is found)"""
        iterations = 0
        bestfit = None
        best_err = 1000000
        best_matches = 0
        best_inlier_idxs = None
        n_data = data.shape[0]
        n = min_data_points
        all_idxs = np.arange(n_data)
        while (best_err > 0.2) or (iterations < max_iter):
            # Partition indices into two random subsets
            np.random.shuffle(all_idxs)
            maybe_idxs, test_idxs = all_idxs[:n], all_idxs[n:]
            maybeinliers = data[maybe_idxs, :]
            test_points = data[test_idxs, :]
            maybemodel = self.fit(maybeinliers)
            # print("maybemodel:{}".format(maybemodel))
            test_err = self.get_error(test_points, maybemodel)
            # print("test_error:{}".format(test_err))
            # select indices of rows with accepted points
            also_idxs = test_idxs[test_err < thresh]
            alsoinliers = data[also_idxs, :]
            # print("alsoinliers:{}".format(alsoinliers))
            num_inliers = len(maybeinliers) + len(alsoinliers)
            if num_inliers >= best_matches:
                best_matches = num_inliers
                # print("best_matches:{}".format(best_matches))
                betterdata = np.concatenate((maybeinliers, alsoinliers))
                better_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
                bettermodel = self.fit(betterdata)
                # print("bettermodel:{}".format(bettermodel))
                # best number of matches
                better_errs = self.get_error(betterdata, bettermodel)
                # print("better_errors:{}".format(better_errs))
                better_err = np.max(better_errs)
                # print("better_error:{}".format(better_err))
                if better_err < best_err:
                    best_err = better_err
                    print("best_error:{}".format(best_err))
                    bestfit = bettermodel
                    print("bestfit:{}".format(bestfit))
                    best_inlier_idxs = better_inlier_idxs
            iterations += 1
        return bestfit, best_inlier_idxs

    def find_bestparam(self, pixel, radec, matches):
        n_invariants = len(matches)
        pixel_controlp = np.array(pixel)
        radec_controlp = np.array(radec)

        if (len(pixel_controlp) == 3
                or len(radec_controlp) == 3) and len(matches) == 1:
            best_t = self.fit(matches)
            inlier_ind = np.arange(len(matches))  # All of the indices
        else:
            # RANSAC parameter
            # min_data_points, max_iter, thresh, min_matches
            min_data_points = 3  # param 1
            max_iter = n_invariants * 3  # param 2
            thresh = 0.05
            MIN_MATCHES_FRACTION = 0.5
            min_matches = max(
                1, min(10, int(n_invariants * MIN_MATCHES_FRACTION)))
            best_t, inlier_ind = self.ransac(matches, min_data_points,
                                             max_iter, thresh, min_matches)
        if best_t is not None:
            if self.method == "Normal":
                self.CRVAL = np.array([best_t[0], best_t[1]])
                self.CD = np.matrix([[best_t[2], best_t[3]],
                                     [best_t[4], best_t[5]]])
                self.trans_func = self.xy2sky
            if self.method == "MCI":
                self.MCRVAL = np.array([best_t[0], best_t[1]])
                self.MCD = np.matrix([[best_t[2], 0], [0, best_t[3]]])
                self.MLONPOLE = best_t[4]
                self.trans_func = self.mci_xy2sky
            if self.method == "IFS":
                # params:crval1,crval2,cd11,cd12,cd21,cd22
                self.ICRVAL = np.array([best_t[0], best_t[1]])
                self.ICD = np.matrix([[best_t[2], 0], [0, best_t[3]]])
                self.ILONPOLE = best_t[4]
                self.trans_func = self.ifs_xy2sky

            triangle_inliers = matches[inlier_ind]
            d1, d2, d3 = triangle_inliers.shape
            inl_arr = triangle_inliers.reshape(d1 * d2, d3)
            inl_unique = set(tuple(pair) for pair in inl_arr)
            inl_dict = {}
            for s_i, t_i in inl_unique:
                # calculate error
                s_vertex = pixel_controlp[s_i].reshape(1, 2)
                t_vertex = radec_controlp[t_i].reshape(1, 2)
                t_vertex_pred = self.trans_func(s_vertex)
                error = np.linalg.norm(t_vertex_pred - t_vertex)
                # if s_i not in dict, or if its error is smaller than previous error
                if s_i not in inl_dict or (error < inl_dict[s_i][1]):
                    inl_dict[s_i] = (t_i, error)
            inl_arr_unique = np.array([[s_i, t_i]
                                       for s_i, (t_i, e) in inl_dict.items()])
            s, d = inl_arr_unique.T
            pixel_unique = pixel_controlp[s]
            radec_unique = radec_controlp[d]
            print("Best Fit WCS is %s" % best_t)
        if best_t is None:
            pixel_unique = None
            radec_unique = None
            print("Failed to find WCS parameter ")
        return best_t, (pixel_unique, radec_unique)
