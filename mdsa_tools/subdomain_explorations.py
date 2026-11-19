'''

Use results of systems analysis to explore potential preferred structural conformations

- Evaluating PCA/UMAP clusters at different target dimensions.
- Cohesion over time, using sliding and shrinking windows.
- Transition matrices without cross-replicate jumps.


See Also
--------
mdsa_tools.Analysis.systems_analysis

'''

import numpy as np
import os
import pandas as pd


class subdomain_explorations:
    """
    Use results of systems analysis to explore potential preferred structural conformations

    - Evaluating PCA/UMAP clusters at different target dimensions.
    - Cohesion over time, using sliding and shrinking windows.
    - Transition matrices without cross-replicate jumps.

    Attributes
    ----------
    labels : np.ndarray of int or None
        Cluster labels per frame (0-based). Transition matrices require consecutive state ids.
    centers : np.ndarray or None
        Cluster centers in the same space as reduced coordinates.
    reduced_coordinates : np.ndarray or None
        Low-dimensional embedding coordinates (e.g., PCA/UMAP).
    frame_scale : list[int] or None
        Number of frames per replicate.

    Notes
    -----
    Labels, centers, coordinates, and frame counts are stored on the object
    and used when no overrides are supplied.
    """

    def __init__(self, labels=None, centers=None, reduced_coordinates=None, frame_scale=None):
        """Init with labels, centers, coordinates, and frame sizes."""
        self.labels = labels
        self.centers = centers
        self.frame_scale = frame_scale
        self.reduced_coordinates = reduced_coordinates

###########################################################################
# candidate state evaluation
###########################################################################

    def rmsd_from_centers(self, X=None, labels=None, centers=None):
        """
        Per-cluster RMSD of points to their assigned cluster center.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_dims), optional
            Points in embedding space (PCA/UMAP). Defaults to stored coordinates.
        labels : np.ndarray of int, shape (n_samples,), optional
            Cluster labels for each row of X. Defaults to stored labels.
        centers : np.ndarray, shape (n_states, n_dims), optional
            Cluster centers. Defaults to stored centers.

        Returns
        -------
        np.ndarray of shape (n_present_states, 2)
            Columns: (cluster_id, rmsd), both stored as floats.

        Notes
        -----
        Uses Euclidean norm in the embedding space; no cluster-size weighting.
        """
        if X is None:
            X = self.reduced_coordinates
        if labels is None:
            labels = self.labels
        if centers is None:
            centers = self.centers

        results = []
        for k in np.unique(labels):
            m = (labels == k)
            d = np.linalg.norm(X[m] - centers[int(k)], axis=1)
            rmsd = float(np.sqrt(np.mean(d**2)))
            results.append((int(k), rmsd))
        return np.array(results)

    def evaluate_cohesion_slidingwindow(
        self,
        labels=None,
        centers=None,
        reduced_coordinates=None,
        frame_scale=None,
        step_size=None,
    ):
        """
        Fixed-size *sliding window* per replicate.

        At window j, take up to step_size remaining frames from each replicate,
        concatenate, then compute per-cluster RMSD to centers for that slice.
        Advance by step_size for max(frame_scale) // step_size windows.

        Parameters
        ----------
        labels, centers, reduced_coordinates, frame_scale : optional
            Override stored attributes.
        step_size : int, default 10
            Window length (in frames) and hop size.

        Returns
        -------
        pandas.DataFrame
            Columns: ['cluster', 'rmsd', 'window'] where window is 1-based.

        Notes
        -----
        - Replicates with no frames left at the window start contribute nothing.
        - Windows never cross replicate boundaries.
        - Shows changes in cluster cohesion over time.
        """
        if reduced_coordinates is None:
            reduced_coordinates = self.reduced_coordinates
        if frame_scale is None:
            frame_scale = self.frame_scale
        if step_size is None:
            step_size = 10
        if labels is None:
            labels = self.labels
        if centers is None:
            centers = self.centers

        slidingwindow = 0
        window_df_all = []
        for j in range(1, (np.max(frame_scale) // step_size) + 1):
            print(f"shrink: {j}")
            mask = []
            for rep_length in frame_scale:
                if slidingwindow > rep_length:
                    mask.append(np.full(rep_length, False))
                    continue
                replicate_bools = np.full(rep_length, False)
                replicate_bools[slidingwindow:slidingwindow + step_size] = True
                mask.append(replicate_bools)
            slidingwindow += step_size

            window_mask = np.concatenate(mask)
            window_labels = labels[window_mask]
            window_coordinates = reduced_coordinates[window_mask, :]
            rmsd_results = self.rmsd_from_centers(window_coordinates, window_labels, centers)
            windowdf = pd.DataFrame(rmsd_results, columns=('cluster', 'rmsd'))
            windowdf['window'] = j
            window_df_all.append(windowdf)

        return pd.concat(window_df_all)

    def evaluate_cohesion_shrinkingwindow(
        self,
        labels=None,
        centers=None,
        reduced_coordinates=None,
        frame_scale=None,
        step_size=None,
    ):
        """
        *Shrinking-from-the-start* window (aka keep the tail).

        At step j (1-based), drop the first (j - 1) * step_size frames of each
        replicate and use the rest. The first window uses all frames.

        Parameters
        ----------
        labels, centers, reduced_coordinates, frame_scale : optional
            Override stored attributes.
        step_size : int, default 10
            How much to move the left edge each step.

        Returns
        -------
        pandas.DataFrame
            Columns: ['cluster', 'rmsd', 'window'] where window is 1-based.

        Notes
        -----
        Shows how cohesion changes as early frames are removed.
        """
        if reduced_coordinates is None:
            reduced_coordinates = self.reduced_coordinates
        if frame_scale is None:
            frame_scale = self.frame_scale
        if step_size is None:
            step_size = 10
        if labels is None:
            labels = self.labels
        if centers is None:
            centers = self.centers

        creepingstart = 0
        window_df_all = []
        for j in range(1, (np.max(frame_scale) // step_size) + 1):
            print(f"shrink: {j}")
            mask = []
            for rep_length in frame_scale:
                if creepingstart > rep_length:
                    mask.append(np.full(rep_length, False))
                    continue
                replicate_bools = np.full(rep_length, True)
                replicate_bools[0:creepingstart] = False
                mask.append(replicate_bools)

            window_mask = np.concatenate(mask)
            window_labels = labels[window_mask]
            window_coordinates = reduced_coordinates[window_mask, :]
            rmsd_results = self.rmsd_from_centers(window_coordinates, window_labels, centers)
            windowdf = pd.DataFrame(rmsd_results, columns=('cluster', 'rmsd'))
            windowdf['window'] = j
            creepingstart += step_size
            window_df_all.append(windowdf)

        window_df_all = pd.concat(window_df_all)
        print(window_df_all)
        return window_df_all


###########################################################################
# transition probability matrix
###########################################################################

    def create_transition_probability_matrix(self, labels=None, frame_list=None, lag=None):
        """
        Build a row-normalized transition matrix from labels (no cross-replicate jumps).

        Parameters
        ----------
        labels : np.ndarray of int, optional
            Override stored labels. Integer states per frame, consecutive from 0.
        frame_list : list[int], optional
            Override stored frame_scale. Frames per replicate.
        lag : int, default 1
            Transition lag (frames).

        Returns
        -------
        np.ndarray
            (n_states+1, n_states+1) with header row/col for state ids.
            Also stored as self.transition_probability_matrix.

        Notes
        -----
        - Probability rows with zero outgoing counts are all zeros, excluding the header.
        - Prints raw counts pre-normalization for sanity check.
        """
        if labels is None:
            labels = self.labels
        if frame_list is None:
            frame_list = self.frame_scale
        if lag is None:
            lag = 1

        unique_states = np.unique(labels)
        number_of_states = len(unique_states)
        transtion_prob_matrix = np.zeros((number_of_states, number_of_states))

        iterator = 0
        for trajectory_length in frame_list:
            current_trajectory = labels[iterator:iterator + trajectory_length]
            iterator += trajectory_length
            if lag >= trajectory_length:
                continue
            for i in range(current_trajectory.shape[0] - lag):
                current_state = current_trajectory[i]
                next_state = current_trajectory[i + lag]
                transtion_prob_matrix[current_state, next_state] += 1

        row_sums = transtion_prob_matrix.sum(axis=1, keepdims=True)
        print(f"matrix counts before rownorm:\n{transtion_prob_matrix}")
        transition_probs = np.divide(
            transtion_prob_matrix,
            row_sums,
            out=np.zeros_like(transtion_prob_matrix),
            where=row_sums > 0,
        )

        final_transition_prob_matrix = np.zeros((number_of_states + 1, number_of_states + 1))
        final_transition_prob_matrix[1:, 1:] = transition_probs
        final_transition_prob_matrix[0, 1:], final_transition_prob_matrix[1:, 0] = unique_states, unique_states
        self.transition_probability_matrix = final_transition_prob_matrix
        return final_transition_prob_matrix

