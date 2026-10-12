import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
import os
import argparse

from repl.data.tlt12_wds import CIRCULAR_LATENT_INDICES, HUE_LATENT_INDICES



hue_indices = list(sorted(HUE_LATENT_INDICES))
circular_indices = list(sorted(CIRCULAR_LATENT_INDICES))
non_circular_indices = [i for i in range(10) if i not in circular_indices]


class ResultDicts:
    def __init__(self, method_names, splits, is_per_class=False,
                 rot_probe_thresh=None):
        """
        The ResultDicts object that stores the results of the evaluation for each method and split. It also stores the trained decoders for each method.

        Args:
            method_names (list): List of method names to evaluate.
            splits (list): List of splits to evaluate (e.g., ["train", "val", "test", "ood_test"]).
            is_per_class (bool): Whether to train per-class decoders.
            rot_probe_thresh (float): The threshold for the rotation probe. The default is None.

        Note:
            A postfix is added to the final file name based on the evaluation settings:
            - "_linear" is always added to indicate that linear decoders were used.
            - "_per_class" is added if per-class decoders were trained.
            - "_thresh_{rot_probe_thresh}" is added if a rotation probe threshold was specified
        """
        self.method_names = method_names
        self.splits = splits
        self.is_per_class = is_per_class
        self.initialize_method_dicts()

        self.file_name_postfix = "_linear"
        self.file_name_postfix += "_per_class" if self.is_per_class else ""
        self.file_name_postfix += f"_thresh_{rot_probe_thresh}" if rot_probe_thresh is not None else ""

    def initialize_method_dicts(self):
        """
        Initialize the dictionaries for storing the results and trained decoders for each method.
        """
        self.method_results = {method: {
            splt: [] for splt in self.splits
        } for method in self.method_names}
    
        self.method_decoders = {method: {} for method in self.method_names}
        for method in self.method_names:
            self.method_decoders[method]["lin_clf"] = []
            self.method_decoders[method]["ridge"] = []
            self.method_decoders[method]["ridge_rot_z"] = []
            self.method_decoders[method]["phases"] = []

    def add_empty_results_for_method(self, method_name):
        """
        Add empty lists for the results of a method.

        Args:
            method_name (str): The name of the method to add empty results for.
        """
        if self.is_per_class:
            self.method_decoders[method_name]["ridge"].append([])
            for splt in self.splits:
                self.method_results[method_name][splt].append([])

    def add_method_info_to_dict(self, method_name, results_dict):
        """
        Add the results and trained decoders for a method to the ResultDicts object.

        Args:
            method_name (str): The name of the method to add information for.
            results_dict (dict): A dictionary containing the results and trained decoders for the method.
        """
        if not self.is_per_class:
            self.method_decoders[method_name]["ridge"].append(results_dict["ridge_model"])
            self.method_decoders[method_name]["lin_clf"].append(results_dict["lin_clf_model"])
            self.method_decoders[method_name]["ridge_rot_z"].append(results_dict["ridge_rot_z"])
            self.method_decoders[method_name]["phases"].append(results_dict["phases"])
            for splt in self.splits:
                self.method_results[method_name][splt].append(results_dict["results"][splt])
        else:
            for i in range(len(results_dict)):
                self.method_decoders[method_name]["ridge"][-1].append(results_dict[i]["ridge_model"])
                for splt in self.splits:
                    self.method_results[method_name][splt][-1].append(results_dict[i]["results"][splt])

    def save_decoders(self, save_path, name=""):
        """
        Save the trained decoders for each method to a compressed .npz file.

        Args:
            save_path (str): The path to the directory where the decoders will be saved.
            name (str): An optional name to prepend to the file name. The default is an empty string.
        """
        file_name = name + self.file_name_postfix + "_decoders.npz"
        path = os.path.join(save_path, file_name)
        print(f"Saving trained decoders to {path}")
        np.savez_compressed(path, **self.method_decoders)

    def _get_results_df_per_class(self, factor_names):
        """
        Get a DataFrame containing the results for each method, split, seed, and synset (per class evaluation).

        Args:
            factor_names (list): List of factor names to include in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with columns for method, split, seed, synset, and the specified factor names.
        """
        columns = ["method", "split", "seed", "synset"] + factor_names
        df = pd.DataFrame(columns=columns)
        for method_name, splits in self.method_results.items():
            for split in splits:
                if split == []:
                    continue
                for seed in range(len(splits[split])):
                    for synset in range(len(splits[split][seed])):
                        if splits[split][seed][synset] == {}:
                            continue
                        row = [method_name, split, seed, synset]
                        for factor in factor_names:
                            row.append(splits[split][seed][synset][factor])
                        df.loc[len(df)] = row
        return df

    def get_results_df(self, factor_names):
        """
        Get a DataFrame containing the results for each method, split, and seed.

        Args:
            factor_names (list): List of factor names to include in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with columns for method, split, seed, and the specified factor names.
        """
        columns = ["method", "split", "seed"] + factor_names
        results_df = pd.DataFrame(columns=columns)
        for method_name, splits in self.method_results.items():
            for split in splits:
                if split == []:
                    continue
                for seed in range(len(splits[split])):
                    if splits[split][seed] == {}:
                        continue
                    row = [method_name, split, seed]
                    for factor in factor_names:
                        row.append(splits[split][seed][factor])
                    results_df.loc[len(results_df)] = row
        return results_df

    def save_results_to_csv(self, save_path, factor_names, name=""):
        """
        Save the results to a CSV file.

        Args:
            save_path (str): The path to the directory where the CSV file will be saved.
            factor_names (list): List of factor names to include in the CSV file.
            name (str): An optional name to prepend to the file name. The default is an empty string.
        """
        if self.is_per_class:
            df = self._get_results_df_per_class(factor_names)
        else:
            df = self.get_results_df(factor_names)
        file_name = name + self.file_name_postfix + "_results.csv"
        path = os.path.join(save_path, file_name)
        print(f"Saving results to {path}")
        df.to_csv(path, index=False)


def initialize_data_dict():
    return {
        "X_train": [],
        "y_train_clf": [],
        "y_train_regr": [],
        "X_val": [],
        "y_val_clf": [],
        "y_val_regr": [],
        "X_test": [],
        "y_test_clf": [],
        "y_test_regr": [],
        "X_ood_test": [],
        "y_ood_test_clf": [],
        "y_ood_test_regr": [],
    }


def load_data_from_folder(folder_path, method_names_list):
    """
    Load data from .npz files in the specified folder and organize it into a dictionary.

    Args:
        folder_path (str): The path to the folder containing the .npz files.
        method_names_list (list): A list of method names to filter the .npz files.

    Returns:
        dict: A dictionary containing the loaded data, organized by method.
    """
    method_data = {}
    splits = set()
    npz_files = [f for f in os.listdir(folder_path) if f.endswith(".npz")]

    for npz_file in npz_files:
        method_key = npz_file.split("_")[-2]
        if method_key not in method_names_list:
            continue
        data = np.load(os.path.join(folder_path, npz_file))
        if method_key not in method_data:
            method_data[method_key] = initialize_data_dict()
        for key in method_data[method_key]:
            if key in data:
                try:
                    method_data[method_key][key].append(data[key])
                except Exception as e:
                    print(f"Error appending data for key '{key}' in {npz_file}: {e}")
                splits.add(key.split("_")[1])
            else:
                print(f"Warning: Key '{key}' not found in {npz_file}.")
    
    return method_data


def get_delta_features(latent_values):
    """
    Given a 2D array of latent values, return a 2D array of delta features.
    """
    latent_values = latent_values.reshape((-1, 32, latent_values.shape[-1]))
    delta_features = np.diff(latent_values, axis=1)
    # pad the first time step with zeros to maintain the same shape
    delta_features = np.pad(delta_features, ((0, 0), (1, 0), (0, 0)),
                            mode='constant', constant_values=0)
    delta_features = delta_features.reshape((-1, latent_values.shape[-1]))
    return delta_features


def get_regression_true_targets(y, thresholded_synsets=None):
    y_regr = y.copy()
    y_regr[:, hue_indices] *= 2 * np.pi
    y_circular = y_regr[:, circular_indices]
    y_sin = np.sin(y_circular)
    y_cos = np.cos(y_circular)
    y_deltas_non_circular = get_delta_features(y[:, non_circular_indices])
    y_deltas_circular = get_delta_features(y_circular)
    y_deltas_sin = np.sin(y_deltas_circular)
    y_deltas_cos = np.cos(y_deltas_circular)

    if thresholded_synsets is not None:
        y_regr = np.concatenate([
            y[:, [0, 1]], # rot_x, rot_y
            y_deltas_non_circular[:, [0, 1]], # delta_trans_x, delta_trans_y
            y_deltas_sin[:, [0]], # delta_rot_z_sin
            y_deltas_cos[:, [0]], # delta_rot_z_cos
            y_sin[:, [0]], # rot_z_sin
            y_cos[:, [0]], # rot_z_cos
        ], axis=-1)
        factor_names = [
            "rot_x", "rot_y", "delta_rot_x", "delta_rot_y",
            "delta_rot_z_sin", "delta_rot_z_cos", "rot_z_sin", "rot_z_cos"
        ]
    else:
        y_regr = np.concatenate([
            y[:, non_circular_indices],
            y_sin[:, 1:], # exclude rot_z
            y_cos[:, 1:], # exclude rot_z
            y_deltas_non_circular,
            y_deltas_sin,
            y_deltas_cos,
            y_sin[:, [0]], # rot_z
            y_cos[:, [0]], # rot_z
        ], axis=-1)
        factor_names = [
            "rot_x", "rot_y", "spot_theta", "trans_x", "trans_y", "trans_z",
            "floor_hue_sin", "spot_phi_sin", "spot_hue_sin",
            "floor_hue_cos", "spot_phi_cos", "spot_hue_cos",
            "delta_rot_x", "delta_rot_y", "delta_spot_theta", "delta_trans_x", "delta_trans_y", "delta_trans_z",
            "delta_rot_z_sin", "delta_floor_hue_sin", "delta_spot_phi_sin", "delta_spot_hue_sin",
            "delta_rot_z_cos", "delta_floor_hue_cos", "delta_spot_phi_cos", "delta_spot_hue_cos",
            "rot_z_sin", "rot_z_cos"
        ]
    return y_regr, factor_names


def rotation_matrix(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])


def update_phase(C_k, T_k, ridge):
    """
    Update the phase for a given class using the current ridge regression model.
    """
    y_hat_k = ridge.predict(C_k)
    A_k = y_hat_k.T @ T_k
    phi_k = np.arctan2(A_k[0, 1] - A_k[1, 0], A_k[0, 0] + A_k[1, 1])
    return phi_k


def train_rotation_probe(X, y_clf, y_sin_cos, categories):
    """
    Train a rotation probe with per-class phase correction using ridge regression.

    Args:
        X (np.ndarray): The input representations of shape (n_samples, n_features).
        y_clf (np.ndarray): The class labels of shape (n_samples,).
        y_sin_cos (np.ndarray): The target values for the rotation probe of shape (n_samples, 2).
        categories (list): The list of unique class labels.

    Returns:
        ridge (Ridge): The trained ridge regression model.
        phases (np.ndarray): The learned phase corrections for each class of shape (n_classes,).
    """
    n_classes = len(categories)
    print(f"Training rotation probe for {n_classes} classes...")
    C_k = [X[y_clf == k] for k in categories]
    T_k = [y_sin_cos[y_clf == k] for k in categories]

    # initialization
    phases = np.zeros((n_classes,), dtype=np.float32)
    R_phi_k = [rotation_matrix(phases[k]) for k in range(n_classes)]
    T_tilde_k = [T_k[k] @ R_phi_k[k] for k in range(n_classes)]
    C = np.concatenate(C_k, axis=0)
    T_tilde = np.concatenate(T_tilde_k, axis=0)
    
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(C, T_tilde)
    print(f"Initial ridge score: {ridge.score(C, T_tilde):.4f}")

    for i in range(20):
        # step A: phase update
        for k in range(n_classes):
            phases[k] = update_phase(C_k[k], T_k[k], ridge)
            R_phi_k[k] = rotation_matrix(phases[k])
            T_tilde_k[k] = T_k[k] @ R_phi_k[k]
        # step B: ridge regression update
        T_tilde = np.concatenate(T_tilde_k, axis=0)
        ridge.fit(C, T_tilde)
        print(f"Ridge score after iteration {i + 1}: {ridge.score(C, T_tilde):.4f}")
    return ridge, phases


def evaluate_rotation_probe(X, y_clf, y_sin_cos, ridge, phases, categories):
    """
    Evaluate the rotation probe with per-class phase correction using ridge regression.
    """
    n_classes = len(categories)
    R_phi_k = [rotation_matrix(phases[k]) for k in range(n_classes)]
    T_tilde_k = [y_sin_cos[y_clf == categories[k]] @ R_phi_k[k] for k in range(n_classes)]
    C_k = [X[y_clf == categories[k]] for k in range(n_classes)]
    per_class_r2 = []
    for k in range(n_classes):
        y_hat_k = ridge.predict(C_k[k])
        r2_k = r2_score(T_tilde_k[k], y_hat_k)
        per_class_r2.append(r2_k)
    T_tilde = np.concatenate(T_tilde_k, axis=0)
    C = np.concatenate(C_k, axis=0)
    y_hat = ridge.predict(C)
    r2 = r2_score(T_tilde, y_hat, multioutput="raw_values")
    return r2, per_class_r2



def train_models(X, y_clf, y_regr, is_one_class=False, categories=None):
    """
    Train linear classification and regression models, as well as phase-corrected rotation probe for rot_z.

    Args:
        X (np.ndarray): The input representations of shape (n_samples, n_features).
        y_clf (np.ndarray): The class labels of shape (n_samples,).
        y_regr (np.ndarray): The target values for regression of shape (n_samples, n_targets).
        is_one_class (bool): Whether to train per-class models.
        categories (list): The list of unique class labels.

    Returns:
        clf_model (LogisticRegression): The trained linear classification model.
        ridge_model (Ridge): The trained linear regression model.
        ridge_rot_z (Ridge): The trained phase-corrected rotation probe for rot_z.
        phases (np.ndarray): The learned phase corrections for each class of shape (n_classes,).
    """
    clf_model, ridge_model, ridge_rot_z, phases = None, None, None, None

    # Train linear classification model
    if not is_one_class:
        clf_model = LogisticRegression(max_iter=1000)
        clf_model.fit(X, y_clf[:, 0])

    # Train linear regression model
    ridge_model = Ridge(alpha=1.0)
    if is_one_class:
        ridge_model.fit(X, y_regr)
    else:
        ridge_model.fit(X, y_regr[:, :-2])  # exclude rot_z_sin and rot_z_cos

        # Train rotation probe for rot_z
        ridge_rot_z, phases = train_rotation_probe(X, 
                                                y_clf[:, 0], 
                                                y_regr[:, -2:],
                                                categories)  # only rot_z_sin and rot_z_cos
    return clf_model, ridge_model, ridge_rot_z, phases



def evaluate_models(X, y_clf, y_regr, factor_names, ridge_model, clf_model=None, 
                    ridge_rot_z=None, phases=None, categories=None, is_per_class=False):
    """
    Evaluate the trained models on the given data.

    Args:
        X (np.ndarray): The input representations of shape (n_samples, n_features).
        y_clf (np.ndarray): The class labels of shape (n_samples,).
        y_regr (np.ndarray): The target values for regression of shape (n_samples, n_targets).
        factor_names (list): The list of factor names.
        ridge_model (Ridge): The trained linear regression model.
        clf_model (LogisticRegression): The trained linear classification model.
        ridge_rot_z (Ridge): The trained phase-corrected rotation probe for rot_z.
        phases (np.ndarray): The learned phase corrections for each class of shape (n_classes,).
        categories (list): The list of unique class labels.
        is_per_class (bool): Whether to evaluate a per-class model.

    Returns:
        results (dict): A dictionary containing the evaluation results.
    """
    results = {}
    # Classification accuracy
    if clf_model is not None:
        y_clf_pred = clf_model.predict(X)
        clf_accuracy = accuracy_score(y_clf, y_clf_pred)
        results["synset"] = clf_accuracy

    # Regression R^2 scores
    y_regr_pred = ridge_model.predict(X)
    if is_per_class:
        start_idx = 0
        end_idx = len(factor_names)
    else:
        start_idx = 1  # skip synset
        end_idx = -2  # skip rot_z_sin and rot_z_cos
    regr_r2 = r2_score(y_regr[:, :end_idx], y_regr_pred, multioutput="raw_values")
    for j, factor in enumerate(factor_names[start_idx:end_idx]):
        results[factor] = regr_r2[j]

    # evaluate rot_z
    if ridge_rot_z is not None and phases is not None:
        if categories is None:
            print("Using all categories for phase-corrected rotation probe evaluation.")
            categories = np.unique(y_clf[:, 0])
        regr_r2_rot_z, per_class_r2_rot_z = evaluate_rotation_probe(X, y_clf[:, 0],
                                                    y_regr[:, -2:],  # only rot_z_sin and rot_z_cos
                                                    ridge_rot_z,
                                                    phases,
                                                    categories)
        results["rot_z_sin"] = regr_r2_rot_z[0]
        results["rot_z_cos"] = regr_r2_rot_z[1]
        results["per_class_r2_rot_z"] = per_class_r2_rot_z
    else:
        print("Skipping phase-corrected rotation probe.")
    return results


def get_filtered_data(X, y_clf, y_regr, thresholded_synsets):
    """
    Filter the data based on the specified thresholded synsets, i.e., only 
    include samples with synset labels in the thresholded_synsets list.

    Args:
        X (np.ndarray): The input representations of shape (n_samples, n_features).
        y_clf (np.ndarray): The class labels of shape (n_samples,).
        y_regr (np.ndarray): The target values for regression of shape (n_samples, n_targets).
        thresholded_synsets (list): A list of synset labels to include in the filtered data. If None, no filtering is applied.

    Returns:
        X_filtered (np.ndarray): The filtered input representations.
        y_clf_filtered (np.ndarray): The filtered class labels.
        y_regr_filtered (np.ndarray): The filtered target values for regression.
    """
    if thresholded_synsets is not None:
        mask = np.isin(y_clf[:, 0], thresholded_synsets)
        X_filtered = X[mask]
        y_clf_filtered = y_clf[mask]
        y_regr_filtered = y_regr[mask]
    else:
        X_filtered, y_clf_filtered, y_regr_filtered = X, y_clf, y_regr
    return X_filtered, y_clf_filtered, y_regr_filtered


def train_and_evaluate_models(X_train, y_clf_train, y_regr_train,
                              X_val, y_clf_val, y_regr_val,
                              X_test, y_clf_test, y_regr_test,
                              X_ood_test=None, y_clf_ood_test=None, y_regr_ood_test=None,
                              factor_names=None, is_per_class=False,
                              thresholded_synsets=None):
    """
    Train and evaluate linear classification and regression models, as well as phase-corrected rotation probe for rot_z.
    The evaluation is performed on all the provided splits.

    Args:
        X_train (np.ndarray): The input representations for the training set of shape (n_samples, n_features).
        y_clf_train (np.ndarray): The class labels for the training set of shape (n_samples,).
        y_regr_train (np.ndarray): The target values for regression for the training set of shape (n_samples, n_targets).
        X_val (np.ndarray): The input representations for the validation set of shape (n_samples, n_features).
        y_clf_val (np.ndarray): The class labels for the validation set of shape (n_samples,).
        y_regr_val (np.ndarray): The target values for regression for the validation set of shape (n_samples, n_targets).
        X_test (np.ndarray): The input representations for the test set of shape (n_samples, n_features).
        y_clf_test (np.ndarray): The class labels for the test set of shape (n_samples,).
        y_regr_test (np.ndarray): The target values for regression for the test set of shape (n_samples, n_targets).
        X_ood_test (np.ndarray): The input representations for the out-of-distribution test set of shape (n_samples, n_features). Default is None.
        y_clf_ood_test (np.ndarray): The class labels for the out-of-distribution test set of shape (n_samples,). Default is None.
        y_regr_ood_test (np.ndarray): The target values for regression for the out-of-distribution test set of shape (n_samples, n_targets). Default is None.
        factor_names (list): The list of factor names.
        is_per_class (bool): Whether to train per-class models. Default is False.
        thresholded_synsets (list): A list of synset labels to include in the evaluation. If None, no filtering is applied. Default is None.

    Returns:
        output_dict (dict): A dictionary containing the trained models and evaluation results.
    """
    
    categories = np.unique(y_clf_train[:, 0])

    if thresholded_synsets is not None:
        categories = np.array(thresholded_synsets)

    X_train, y_clf_train, y_regr_train = get_filtered_data(X_train,
                                                           y_clf_train,
                                                           y_regr_train,
                                                           thresholded_synsets)
    X_val, y_clf_val, y_regr_val = get_filtered_data(X_val,
                                                     y_clf_val,
                                                     y_regr_val,
                                                     thresholded_synsets)
    X_test, y_clf_test, y_regr_test = get_filtered_data(X_test,
                                                        y_clf_test,
                                                        y_regr_test,
                                                        thresholded_synsets)
    if X_ood_test is not None:
        X_ood_test, y_clf_ood_test, y_regr_ood_test = get_filtered_data(
            X_ood_test, y_clf_ood_test, y_regr_ood_test, thresholded_synsets)

    lin_clf_model, ridge_model, ridge_rot_z, phases = train_models(
        X_train, y_clf_train, y_regr_train,
        is_one_class=is_per_class, categories=categories)

    # evaluate models on train, val, and test sets
    results = {"train": {}, "val": {}, "test": {}, "ood_test": {}}
    r2_rot_z_per_synset = {"train": [], "val": [], "test": [], "ood_test": []}
    for split, X, y_clf, y_regr in [("train", X_train, y_clf_train, y_regr_train),
                                    ("val", X_val, y_clf_val, y_regr_val),
                                    ("test", X_test, y_clf_test, y_regr_test),
                                    ("ood_test", X_ood_test, y_clf_ood_test, y_regr_ood_test)]:
        if X is None:
            continue
        # Classification accuracy
        results[split] = evaluate_models(X, y_clf, y_regr, factor_names,
                                        ridge_model, lin_clf_model,
                                        ridge_rot_z, phases, categories, is_per_class)

    output_dict = {
        "ridge_model": ridge_model,
        "results": results
    }
    if not is_per_class:
        output_dict.update({
            "lin_clf_model": lin_clf_model,
            "ridge_rot_z": ridge_rot_z,
            "phases": phases,
            "r2_rot_z_per_synset": r2_rot_z_per_synset
        })
    return output_dict


def train_and_evaluate_per_class_models(X_train, y_clf_train, y_regr_train,
                                        X_val, y_clf_val, y_regr_val,
                                        X_test, y_clf_test, y_regr_test,
                                        X_ood_test=None, y_clf_ood_test=None, y_regr_ood_test=None,
                                        factor_names=None):
    """
    Train and evaluate linear classification and regression models for each class separately.
    
    Args:
        X_train (np.ndarray): The input representations for the training set of shape (n_samples, n_features).
        y_clf_train (np.ndarray): The class labels for the training set of shape (n_samples,).
        y_regr_train (np.ndarray): The target values for regression for the training set of shape (n_samples, n_targets).
        X_val (np.ndarray): The input representations for the validation set of shape (n_samples, n_features).
        y_clf_val (np.ndarray): The class labels for the validation set of shape (n_samples,).
        y_regr_val (np.ndarray): The target values for regression for the validation set of shape (n_samples, n_targets).
        X_test (np.ndarray): The input representations for the test set of shape (n_samples, n_features).
        y_clf_test (np.ndarray): The class labels for the test set of shape (n_samples,).
        y_regr_test (np.ndarray): The target values for regression for the test set of shape (n_samples, n_targets).
        X_ood_test (np.ndarray): The input representations for the out-of-distribution test set of shape (n_samples, n_features). Default is None.
        y_clf_ood_test (np.ndarray): The class labels for the out-of-distribution test set of shape (n_samples,). Default is None.
        y_regr_ood_test (np.ndarray): The target values for regression for the out-of-distribution test set of shape (n_samples, n_targets). Default is None.
        factor_names (list): The list of factor names.

    Returns:
        output (list): A list of dictionaries containing the trained models and evaluation results for each class.
    """
    synset_categories = np.unique(y_clf_train[:, 0])
    n_classes = len(synset_categories)
    output = [
        {
            "ridge_model": None,
            "results": {splt: None for splt in ["train", "val", "test", "ood_test"]},
        } for _ in range(n_classes)
    ]
    for k in range(n_classes):
        mask_train = y_clf_train[:, 0] == k
        mask_val = y_clf_val[:, 0] == k
        mask_test = y_clf_test[:, 0] == k
        if X_ood_test is not None:
            mask_ood_test = y_clf_ood_test[:, 0] == k

        output[k] = train_and_evaluate_models(
            X_train[mask_train], y_clf_train[mask_train], y_regr_train[mask_train],
            X_val[mask_val], y_clf_val[mask_val], y_regr_val[mask_val],
            X_test[mask_test], y_clf_test[mask_test], y_regr_test[mask_test],
            X_ood_test[mask_ood_test] if X_ood_test is not None else None,
            y_clf_ood_test[mask_ood_test] if y_clf_ood_test is not None else None,
            y_regr_ood_test[mask_ood_test] if y_regr_ood_test is not None else None,
            factor_names=factor_names, is_per_class=True
        )
        output[k]["lin_clf_model"] = None
    return output


def get_thresholded_synsets(dir_path, threshold, name=""):
    """
    Get the list of synsets that have a mean R^2 score for rot_z above the specified threshold.
    If there is an existing CSV file with per-class rotation probe results, 
    it will be loaded and filtered for the validation split and the "sup" method to find the synsets to be included.
    If no such file exists, a default list of synsets (removing earphones only) will be returned.

    Args:
        dir_path (str): The path to the directory containing the CSV file with per-class rotation probe results.
        threshold (float): The R^2 score threshold for including synsets.
        name (str): An optional name to prepend to the CSV file name. The default is an empty string.

    Returns:
        list: A list of synset IDs that have a mean R^2 score for rot_z above the specified threshold, or a default list of synsets if no CSV file is found.
    """
    # load existing per-class rotation probe results from a CSV file
    if os.path.exists(os.path.join(dir_path, f"{name}_linear_per_class_results.csv")):
        per_class_rot_df = pd.read_csv(os.path.join(dir_path, f"{name}_linear_per_class_results.csv"))
        # filter for validation split
        per_class_rot_df = per_class_rot_df[per_class_rot_df["split"] == "val"]
        try:
            per_class_rot_df = per_class_rot_df[per_class_rot_df["method"] == "sup"]
        except KeyError:
            print("No existing per-class rotation probe results found for sup. Please run the file with --per_class first, making sure sup is included.")
            import sys
            sys.exit(1)
        
        per_class_rot_df["rot_z"] = per_class_rot_df[["rot_z_sin", "rot_z_cos"]].mean(axis=1)
        per_class_mean_rot_z = per_class_rot_df.groupby("synset")["rot_z"].mean().reset_index()
        thresholded_synsets = per_class_mean_rot_z[per_class_mean_rot_z["rot_z"] > threshold].loc[:, "synset"].tolist()
    else:
        print("No existing per-class rotation probe results found. Removing earphones only.")
        thresholded_synsets = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11] # remove earphones only
    return thresholded_synsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr_path", type=str, required=True, help="Path to the saved representations and labels")
    parser.add_argument("--output_file_name", type=str, default="", help="Name of the output CSV file")
    parser.add_argument("--methods", nargs="+", default=None, help="List of methods to evaluate. Use the method names that were used when saving the representations (e.g., sup, RPL, etc.).")
    parser.add_argument("--per_class", action="store_true", help="train per-class decoders")
    parser.add_argument("--rot_probe_threshold", type=float, default=None, help="Threshold for rotation probe evaluation")

    args = parser.parse_args()

    assert args.methods is not None, "Please specify at least one method to evaluate using --methods"

    DIR = args.repr_path
    file_name = args.output_file_name

    method_names = args.methods
    splits = ["train", "val", "test", "ood_test"]
    method_results_dicts = ResultDicts(method_names, splits=splits, 
                                       is_per_class=args.per_class,
                                       rot_probe_thresh=args.rot_probe_threshold)
    method_data = load_data_from_folder(DIR, method_names)

    thresholded_synsets = None
    if args.rot_probe_threshold is not None:
        thresholded_synsets = get_thresholded_synsets(DIR,
                                                      args.rot_probe_threshold,
                                                      name=file_name)


    for method_name, data in method_data.items():
        for split in splits:
            for seed in range(len(data[f"X_{split}"])):
                y_true, factor_names = get_regression_true_targets(
                    data[f"y_{split}_regr"][seed], 
                    thresholded_synsets=thresholded_synsets)
                
                if not args.per_class: # include synset in factor names
                    factor_names = ["synset"] + factor_names
                if f"y_{split}_true" not in data:
                    data[f"y_{split}_true"] = []
                data[f"y_{split}_true"].append(y_true)

        n_seeds = len(data["X_train"])

        for i in range(n_seeds):
            X_train = data["X_train"][i]
            y_clf_train = data["y_train_clf"][i]
            y_true_train = data["y_train_true"][i]
            X_val = data["X_val"][i]
            y_clf_val = data["y_val_clf"][i]
            y_true_val = data["y_val_true"][i]
            X_test = data["X_test"][i]
            y_clf_test = data["y_test_clf"][i]
            y_true_test = data["y_test_true"][i]
            try:
                X_ood_test = data["X_ood_test"][i]
                y_clf_ood_test = data["y_ood_test_clf"][i]
                y_true_ood_test = data["y_ood_test_true"][i]
            except IndexError:
                X_ood_test = y_clf_ood_test = y_true_ood_test = None
            
            if not args.per_class:
                results_dict = train_and_evaluate_models(
                    X_train, y_clf_train, y_true_train,
                    X_val, y_clf_val, y_true_val,
                    X_test, y_clf_test, y_true_test,
                    X_ood_test=X_ood_test,
                    y_clf_ood_test=y_clf_ood_test,
                    y_regr_ood_test=y_true_ood_test,
                    factor_names=factor_names,
                    is_per_class=False,
                    thresholded_synsets=thresholded_synsets
                )

            else:
                results_dict = train_and_evaluate_per_class_models(
                    X_train, y_clf_train, y_true_train,
                    X_val, y_clf_val, y_true_val,
                    X_test, y_clf_test, y_true_test,
                    X_ood_test=X_ood_test,
                    y_clf_ood_test=y_clf_ood_test,
                    y_regr_ood_test=y_true_ood_test,
                    factor_names=factor_names
                )
                method_results_dicts.add_empty_results_for_method(method_name)
            method_results_dicts.add_method_info_to_dict(method_name, results_dict)

    method_results_dicts.save_results_to_csv(DIR, factor_names, name=file_name)
    method_results_dicts.save_decoders(DIR, name=file_name)

if __name__ == "__main__":
    main()
