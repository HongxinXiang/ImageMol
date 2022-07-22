import random
from collections import defaultdict
from typing import List, Set, Union, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def split_train_val_test_idx(idx, frac_train=0.8, frac_valid=0.1, frac_test=0.1, sort=False, seed=42):
    random.seed(seed)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    total = len(idx)

    train_idx, valid_idx = train_test_split(idx, test_size=frac_valid, shuffle=True, random_state=seed)
    train_idx, test_idx = train_test_split(train_idx, test_size=frac_test * total / (total - len(valid_idx)),
                                           shuffle=True, random_state=seed)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == total

    if sort:
        train_idx = sorted(train_idx)
        valid_idx = sorted(valid_idx)
        test_idx = sorted(test_idx)

    return train_idx, valid_idx, test_idx


def split_train_val_test_idx_stratified(idx, y, frac_train=0.8, frac_valid=0.1, frac_test=0.1, sort=False, seed=42):
    random.seed(seed)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    total = len(idx)

    train_idx, valid_idx, y_train, _ = train_test_split(idx, y, test_size=frac_valid, shuffle=True, stratify=y,
                                                        random_state=seed)
    train_idx, test_idx = train_test_split(train_idx, test_size=frac_test * total / (total - len(valid_idx)),
                                           shuffle=True, stratify=y_train, random_state=seed)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == total

    if sort:
        train_idx = sorted(train_idx)
        valid_idx = sorted(valid_idx)
        test_idx = sorted(test_idx)

    return train_idx, valid_idx, test_idx


def scaffold_split_train_val_test(index, smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, sort=False):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    index = np.array(index)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_index, val_index, test_index = index[train_idx], index[valid_idx], index[test_idx]

    if sort:
        train_index = sorted(train_index)
        val_index = sorted(val_index)
        test_index = sorted(test_index)

    return train_index, val_index, test_index


def random_scaffold_split_train_val_test(index, smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                                         sort=False, seed=42):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    index = np.array(index)

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(index)))
    n_total_test = int(np.floor(frac_test * len(index)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_index, val_index, test_index = index[train_idx], index[valid_idx], index[test_idx]

    if sort:
        train_index = sorted(train_index)
        val_index = sorted(val_index)
        test_index = sorted(test_index)

    return train_index, val_index, test_index


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        if Chem.MolFromSmiles(mol) != None:
            scaffold = generate_scaffold(mol)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split_balanced_train_val_test(index, smiles_list,
                                           frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                                           balanced: bool = False,
                                           seed: int = 0):
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    index = np.array(index)

    # Split
    train_size, val_size, test_size = frac_train * len(smiles_list), frac_valid * len(smiles_list), frac_test * len(
        smiles_list)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the smiles
    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()), key=lambda index_set: len(index_set), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1
    print(
        f'Total scaffolds = {len(scaffold_to_indices)} | train scaffolds = {train_scaffold_count} | val scaffolds = {val_scaffold_count} | test scaffolds = {test_scaffold_count}')

    train_idx = index[train]
    val_idx = index[val]
    test_idx = index[test]

    return train_idx, val_idx, test_idx
