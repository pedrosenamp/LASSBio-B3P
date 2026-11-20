import copy
import random
import numpy as np
import pandas as pd
import knime.scripting.io as knio

import torch
import torch.nn as nn
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from rdkit import Chem
import rdkit

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ParameterGrid,
)

from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

ID_COLUMN = "Old RowID"
SMILES_COLUMN = "SMILES"
TARGET_COLUMN = "Activity"

POSITIVE_CLASS = "active"
NEGATIVE_CLASS = "inactive"


# ============================================================
# CROSS-VALIDATION
# ============================================================

N_SPLITS = 5

TUNING_CV_SEED = 2026
INTERNAL_CV_SEED = 2027
FINAL_MODEL_SEED = 2028


# ============================================================
# TRAINING
# ============================================================

MAX_EPOCHS = 200
PATIENCE = 20
EARLY_STOP_FRACTION = 0.10
MIN_DELTA = 1e-4


# ============================================================
# CLASS IMBALANCE
# ============================================================

CLASS_WEIGHTING = True


# ============================================================
# OPTIMIZER
# ============================================================

WEIGHT_DECAY = 1e-5


# ============================================================
# CLASSIFICATION THRESHOLD
# ============================================================

DECISION_THRESHOLD = 0.50


# ============================================================
# WINDOWS / KNIME
# ============================================================

NUM_WORKERS = 0


# ============================================================
# DEVICE / CUDA
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PIN_MEMORY = DEVICE.type == "cuda"
NON_BLOCKING = DEVICE.type == "cuda"

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


# ============================================================
# HYPERPARAMETER GRID
#
# 2 x 2 x 2 x 2 x 1 = 16 configurations
# 16 x 5 folds = 80 training fits during tuning
# ============================================================

TUNING_GRID = {

    "num_layers": [
        2,
        3,
    ],

    "hidden_dim": [
        64,
        128,
    ],

    "dropout": [
        0.20,
        0.40,
    ],

    "learning_rate": [
        1e-3,
        3e-4,
    ],

    "batch_size": [
        128,
    ],
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# MOLECULAR GRAPH FEATURIZATION
# ============================================================

ATOM_TYPES = [
    1,   # H
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
]


ATOM_DEGREES = [
    0,
    1,
    2,
    3,
    4,
    5,
]


FORMAL_CHARGES = [
    -2,
    -1,
    0,
    1,
    2,
]


TOTAL_H = [
    0,
    1,
    2,
    3,
    4,
]


HYBRIDIZATIONS = [

    Chem.rdchem.HybridizationType.SP,

    Chem.rdchem.HybridizationType.SP2,

    Chem.rdchem.HybridizationType.SP3,

    Chem.rdchem.HybridizationType.SP3D,

    Chem.rdchem.HybridizationType.SP3D2,

    Chem.rdchem.HybridizationType.UNSPECIFIED,
]


CHIRAL_TAGS = [

    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,

    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,

    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]


BOND_TYPES = [

    Chem.rdchem.BondType.SINGLE,

    Chem.rdchem.BondType.DOUBLE,

    Chem.rdchem.BondType.TRIPLE,

    Chem.rdchem.BondType.AROMATIC,
]


BOND_STEREO = [

    Chem.rdchem.BondStereo.STEREONONE,

    Chem.rdchem.BondStereo.STEREOANY,

    Chem.rdchem.BondStereo.STEREOZ,

    Chem.rdchem.BondStereo.STEREOE,

    Chem.rdchem.BondStereo.STEREOCIS,

    Chem.rdchem.BondStereo.STEREOTRANS,
]


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def one_hot_other(value, choices):

    output = [
        float(value == item)
        for item in choices
    ]

    # Additional "other" category
    output.append(
        float(value not in choices)
    )

    return output


# ============================================================
# ATOM FEATURES
#
# Atomic number        = 13
# Degree               = 7
# Formal charge        = 6
# Hybridization        = 7
# Total H              = 6
# Aromaticity          = 1
# Ring membership      = 1
# Chirality            = 4
#
# TOTAL = 45
# ============================================================

def atom_features(atom):

    return (

        one_hot_other(
            atom.GetAtomicNum(),
            ATOM_TYPES,
        )

        +

        one_hot_other(
            atom.GetDegree(),
            ATOM_DEGREES,
        )

        +

        one_hot_other(
            atom.GetFormalCharge(),
            FORMAL_CHARGES,
        )

        +

        one_hot_other(
            atom.GetHybridization(),
            HYBRIDIZATIONS,
        )

        +

        one_hot_other(
            atom.GetTotalNumHs(),
            TOTAL_H,
        )

        +

        [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ]

        +

        one_hot_other(
            atom.GetChiralTag(),
            CHIRAL_TAGS,
        )
    )


# ============================================================
# BOND FEATURES
#
# Bond type            = 5
# Conjugation          = 1
# Aromaticity          = 1
# Ring membership      = 1
# Stereochemistry      = 7
#
# TOTAL = 15
# ============================================================

def bond_features(bond):

    return (

        one_hot_other(
            bond.GetBondType(),
            BOND_TYPES,
        )

        +

        [
            float(bond.GetIsConjugated()),
            float(bond.GetIsAromatic()),
            float(bond.IsInRing()),
        ]

        +

        one_hot_other(
            bond.GetStereo(),
            BOND_STEREO,
        )
    )


# ============================================================
# DETERMINE FEATURE DIMENSIONS
# ============================================================

_test_atom_mol = Chem.MolFromSmiles("C")
_test_bond_mol = Chem.MolFromSmiles("CC")


NODE_DIM = len(
    atom_features(
        _test_atom_mol.GetAtomWithIdx(0)
    )
)


EDGE_DIM = len(
    bond_features(
        _test_bond_mol.GetBondWithIdx(0)
    )
)


# ============================================================
# SMILES -> PYTORCH GEOMETRIC GRAPH
# ============================================================

def smiles_to_graph(
    smiles,
    label,
):

    if (
        not isinstance(smiles, str)
        or
        not smiles.strip()
    ):

        raise ValueError(
            "Empty or non-string SMILES."
        )


    mol = Chem.MolFromSmiles(smiles)


    if mol is None:

        raise ValueError(
            "RDKit could not parse SMILES."
        )


    Chem.AssignStereochemistry(
        mol,
        cleanIt=True,
        force=True,
    )


    # --------------------------------------------------------
    # NODE FEATURES
    # --------------------------------------------------------

    x = torch.tensor(

        [
            atom_features(atom)
            for atom in mol.GetAtoms()
        ],

        dtype=torch.float32,
    )


    # --------------------------------------------------------
    # EDGES
    #
    # Each chemical bond is represented in both directions.
    # --------------------------------------------------------

    source = []
    target = []
    edge_attributes = []


    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        features = bond_features(bond)


        source.extend([
            i,
            j,
        ])

        target.extend([
            j,
            i,
        ])

        edge_attributes.extend([
            features,
            features,
        ])


    if source:

        edge_index = torch.tensor(
            [
                source,
                target,
            ],
            dtype=torch.long,
        )


        edge_attr = torch.tensor(
            edge_attributes,
            dtype=torch.float32,
        )


    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )


        edge_attr = torch.empty(
            (0, EDGE_DIM),
            dtype=torch.float32,
        )


    y = torch.tensor(
        [float(label)],
        dtype=torch.float32,
    )


    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )


# ============================================================
# GINE CLASSIFIER
# ============================================================

class GINEClassifier(nn.Module):

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):

        super().__init__()

        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = node_dim


        for _ in range(int(num_layers)):

            mlp = nn.Sequential(

                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),

                nn.ReLU(),

                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
            )


            conv = GINEConv(
                nn=mlp,
                train_eps=True,
                edge_dim=edge_dim,
            )


            self.convs.append(conv)

            self.norms.append(
                nn.LayerNorm(hidden_dim)
            )

            input_dim = hidden_dim


        self.head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                self.dropout
            ),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )


    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):

        for conv, norm in zip(
            self.convs,
            self.norms,
        ):

            x = conv(
                x,
                edge_index,
                edge_attr,
            )

            x = norm(x)

            x = torch.relu(x)

            x = nn.functional.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )


        # Atom embeddings -> molecular embedding
        x = global_mean_pool(
            x,
            batch,
        )


        return self.head(x).view(-1)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(params):

    model = GINEClassifier(

        node_dim=NODE_DIM,

        edge_dim=EDGE_DIM,

        hidden_dim=int(
            params["hidden_dim"]
        ),

        num_layers=int(
            params["num_layers"]
        ),

        dropout=float(
            params["dropout"]
        ),
    )


    return model.to(DEVICE)


# ============================================================
# DATA LOADER
# ============================================================

def make_loader(
    graphs,
    indices,
    batch_size,
    shuffle,
    seed,
):

    generator = torch.Generator()

    generator.manual_seed(
        int(seed)
    )


    subset = [
        graphs[int(i)]
        for i in indices
    ]


    return DataLoader(

        subset,

        batch_size=int(
            batch_size
        ),

        shuffle=shuffle,

        num_workers=NUM_WORKERS,

        generator=generator,

        pin_memory=PIN_MEMORY,
    )


# ============================================================
# LOSS FUNCTION
# ============================================================

def make_loss(
    labels,
    indices,
):

    subset_y = labels[
        np.asarray(
            indices,
            dtype=int,
        )
    ]


    n_active = int(
        (subset_y == 1).sum()
    )


    n_inactive = int(
        (subset_y == 0).sum()
    )


    if (
        n_active == 0
        or
        n_inactive == 0
    ):

        raise ValueError(
            "A training subset contains only one class."
        )


    if CLASS_WEIGHTING:

        positive_weight = (
            n_inactive
            /
            n_active
        )


        pos_weight = torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=DEVICE,
        )


        return nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )


    return nn.BCEWithLogitsLoss()


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()


    for batch in loader:

        batch = batch.to(
            DEVICE,
            non_blocking=NON_BLOCKING,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )


        target = (
            batch.y
            .view(-1)
            .float()
        )


        loss = criterion(
            logits,
            target,
        )


        loss.backward()

        optimizer.step()


# ============================================================
# MEAN VALIDATION LOSS
# ============================================================

@torch.no_grad()
def mean_loss(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    total_n = 0


    for batch in loader:

        batch = batch.to(
            DEVICE,
            non_blocking=NON_BLOCKING,
        )


        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )


        target = (
            batch.y
            .view(-1)
            .float()
        )


        loss = criterion(
            logits,
            target,
        )


        n = int(
            target.numel()
        )


        total_loss += (
            float(loss.item())
            *
            n
        )


        total_n += n


    return (
        total_loss
        /
        max(
            total_n,
            1,
        )
    )


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

@torch.no_grad()
def predict_proba(
    model,
    graphs,
    indices,
    batch_size,
    seed,
):

    loader = make_loader(

        graphs=graphs,

        indices=indices,

        batch_size=batch_size,

        shuffle=False,

        seed=seed,
    )


    model.eval()

    probabilities = []


    for batch in loader:

        batch = batch.to(
            DEVICE,
            non_blocking=NON_BLOCKING,
        )


        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )


        probability = (

            torch.sigmoid(logits)

            .cpu()

            .numpy()
        )


        probabilities.append(
            probability
        )


    return (
        np.concatenate(
            probabilities
        )
        .astype(float)
    )


# ============================================================
# TRAIN WITH EARLY STOPPING
# ============================================================

def fit_with_early_stopping(
    graphs,
    labels,
    fit_indices,
    early_stop_indices,
    params,
    seed,
):

    set_seed(seed)


    model = build_model(params)


    criterion = make_loss(
        labels,
        fit_indices,
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=float(
            params["learning_rate"]
        ),

        weight_decay=WEIGHT_DECAY,
    )


    training_loader = make_loader(

        graphs=graphs,

        indices=fit_indices,

        batch_size=params[
            "batch_size"
        ],

        shuffle=True,

        seed=seed,
    )


    early_stop_loader = make_loader(

        graphs=graphs,

        indices=early_stop_indices,

        batch_size=params[
            "batch_size"
        ],

        shuffle=False,

        seed=seed,
    )


    best_loss = np.inf
    best_epoch = 1
    best_state = None

    epochs_without_improvement = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
        )


        validation_loss = mean_loss(
            model,
            early_stop_loader,
            criterion,
        )


        if (
            validation_loss
            <
            best_loss - MIN_DELTA
        ):

            best_loss = validation_loss

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0


        else:

            epochs_without_improvement += 1


        if (
            epochs_without_improvement
            >=
            PATIENCE
        ):

            break


    if best_state is None:

        raise RuntimeError(
            "Training failed to obtain "
            "a valid early-stopping state."
        )


    model.load_state_dict(
        best_state
    )


    return model, best_epoch


# ============================================================
# FINAL TRAINING WITH FIXED NUMBER OF EPOCHS
# ============================================================

def fit_fixed_epochs(
    graphs,
    labels,
    indices,
    params,
    epochs,
    seed,
):

    set_seed(seed)


    model = build_model(params)


    criterion = make_loss(
        labels,
        indices,
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=float(
            params["learning_rate"]
        ),

        weight_decay=WEIGHT_DECAY,
    )


    loader = make_loader(

        graphs=graphs,

        indices=indices,

        batch_size=params[
            "batch_size"
        ],

        shuffle=True,

        seed=seed,
    )


    for _ in range(
        int(epochs)
    ):

        train_epoch(
            model,
            loader,
            optimizer,
            criterion,
        )


    return model


# ============================================================
# CREATE STRATIFIED 5-FOLD PLAN
# ============================================================

def make_cv_plan(
    labels,
    seed,
):

    skf = StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=seed,
    )


    plan = []

    dummy_x = np.zeros(
        len(labels)
    )


    for (
        fold,
        (
            outer_train,
            validation,
        ),
    ) in enumerate(

        skf.split(
            dummy_x,
            labels,
        ),

        start=1,
    ):

        outer_labels = labels[
            outer_train
        ]


        stop_size = max(

            len(
                np.unique(
                    outer_labels
                )
            ),

            int(
                round(
                    len(outer_train)
                    *
                    EARLY_STOP_FRACTION
                )
            ),
        )


        splitter = StratifiedShuffleSplit(

            n_splits=1,

            test_size=stop_size,

            random_state=(
                seed
                +
                fold
            ),
        )


        (
            fit_relative,
            stop_relative,
        ) = next(

            splitter.split(

                np.zeros(
                    len(outer_train)
                ),

                outer_labels,
            )
        )


        plan.append({

            "fold":
                fold,

            "fit":
                outer_train[
                    fit_relative
                ],

            "stop":
                outer_train[
                    stop_relative
                ],

            "valid":
                validation,
        })


    return plan


# ============================================================
# READ KNIME INPUT
#
# IMPORTANT:
# Do NOT reset the DataFrame index.
#
# The DataFrame index corresponds to the KNIME RowID and will
# be reused in the internal-validation output.
# ============================================================

df_input = (
    knio.input_tables[0]
    .to_pandas()
)


# ============================================================
# CHECK REQUIRED COLUMNS
#
# Old RowID is preserved for traceability only.
#
# ONLY SMILES and Activity are used by the model.
# ============================================================

for column in (
    ID_COLUMN,
    SMILES_COLUMN,
    TARGET_COLUMN,
):

    if column not in df_input.columns:

        raise ValueError(
            f"Required column "
            f"'{column}' "
            f"was not found."
        )


# ============================================================
# WORKING DATAFRAME
#
# Keep:
# - Old RowID
# - SMILES
# - Activity
#
# The original Pandas index / KNIME RowID is also preserved.
# ============================================================

df = df_input[
    [
        ID_COLUMN,
        SMILES_COLUMN,
        TARGET_COLUMN,
    ]
].copy()


# ============================================================
# MISSING VALUES
#
# Only SMILES and Activity are required for model development.
#
# Old RowID is not used by the model.
# ============================================================

if df[
    [
        SMILES_COLUMN,
        TARGET_COLUMN,
    ]
].isna().any().any():

    raise ValueError(
        "SMILES and Activity "
        "cannot contain missing values."
    )


# ============================================================
# TARGET CONVERSION
#
# inactive -> 0
# active   -> 1
# ============================================================

activity = (

    df[TARGET_COLUMN]

    .astype(str)

    .str.strip()

    .str.lower()
)


unexpected = (

    set(
        activity.unique()
    )

    -

    {
        POSITIVE_CLASS,
        NEGATIVE_CLASS,
    }
)


if unexpected:

    raise ValueError(

        "Activity must contain only "
        "'active' and 'inactive'. "

        f"Unexpected values: "
        f"{sorted(unexpected)}"
    )


labels = (

    activity.to_numpy()
    ==
    POSITIVE_CLASS

).astype(int)


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = {

    NEGATIVE_CLASS:

        int(
            (labels == 0).sum()
        ),

    POSITIVE_CLASS:

        int(
            (labels == 1).sum()
        ),
}


if (
    min(
        class_counts.values()
    )
    <
    N_SPLITS
):

    raise ValueError(

        f"At least {N_SPLITS} "
        f"compounds per class are required "
        f"for {N_SPLITS}-fold CV. "

        f"Counts: {class_counts}"
    )


# ============================================================
# SMILES -> GRAPH CONVERSION
#
# IMPORTANT:
# Old RowID and KNIME RowID are NOT used here.
# ============================================================

graphs = []
errors = []


for (
    i,
    (
        smiles,
        label,
    ),
) in enumerate(

    zip(

        df[SMILES_COLUMN]
        .astype(str),

        labels,
    )
):

    try:

        graphs.append(

            smiles_to_graph(
                smiles,
                int(label),
            )
        )


    except Exception as exc:

        errors.append(
            (
                i,
                smiles,
                str(exc),
            )
        )


if errors:

    preview = "\n".join(

        (
            f"row {i}: "
            f"{smiles} -> "
            f"{message}"
        )

        for (
            i,
            smiles,
            message,
        )
        in errors[:10]
    )


    raise ValueError(

        f"{len(errors)} invalid SMILES "
        f"were found. "

        "Training stopped; "
        "no rows were silently removed.\n"

        +
        preview
    )


# ============================================================
# SUMMARY
# ============================================================

print("====================================")
print("GNN LEARNER")
print("====================================")

print(
    f"Molecules: "
    f"{len(graphs)}"
)

print(
    f"Inactive: "
    f"{class_counts[NEGATIVE_CLASS]}"
)

print(
    f"Active: "
    f"{class_counts[POSITIVE_CLASS]}"
)

print(
    f"Node features: "
    f"{NODE_DIM}"
)

print(
    f"Edge features: "
    f"{EDGE_DIM}"
)

print(
    f"Torch: "
    f"{torch.__version__}"
)

print(
    f"PyTorch Geometric: "
    f"{torch_geometric.__version__}"
)

print(
    f"Device: "
    f"{DEVICE}"
)

print(
    f"Pin memory: "
    f"{PIN_MEMORY}"
)

print(
    f"Non-blocking transfers: "
    f"{NON_BLOCKING}"
)


if torch.cuda.is_available():

    print(
        f"CUDA runtime: "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


print("====================================")


# ============================================================
# 1. HYPERPARAMETER TUNING
#
# Each hyperparameter configuration receives ONE pooled
# out-of-fold Accuracy across all five folds.
# ============================================================

tuning_plan = make_cv_plan(
    labels,
    TUNING_CV_SEED,
)


candidates = list(
    ParameterGrid(
        TUNING_GRID
    )
)


params_by_id = {}
tuning_rows = []

n_samples = len(labels)


for (
    candidate_id,
    params,
) in enumerate(

    candidates,
    start=1,
):

    print("------------------------------------")

    print(
        f"Tuning candidate "
        f"{candidate_id}/"
        f"{len(candidates)}"
    )

    print(params)


    fold_accuracies = []
    fold_best_epochs = []


    candidate_oof_prediction = np.full(
        n_samples,
        -1,
        dtype=int,
    )


    candidate_oof_probability = np.full(
        n_samples,
        np.nan,
        dtype=float,
    )


    for info in tuning_plan:

        fold = info["fold"]

        seed = (
            TUNING_CV_SEED
            +
            fold
        )


        (
            model,
            best_epoch,
        ) = fit_with_early_stopping(

            graphs=graphs,

            labels=labels,

            fit_indices=
                info["fit"],

            early_stop_indices=
                info["stop"],

            params=params,

            seed=seed,
        )


        validation_indices = info[
            "valid"
        ]


        probability = predict_proba(

            model=model,

            graphs=graphs,

            indices=
                validation_indices,

            batch_size=
                params[
                    "batch_size"
                ],

            seed=seed,
        )


        prediction = (

            probability
            >=
            DECISION_THRESHOLD

        ).astype(int)


        candidate_oof_probability[
            validation_indices
        ] = probability


        candidate_oof_prediction[
            validation_indices
        ] = prediction


        fold_accuracy = accuracy_score(

            labels[
                validation_indices
            ],

            prediction,
        )


        fold_accuracies.append(
            float(fold_accuracy)
        )


        fold_best_epochs.append(
            int(best_epoch)
        )


        print(

            f"  Fold {fold}: "

            f"accuracy="
            f"{fold_accuracy:.4f}; "

            f"best epoch="
            f"{best_epoch}"
        )


        del model


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    if (
        (
            candidate_oof_prediction
            <
            0
        ).any()

        or

        np.isnan(
            candidate_oof_probability
        ).any()
    ):

        raise RuntimeError(
            "Tuning OOF predictions "
            "are incomplete."
        )


    overall_cv_accuracy = accuracy_score(
        labels,
        candidate_oof_prediction,
    )


    mean_fold_accuracy = float(
        np.mean(
            fold_accuracies
        )
    )


    sd_fold_accuracy = float(
        np.std(
            fold_accuracies,
            ddof=1,
        )
    )


    mean_best_epoch = float(
        np.mean(
            fold_best_epochs
        )
    )


    print(

        "  Overall pooled "
        "5-fold CV Accuracy: "

        f"{overall_cv_accuracy:.4f}"
    )


    row = {

        "Candidate":
            candidate_id,

        "GNN Layers":
            int(
                params[
                    "num_layers"
                ]
            ),

        "Hidden Dimension":
            int(
                params[
                    "hidden_dim"
                ]
            ),

        "Dropout":
            float(
                params[
                    "dropout"
                ]
            ),

        "Learning Rate":
            float(
                params[
                    "learning_rate"
                ]
            ),

        "Batch Size":
            int(
                params[
                    "batch_size"
                ]
            ),

        "Overall CV Accuracy":
            float(
                overall_cv_accuracy
            ),

        "Mean Fold Accuracy":
            mean_fold_accuracy,

        "SD Fold Accuracy":
            sd_fold_accuracy,

        "Mean Best Epoch":
            mean_best_epoch,
    }


    for (
        fold,
        accuracy,
    ) in enumerate(
        fold_accuracies,
        start=1,
    ):

        row[
            f"Fold {fold} Accuracy"
        ] = accuracy


    tuning_rows.append(row)

    params_by_id[
        candidate_id
    ] = dict(params)


# ============================================================
# TUNING RESULTS TABLE
# ============================================================

tuning_df = pd.DataFrame(
    tuning_rows
)


tuning_df = (

    tuning_df

    .sort_values(

        by=[
            "Overall CV Accuracy",
            "SD Fold Accuracy",
            "GNN Layers",
            "Hidden Dimension",
            "Candidate",
        ],

        ascending=[
            False,
            True,
            True,
            True,
            True,
        ],
    )

    .reset_index(
        drop=True
    )
)


tuning_df.insert(

    0,

    "Rank",

    np.arange(
        1,
        len(tuning_df) + 1,
    ),
)


# ============================================================
# BEST HYPERPARAMETERS
# ============================================================

best_candidate = int(
    tuning_df.loc[
        0,
        "Candidate",
    ]
)


best_params = dict(
    params_by_id[
        best_candidate
    ]
)


print("====================================")
print("BEST HYPERPARAMETERS")
print("====================================")

print(best_params)

print(
    "Overall pooled CV Accuracy: "
    f"{tuning_df.loc[0, 'Overall CV Accuracy']:.4f}"
)

print(
    "Mean Fold Accuracy: "
    f"{tuning_df.loc[0, 'Mean Fold Accuracy']:.4f}"
)

print(
    "SD Fold Accuracy: "
    f"{tuning_df.loc[0, 'SD Fold Accuracy']:.4f}"
)

print("====================================")


# ============================================================
# 2. FINAL INTERNAL VALIDATION
# ============================================================

internal_plan = make_cv_plan(
    labels,
    INTERNAL_CV_SEED,
)


oof_probability = np.full(
    n_samples,
    np.nan,
)


oof_prediction = np.full(
    n_samples,
    -1,
    dtype=int,
)


oof_fold = np.full(
    n_samples,
    -1,
    dtype=int,
)


internal_best_epochs = []


for info in internal_plan:

    fold = info["fold"]

    seed = (
        INTERNAL_CV_SEED
        +
        fold
    )


    (
        model,
        best_epoch,
    ) = fit_with_early_stopping(

        graphs=graphs,

        labels=labels,

        fit_indices=
            info["fit"],

        early_stop_indices=
            info["stop"],

        params=best_params,

        seed=seed,
    )


    validation_indices = info[
        "valid"
    ]


    probability = predict_proba(

        model=model,

        graphs=graphs,

        indices=
            validation_indices,

        batch_size=
            best_params[
                "batch_size"
            ],

        seed=seed,
    )


    prediction = (

        probability
        >=
        DECISION_THRESHOLD

    ).astype(int)


    oof_probability[
        validation_indices
    ] = probability


    oof_prediction[
        validation_indices
    ] = prediction


    oof_fold[
        validation_indices
    ] = fold


    internal_best_epochs.append(
        int(best_epoch)
    )


    fold_accuracy = accuracy_score(

        labels[
            validation_indices
        ],

        prediction,
    )


    print(

        f"Internal fold "
        f"{fold}: "

        f"accuracy="
        f"{fold_accuracy:.4f}; "

        f"best epoch="
        f"{best_epoch}"
    )


    del model


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# CHECK INTERNAL VALIDATION
# ============================================================

if (
    np.isnan(
        oof_probability
    ).any()

    or

    (
        oof_prediction
        <
        0
    ).any()
):

    raise RuntimeError(
        "Internal-validation "
        "OOF predictions "
        "are incomplete."
    )


overall_oof_accuracy = accuracy_score(
    labels,
    oof_prediction,
)


print("====================================")

print(
    "Overall internal "
    "OOF accuracy: "
    f"{overall_oof_accuracy:.4f}"
)

print("====================================")


# ============================================================
# 3. FINAL MODEL TRAINING
# ============================================================

final_epochs = max(

    1,

    int(
        np.round(
            np.median(
                internal_best_epochs
            )
        )
    ),
)


all_indices = np.arange(
    n_samples
)


print("====================================")
print("FINAL MODEL TRAINING")
print("====================================")

print(
    f"Training molecules: "
    f"{n_samples}"
)

print(
    f"Final epochs: "
    f"{final_epochs}"
)

print(
    f"Device: "
    f"{DEVICE}"
)

print("====================================")


final_model = fit_fixed_epochs(

    graphs=graphs,

    labels=labels,

    indices=all_indices,

    params=best_params,

    epochs=final_epochs,

    seed=FINAL_MODEL_SEED,
)


final_model.eval()


# ============================================================
# SERIALIZABLE MODEL PACKAGE
# ============================================================

cpu_state_dict = {

    key:
        value
        .detach()
        .cpu()

    for (
        key,
        value,
    )
    in final_model
        .state_dict()
        .items()
}


model_package = {

    "model_type":
        "GINE_binary_molecular_classifier",

    # Version maintained for Predictor compatibility
    "model_version":
        "1.0",

    "state_dict":
        cpu_state_dict,


    "architecture": {

        "node_feature_dim":
            int(NODE_DIM),

        "edge_feature_dim":
            int(EDGE_DIM),

        "hidden_dim":
            int(
                best_params[
                    "hidden_dim"
                ]
            ),

        "num_layers":
            int(
                best_params[
                    "num_layers"
                ]
            ),

        "dropout":
            float(
                best_params[
                    "dropout"
                ]
            ),
    },


    "training": {

        # Identifier is stored only for metadata /
        # traceability. It is never used by the model.
        "id_column":
            ID_COLUMN,

        "smiles_column":
            SMILES_COLUMN,

        "target_column":
            TARGET_COLUMN,

        "positive_class":
            POSITIVE_CLASS,

        "negative_class":
            NEGATIVE_CLASS,

        "decision_threshold":
            float(
                DECISION_THRESHOLD
            ),

        "class_weighting":
            bool(
                CLASS_WEIGHTING
            ),

        "weight_decay":
            float(
                WEIGHT_DECAY
            ),

        "n_splits":
            int(
                N_SPLITS
            ),

        "tuning_metric":
            "Overall pooled 5-fold CV Accuracy",

        "tuning_strategy":
            "pooled_out_of_fold_predictions",

        "tuning_cv_seed":
            int(
                TUNING_CV_SEED
            ),

        "internal_cv_seed":
            int(
                INTERNAL_CV_SEED
            ),

        "final_model_seed":
            int(
                FINAL_MODEL_SEED
            ),

        "final_epochs":
            int(
                final_epochs
            ),

        "best_hyperparameters":
            dict(
                best_params
            ),

        "best_tuning_accuracy":
            float(
                tuning_df.loc[
                    0,
                    "Overall CV Accuracy",
                ]
            ),

        "internal_oof_accuracy":
            float(
                overall_oof_accuracy
            ),
    },


    "featurization": {

        "name":
            "LASSBio_GINE_features_v1",

        "atom_features": [

            "atomic_number",

            "degree",

            "formal_charge",

            "hybridization",

            "total_H",

            "aromaticity",

            "ring_membership",

            "chirality",
        ],

        "bond_features": [

            "bond_type",

            "conjugation",

            "aromaticity",

            "ring_membership",

            "stereochemistry",
        ],

        "node_feature_dimension":
            int(
                NODE_DIM
            ),

        "edge_feature_dimension":
            int(
                EDGE_DIM
            ),

        "directed_edges_per_bond":
            2,
    },


    "software_versions": {

        "torch":
            str(
                torch.__version__
            ),

        "torch_geometric":
            str(
                torch_geometric.__version__
            ),

        "rdkit":
            str(
                rdkit.__version__
            ),
    },
}


# ============================================================
# OUTPUT TABLE 0
#
# INTERNAL VALIDATION
#
# IMPORTANT:
#
# Both identifiers are retained:
#
# 1. KNIME RowID -> DataFrame index
# 2. Old RowID   -> normal column
#
# Neither identifier participates in model development.
# ============================================================

internal_validation_df = pd.DataFrame(

    {

        ID_COLUMN:

            df[
                ID_COLUMN
            ]
            .to_numpy(),


        SMILES_COLUMN:

            df[
                SMILES_COLUMN
            ]
            .astype(str)
            .to_numpy(),


        TARGET_COLUMN:

            np.where(
                labels == 1,
                POSITIVE_CLASS,
                NEGATIVE_CLASS,
            ),


        "Prediction (GNN)":

            np.where(
                oof_prediction == 1,
                POSITIVE_CLASS,
                NEGATIVE_CLASS,
            ),


        "P (active)":

            oof_probability.astype(
                float
            ),


        "CV Fold":

            oof_fold.astype(
                int
            ),
    },

    # Preserve the original KNIME RowID
    index=df.index.copy(),
)


# Preserve the original index name if one exists
internal_validation_df.index.name = df.index.name


# ============================================================
# OUTPUT TABLE 1
#
# HYPERPARAMETER TUNING RESULTS
# ============================================================

tuning_columns = [

    "Rank",

    "Candidate",

    "GNN Layers",

    "Hidden Dimension",

    "Dropout",

    "Learning Rate",

    "Batch Size",

    "Overall CV Accuracy",

    "Mean Fold Accuracy",

    "SD Fold Accuracy",

    "Mean Best Epoch",

] + [

    f"Fold {i} Accuracy"

    for i in range(
        1,
        N_SPLITS + 1,
    )
]


tuning_df = tuning_df[
    tuning_columns
]


# ============================================================
# SEND OUTPUTS TO KNIME
# ============================================================

# ------------------------------------------------------------
# OUTPUT OBJECT 0
#
# Final GNN model
# ------------------------------------------------------------

knio.output_objects[0] = (
    model_package
)


# ------------------------------------------------------------
# OUTPUT TABLE 0
#
# Internal validation.
#
# row_ids="keep" tells KNIME to use the Pandas index as
# the RowID instead of generating new RowIDs.
# ------------------------------------------------------------

knio.output_tables[0] = (

    knio.Table.from_pandas(

        internal_validation_df,

        row_ids="keep",
    )
)


# ------------------------------------------------------------
# OUTPUT TABLE 1
#
# Tuning results.
#
# These rows represent hyperparameter configurations rather
# than molecules, so original molecule RowIDs do not apply.
# ------------------------------------------------------------

knio.output_tables[1] = (

    knio.Table.from_pandas(
        tuning_df
    )
)


# ============================================================
# FINAL CUDA / MEMORY INFORMATION
# ============================================================

peak_gpu_memory_gb = None


if torch.cuda.is_available():

    torch.cuda.synchronize()

    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated()
        /
        (1024 ** 3)
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("====================================")
print("GNN LEARNER COMPLETED SUCCESSFULLY")
print("====================================")

print(
    f"Best hyperparameters: "
    f"{best_params}"
)

print(
    "Best pooled tuning Accuracy: "
    f"{tuning_df.loc[0, 'Overall CV Accuracy']:.4f}"
)

print(
    "Final internal OOF Accuracy: "
    f"{overall_oof_accuracy:.4f}"
)

print(
    f"Final model epochs: "
    f"{final_epochs}"
)

print(
    f"Final model trained on: "
    f"{n_samples} molecules"
)

print(
    f"Device: "
    f"{DEVICE}"
)


if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA runtime: "
        f"{torch.version.cuda}"
    )

    print(
        f"Peak GPU memory allocated: "
        f"{peak_gpu_memory_gb:.3f} GB"
    )


print(
    f"Identifier column preserved: "
    f"{ID_COLUMN}"
)

print(
    "Original KNIME RowIDs preserved: True"
)

print("====================================")