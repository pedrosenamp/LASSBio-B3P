import numpy as np
import pandas as pd
import knime.scripting.io as knio

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from rdkit import Chem


# ============================================================
# CONFIGURATION
# ============================================================

SMILES_COLUMN = "SMILES"

# Prediction does not require Activity.
# If Activity is present in the input table, it is simply
# preserved in the output.

DEFAULT_BATCH_SIZE = 64

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# READ TRAINED MODEL PACKAGE
# ============================================================

model_package = knio.input_objects[0]


if not isinstance(model_package, dict):
    raise ValueError(
        "The Python Object input is not a valid "
        "GNN model package."
    )


if (
    model_package.get("model_type")
    != "GINE_binary_molecular_classifier"
):
    raise ValueError(
        "The supplied model is not a compatible "
        "GINE molecular classifier."
    )


model_version = model_package.get(
    "model_version"
)


if model_version != "1.0":
    raise ValueError(
        f"Unsupported GNN model version: "
        f"{model_version}"
    )


architecture = model_package[
    "architecture"
]

training_info = model_package[
    "training"
]


POSITIVE_CLASS = training_info[
    "positive_class"
]

NEGATIVE_CLASS = training_info[
    "negative_class"
]

DECISION_THRESHOLD = float(
    training_info[
        "decision_threshold"
    ]
)


NODE_DIM = int(
    architecture[
        "node_feature_dim"
    ]
)

EDGE_DIM = int(
    architecture[
        "edge_feature_dim"
    ]
)


# Use the training batch size when available
BEST_PARAMS = training_info.get(
    "best_hyperparameters",
    {}
)

BATCH_SIZE = int(
    BEST_PARAMS.get(
        "batch_size",
        DEFAULT_BATCH_SIZE,
    )
)


# ============================================================
# MOLECULAR FEATURIZATION
#
# IMPORTANT:
# This must be IDENTICAL to the featurization used by
# GNN Learner version 1.0.
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


def one_hot_other(
    value,
    choices,
):

    output = [
        float(
            value == item
        )
        for item in choices
    ]

    # Additional "other" category
    output.append(
        float(
            value not in choices
        )
    )

    return output


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
            float(
                atom.GetIsAromatic()
            ),

            float(
                atom.IsInRing()
            ),
        ]
        +
        one_hot_other(
            atom.GetChiralTag(),
            CHIRAL_TAGS,
        )
    )


def bond_features(bond):

    return (
        one_hot_other(
            bond.GetBondType(),
            BOND_TYPES,
        )
        +
        [
            float(
                bond.GetIsConjugated()
            ),

            float(
                bond.GetIsAromatic()
            ),

            float(
                bond.IsInRing()
            ),
        ]
        +
        one_hot_other(
            bond.GetStereo(),
            BOND_STEREO,
        )
    )


# ============================================================
# CHECK FEATURE DIMENSIONS
#
# Protects against accidentally using a Predictor whose
# featurization differs from the Learner.
# ============================================================

_test_atom_mol = Chem.MolFromSmiles(
    "C"
)

_test_bond_mol = Chem.MolFromSmiles(
    "CC"
)


predictor_node_dim = len(
    atom_features(
        _test_atom_mol.GetAtomWithIdx(
            0
        )
    )
)


predictor_edge_dim = len(
    bond_features(
        _test_bond_mol.GetBondWithIdx(
            0
        )
    )
)


if predictor_node_dim != NODE_DIM:
    raise ValueError(
        "Atom featurization is incompatible "
        "with the trained GNN model. "
        f"Model expects {NODE_DIM} features, "
        f"but Predictor generates "
        f"{predictor_node_dim}."
    )


if predictor_edge_dim != EDGE_DIM:
    raise ValueError(
        "Bond featurization is incompatible "
        "with the trained GNN model. "
        f"Model expects {EDGE_DIM} features, "
        f"but Predictor generates "
        f"{predictor_edge_dim}."
    )


# ============================================================
# SMILES -> MOLECULAR GRAPH
# ============================================================

def smiles_to_graph(smiles):

    if (
        not isinstance(
            smiles,
            str,
        )
        or
        not smiles.strip()
    ):

        raise ValueError(
            "Empty or non-string SMILES."
        )


    mol = Chem.MolFromSmiles(
        smiles
    )


    if mol is None:

        raise ValueError(
            "RDKit could not parse SMILES."
        )


    # Explicitly assign stereochemistry
    Chem.AssignStereochemistry(
        mol,
        cleanIt=True,
        force=True,
    )


    # --------------------------------------------------------
    # ATOMS
    # --------------------------------------------------------

    x = torch.tensor(
        [
            atom_features(atom)
            for atom
            in mol.GetAtoms()
        ],
        dtype=torch.float32,
    )


    # --------------------------------------------------------
    # BONDS
    #
    # PyTorch Geometric represents each chemical bond in
    # both directions for message passing:
    #
    # i -> j
    # j -> i
    # --------------------------------------------------------

    source = []
    target = []

    edge_attributes = []


    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        features = bond_features(
            bond
        )


        source.extend(
            [i, j]
        )

        target.extend(
            [j, i]
        )


        edge_attributes.extend(
            [
                features,
                features,
            ]
        )


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

        # Allows single-atom molecules
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_attr = torch.empty(
            (
                0,
                EDGE_DIM,
            ),
            dtype=torch.float32,
        )


    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


# ============================================================
# GINE ARCHITECTURE
#
# Must be IDENTICAL to GNN Learner version 1.0.
# ============================================================

class GINEClassifier(
    nn.Module
):

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):

        super().__init__()


        self.dropout = float(
            dropout
        )


        self.convs = (
            nn.ModuleList()
        )

        self.norms = (
            nn.ModuleList()
        )


        input_dim = node_dim


        for _ in range(
            int(
                num_layers
            )
        ):

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


            self.convs.append(
                conv
            )


            self.norms.append(
                nn.LayerNorm(
                    hidden_dim
                )
            )


            input_dim = (
                hidden_dim
            )


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

        for (
            conv,
            norm,
        ) in zip(
            self.convs,
            self.norms,
        ):

            x = conv(
                x,
                edge_index,
                edge_attr,
            )


            x = norm(
                x
            )


            x = torch.relu(
                x
            )


            x = nn.functional.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )


        # Convert atom-level embeddings
        # into one molecular embedding.
        x = global_mean_pool(
            x,
            batch,
        )


        return (
            self.head(x)
            .view(-1)
        )


# ============================================================
# RECONSTRUCT TRAINED MODEL
# ============================================================

model = GINEClassifier(

    node_dim=NODE_DIM,

    edge_dim=EDGE_DIM,

    hidden_dim=int(
        architecture[
            "hidden_dim"
        ]
    ),

    num_layers=int(
        architecture[
            "num_layers"
        ]
    ),

    dropout=float(
        architecture[
            "dropout"
        ]
    ),

).to(
    DEVICE
)


# Load weights learned by the Learner
model.load_state_dict(
    model_package[
        "state_dict"
    ]
)


model.eval()


# ============================================================
# READ PREDICTION TABLE
# ============================================================

df_input = (
    knio.input_tables[0]
    .to_pandas()
    .reset_index(
        drop=True
    )
)


if (
    SMILES_COLUMN
    not in df_input.columns
):

    raise ValueError(
        f"Required column "
        f"'{SMILES_COLUMN}' "
        f"was not found."
    )


if (
    df_input[
        SMILES_COLUMN
    ]
    .isna()
    .any()
):

    raise ValueError(
        "The SMILES column contains "
        "missing values."
    )


# ============================================================
# CONVERT ALL MOLECULES TO GRAPHS
#
# Prediction stops if an invalid SMILES is found.
# Rows are never silently removed.
# ============================================================

graphs = []

errors = []


for i, smiles in enumerate(
    df_input[
        SMILES_COLUMN
    ]
):

    try:

        graph = (
            smiles_to_graph(
                smiles
            )
        )

        graphs.append(
            graph
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
        f"{len(errors)} invalid "
        f"SMILES were found. "
        f"Prediction stopped; "
        f"no rows were silently removed.\n"
        + preview
    )


# ============================================================
# DATA LOADER
# ============================================================

loader = DataLoader(
    graphs,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# PREDICTION
# ============================================================

active_probabilities = []


with torch.no_grad():

    for batch in loader:

        batch = batch.to(
            DEVICE
        )


        logits = model(

            batch.x,

            batch.edge_index,

            batch.edge_attr,

            batch.batch,

        )


        probabilities = (
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .numpy()
        )


        active_probabilities.append(
            probabilities
        )


active_probabilities = (
    np.concatenate(
        active_probabilities
    )
    .astype(float)
)


inactive_probabilities = (
    1.0
    -
    active_probabilities
)


predicted_numeric = (
    active_probabilities
    >= DECISION_THRESHOLD
).astype(int)


predicted_class = np.where(

    predicted_numeric == 1,

    POSITIVE_CLASS,

    NEGATIVE_CLASS,

)


# ============================================================
# CREATE OUTPUT TABLE
#
# All original columns are preserved.
# ============================================================

output_df = (
    df_input.copy()
)


output_df[
    "Prediction (GNN)"
] = predicted_class


output_df[
    "P (inactive)"
] = inactive_probabilities


output_df[
    "P (active)"
] = active_probabilities


# ============================================================
# SEND RESULT TO KNIME
# ============================================================

knio.output_tables[0] = (
    knio.Table.from_pandas(
        output_df
    )
)


print(
    "===================================="
)

print(
    "GNN Predictor completed successfully."
)

print(
    f"Molecules predicted: "
    f"{len(output_df)}"
)

print(
    f"Device: {DEVICE}"
)

print(
    f"Decision threshold: "
    f"{DECISION_THRESHOLD:.3f}"
)

print(
    f"Predicted {NEGATIVE_CLASS}: "
    f"{int((predicted_numeric == 0).sum())}"
)

print(
    f"Predicted {POSITIVE_CLASS}: "
    f"{int((predicted_numeric == 1).sum())}"
)

print(
    "===================================="
)
