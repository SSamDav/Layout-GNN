## Learn Layout metric using a Neural Render

The main idea of this is to combine the two approaches used in [[1]](https://arxiv.org/pdf/2012.06547.pdf), [[2]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670732.pdf).

Basically, the idea was that a similarity distance between UI layout must encode graph similarity and rendering similarity. To achieve this, the idea is to encode the UI layout as a graph (tree) and train a GNN in order to find a latent space where these two similarities are “merged”.

In order to train the GNN, we would use a loss that consists of:

- A triplet loss between 3 UI, the positive pair and the negative pair, is found using Tree/Graph edit distance. (This part allows the GNN to learn graph structural information)
- A reconstruction loss that from the GNN embedding tries to render the UI. For this, we could use the L2 loss and FID loss between the images. This term allows the GNN to learn that some features influence more the final UI than others.

To achieve this, we will need 2 models:

- One GNN that embeds a UI graph.
- One Neural Render that from a GNN embedding renders the resultant UI.

In order to train all this, we will use the [RICO](https://interactionmining.org/rico) dataset.

## Research Steps
- **Dataset Creation**
  - Defining the core representation and normalize the data.
  - Create the triplets using a pre defined metric.

- **Model Training**
  - Defining which models we want to use and implement the code.
  - Implement the training script that uses multiple losses.

- **Analysis**
  - Ablation study of different models.
  - Ablation study of use of different representations.
  - Ablation study of use of different losses.