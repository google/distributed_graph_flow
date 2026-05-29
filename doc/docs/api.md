# (Distributed) GraphFlow API

All the Apache Beam distributed functions / classes are defined in `dgf.beam.*`. All the other (e.g., in-process, in-memory) functions and classes are defined in `dgf.*` directly:

*   [`dgf.analyse.*`](#section-dgf-analyse): Utilities to analyze graphs, e.g., feature and graph statistics.
*   [`dgf.convert.*`](#section-dgf-convert): Converts object formats, e.g., a graph to a Sparse Deferred struct.
*   [`dgf.data.*`](#section-dgf-data): Classes that represent graph data. Contains no functions or algorithms.
*   [`dgf.exception.*`](#section-dgf-exception): DGF-specific exceptions.
*   [`dgf.filesystem.*`](#section-dgf-filesystem): GraphFlow unified filesystem API.
*   [`dgf.generate.*`](#section-dgf-generate): Tools to generate synthetic data.
*   [`dgf.io.*`](#section-dgf-io): Functions to read and write graphs, schemas, and related data.
*   [`dgf.jax.*`](#section-dgf-jax): Machine Learning and Graph Neural Networks using JAX.
*   [`dgf.learning.*`](#section-dgf-learning): Top-level learning module.
*   [`dgf.plot.*`](#section-dgf-plot): Functions to plot graphs, schemas, and other graph-related data.
*   [`dgf.print.*`](#section-dgf-print): Functions for printing structures.
*   [`dgf.sampling.*`](#section-dgf-sampling): Functions and classes to extract subsets of graphs for GNN training.
*   [`dgf.train.*`](#section-dgf-train): Functions and classes to train core GNN models.
*   [`dgf.transform.*`](#section-dgf-transform): Transforms graph data into other graph structures or formats.
*   [`dgf.validate.*`](#section-dgf-validate): Functions to validate graph data.
*   [`dgf.beam.*`](#section-dgf-beam): Apache Beam-related functions and classes.


Functions not yet part of the official API are available under `dgf.src.*`. Those are not listed in this page.



### Module `dgf.analyse`    # {: #section-dgf-analyse}

Utilities to analyze graphs, e.g., feature and graph statistics.

*   [`dgf.analyse.feature_statistics`](api/dgf-analyse.md#section-feature-statistics): Computes the feature stats from a single graph.
*   [`dgf.analyse.feature_statistics_from_graphs`](api/dgf-analyse.md#section-feature-statistics-from-graphs): Computes the feature stats from multiple graphs.
*   [`dgf.analyse.padding_from_graph_generator`](api/dgf-analyse.md#section-padding-from-graph-generator): Creates a padding configuration from a set of in-memory graphs.
*   [`dgf.analyse.print_schema`](api/dgf-analyse.md#section-print-schema): Generates a human-readable string representation of a graph schema.




### Module `dgf.convert`    # {: #section-dgf-convert}

Converts object formats, e.g., a graph to a Sparse Deferred struct.

*   [`dgf.convert.graph_dict_to_graph`](api/dgf-convert.md#section-graph-dict-to-graph): Converts a TF GNN Graph Sample Dict to an InMemoryGraph.
*   [`dgf.convert.graph_to_jax_graph`](api/dgf-convert.md#section-graph-to-jax-graph): Converts a (NumPy) in-memory graph into a JAX in-memory graph.
*   [`dgf.convert.graph_to_networkx`](api/dgf-convert.md#section-graph-to-networkx): Converts an InMemoryGraph into a NetworkX MultiDiGraph.
*   [`dgf.convert.graph_to_serialized_tfgnn_graph`](api/dgf-convert.md#section-graph-to-serialized-tfgnn-graph): Converts an InMemoryGraph into a serialized TF-GNN graph sample proto.
*   [`dgf.convert.graph_to_sparse_deferred_struct`](api/dgf-convert.md#section-graph-to-sparse-deferred-struct): Converts an in-memory graph into a Sparse Deferred struct.
*   [`dgf.convert.graph_to_tf_graph`](api/dgf-convert.md#section-graph-to-tf-graph): Converts a graph to a TF in-memory graph.
*   [`dgf.convert.graph_to_tfgnn_graph`](api/dgf-convert.md#section-graph-to-tfgnn-graph): Converts an InMemoryGraph to a TF GNN Graph Sample.
*   [`dgf.convert.graph_to_tfgnn_graph_dict`](api/dgf-convert.md#section-graph-to-tfgnn-graph-dict): Converts an InMemoryGraph to a TF GNN Graph Sample Dict.
*   [`dgf.convert.graphs_to_serialized_tfgnn_graphs`](api/dgf-convert.md#section-graphs-to-serialized-tfgnn-graphs): Converts a sequence of InMemoryGraphs into serialized TF-GNN graph sample protos.
*   [`dgf.convert.networkx_to_graph`](api/dgf-convert.md#section-networkx-to-graph): Converts a NetworkX graph into an InMemoryGraph and its schema.
*   [`dgf.convert.schema_to_spanner_ddl`](api/dgf-convert.md#section-schema-to-spanner-ddl): Converts a GraphSchema to a string of CREATE statements for Spanner.
*   [`dgf.convert.schema_to_sparse_deferred_schema`](api/dgf-convert.md#section-schema-to-sparse-deferred-schema): Converts a DGF `GraphSchema` into a Sparse Deferred schema.
*   [`dgf.convert.schema_to_tfgnn_schema`](api/dgf-convert.md#section-schema-to-tfgnn-schema): Converts a GraphSchema object into a TF-GNN schema proto.
*   [`dgf.convert.sparse_deferred_struct_to_graph`](api/dgf-convert.md#section-sparse-deferred-struct-to-graph): Converts a Sparse Deferred struct into an in-memory graph.
*   [`dgf.convert.tf_graph_dict_to_tf_graph`](api/dgf-convert.md#section-tf-graph-dict-to-tf-graph): Converts a flattened TFInMemoryGraphDict back into a TFInMemoryGraph.
*   [`dgf.convert.tf_graph_to_tf_graph_dict`](api/dgf-convert.md#section-tf-graph-to-tf-graph-dict): Converts a TFInMemoryGraph into a flattened TFInMemoryGraphDict.
*   [`dgf.convert.tfgnn_graph_to_graph`](api/dgf-convert.md#section-tfgnn-graph-to-graph): Converts a TF GNN Graph Sample to an InMemoryGraph.
*   [`dgf.convert.tfgnn_schema_to_schema`](api/dgf-convert.md#section-tfgnn-schema-to-schema): Converts a TF-GNN schema proto into a GraphSchema object.




### Module `dgf.data`    # {: #section-dgf-data}

Classes that represent graph data. Contains no functions or algorithms.

*   [`dgf.data.EdgeSchema`](api/dgf-data.md#section-edgeschema): EdgeSchema(source: str, target: str, features: Dict[str, dgf.src.data.schema.FeatureSchema] = <factory>)
*   [`dgf.data.EdgeSetPadding`](api/dgf-data.md#section-edgesetpadding): EdgeSetPadding(num_edges: int)
*   [`dgf.data.FeatureFormat`](api/dgf-data.md#section-featureformat): How a value is represented / stored.
*   [`dgf.data.FeatureSchema`](api/dgf-data.md#section-featureschema): Schema for a single feature.
*   [`dgf.data.FeatureSemantic`](api/dgf-data.md#section-featuresemantic): How a value should be interpreted.
*   [`dgf.data.FeatureSetStatistics`](api/dgf-data.md#section-featuresetstatistics): Statistics for a set of features.
*   [`dgf.data.FeatureStatistics`](api/dgf-data.md#section-featurestatistics): Statistics for a feature.
*   [`dgf.data.GraphFeatureStatistics`](api/dgf-data.md#section-graphfeaturestatistics): Statistics about the features in a graph.
*   [`dgf.data.GraphSchema`](api/dgf-data.md#section-graphschema): GraphSchema(node_sets: Dict[str, dgf.src.data.schema.NodeSchema], edge_sets: Dict[str, dgf.src.data.schema.EdgeSchema])
*   [`dgf.data.GraphSchemaFilter`](api/dgf-data.md#section-graphschemafilter): Filters a GraphSchema to sub-select node sets, edge sets, and features.
*   [`dgf.data.GraphSchemaV2`](api/dgf-data.md#section-graphschemav2): GraphSchema(node_sets: Dict[str, dgf.src.data.schema.NodeSchema], edge_sets: Dict[str, dgf.src.data.schema.EdgeSchema])
*   [`dgf.data.InMemoryEdgeSet`](api/dgf-data.md#section-inmemoryedgeset): An Edge Set.
*   [`dgf.data.InMemoryGraph`](api/dgf-data.md#section-inmemorygraph): An in-memory generic graph.
*   [`dgf.data.InMemoryNodeSet`](api/dgf-data.md#section-inmemorynodeset): A Node Set.
*   [`dgf.data.JaxInMemoryEdgeSet`](api/dgf-data.md#section-jaxinmemoryedgeset): An Edge Set.
*   [`dgf.data.JaxInMemoryGraph`](api/dgf-data.md#section-jaxinmemorygraph): An in-memory generic graph.
*   [`dgf.data.JaxInMemoryNodeSet`](api/dgf-data.md#section-jaxinmemorynodeset): A Node Set.
*   [`dgf.data.NodeSchema`](api/dgf-data.md#section-nodeschema): NodeSchema(features: Dict[str, dgf.src.data.schema.FeatureSchema] = <factory>)
*   [`dgf.data.NodeSetPadding`](api/dgf-data.md#section-nodesetpadding): NodeSetPadding(num_nodes: int)
*   [`dgf.data.Padding`](api/dgf-data.md#section-padding): Information to pad a graph.
*   [`dgf.data.TFInMemoryEdgeSet`](api/dgf-data.md#section-tfinmemoryedgeset): An Edge Set.
*   [`dgf.data.TFInMemoryGraph`](api/dgf-data.md#section-tfinmemorygraph): An in-memory generic graph.
*   [`dgf.data.TFInMemoryNodeSet`](api/dgf-data.md#section-tfinmemorynodeset): A Node Set.




### Module `dgf.exception`    # {: #section-dgf-exception}

DGF-specific exceptions.

*   [`dgf.exception.InsufficientPaddingError`](api/dgf-exception.md#section-insufficientpaddingerror): Inappropriate argument value (of correct type).




### Module `dgf.filesystem`    # {: #section-dgf-filesystem}

GraphFlow unified filesystem API.

*   [`dgf.filesystem.create_gcs_bucket`](api/dgf-filesystem.md#section-create-gcs-bucket): Creates a GCS bucket.
*   [`dgf.filesystem.exists`](api/dgf-filesystem.md#section-exists): Returns True if the path exists.
*   [`dgf.filesystem.glob`](api/dgf-filesystem.md#section-glob): Returns a list of files and directories matching a pattern.
*   [`dgf.filesystem.is_gcs_path`](api/dgf-filesystem.md#section-is-gcs-path): Returns True if the path is a Google Cloud Storage (GCS) path.
*   [`dgf.filesystem.makedirs`](api/dgf-filesystem.md#section-makedirs): Creates directories if it does not exist.
*   [`dgf.filesystem.open_read`](api/dgf-filesystem.md#section-open-read): Opens a file for reading and return a python file handle.
*   [`dgf.filesystem.remove_paths`](api/dgf-filesystem.md#section-remove-paths): Removes all the files in parallel.
*   [`dgf.filesystem.rename`](api/dgf-filesystem.md#section-rename): Renames (moves) a file or directory from old_path to new_path.
*   [`dgf.filesystem.rmtree`](api/dgf-filesystem.md#section-rmtree): Recursively removes a directory and its contents.




### Module `dgf.generate`    # {: #section-dgf-generate}

Tools to generate synthetic data.

*   [`dgf.generate.EdgeNeighborGenerator`](api/dgf-generate.md#section-edgeneighborgenerator): Generates the idx of the positive and negative pairs of nodes.
*   [`dgf.generate.RandomNegativeSampler`](api/dgf-generate.md#section-randomnegativesampler): Replace the target node with a random node.
*   [`dgf.generate.RandomWalkNegativeSampler`](api/dgf-generate.md#section-randomwalknegativesampler): Replace the target node with a random walk generated node.
*   [`dgf.generate.SyntheticGraphSampleConfig`](api/dgf-generate.md#section-syntheticgraphsampleconfig): Configuration for generating synthetic graph samples.
*   [`dgf.generate.generate_synthetic_graph_sample`](api/dgf-generate.md#section-generate-synthetic-graph-sample): Generates a single synthetic graph sample based on a sampling plan.
*   [`dgf.generate.write_synthetic_graph_sample_as_tfgnn_graphs`](api/dgf-generate.md#section-write-synthetic-graph-sample-as-tfgnn-graphs): Generates and writes synthetic graph samples as TF-GNN graphs.




### Module `dgf.io`    # {: #section-dgf-io}

Functions to read and write graphs, schemas, and related data.

*   [`dgf.io.cache`](api/dgf-io.md#section-cache): Returns and caches the variable(s) created by "create_fn".
*   [`dgf.io.create_spanner_tables_from_graph_schema`](api/dgf-io.md#section-create-spanner-tables-from-graph-schema): Creates Spanner tables for a graph schema.
*   [`dgf.io.export_bigquery_to_disk`](api/dgf-io.md#section-export-bigquery-to-disk): Reads a BigQuery Graph in-process and returns a GraphFlow in-memory graph.
*   [`dgf.io.fetch_graphland_graph`](api/dgf-io.md#section-fetch-graphland-graph): Downloads and loads a Graphland dataset into memory.
*   [`dgf.io.fetch_ogb_graph`](api/dgf-io.md#section-fetch-ogb-graph): Downloads and loads an OGB node property prediction dataset into memory.
*   [`dgf.io.read_bigquery_graph`](api/dgf-io.md#section-read-bigquery-graph): Reads a BigQuery Graph in-process and returns a GraphFlow in-memory graph.
*   [`dgf.io.read_bigquery_graph_schema`](api/dgf-io.md#section-read-bigquery-graph-schema): Reads the schema of a BigQuery graph into a GF schema.
*   [`dgf.io.read_feature_statistics`](api/dgf-io.md#section-read-feature-statistics): Reads feature statistics from disk in a JSON format.
*   [`dgf.io.read_graph`](api/dgf-io.md#section-read-graph): Reads a GF graph from a directory to an in-memory graph.
*   [`dgf.io.read_graphai_hgraph`](api/dgf-io.md#section-read-graphai-hgraph): Reads an on-disk HGraph into an in-memory representation.
*   [`dgf.io.read_schema`](api/dgf-io.md#section-read-schema): Loads graph schema from disk in a json format.
*   [`dgf.io.read_spanner_graph`](api/dgf-io.md#section-read-spanner-graph): Reads a Spanner Graph in-process and returns a GraphFlow in-memory graph.
*   [`dgf.io.read_spanner_graph_schema`](api/dgf-io.md#section-read-spanner-graph-schema): Reads the schema of a Spanner Graph.
*   [`dgf.io.read_text_proto`](api/dgf-io.md#section-read-text-proto): Read a proto from disk in text format.
*   [`dgf.io.read_tfgnn_graphs`](api/dgf-io.md#section-read-tfgnn-graphs): Reads a set of in-memory graphs from disk stored as TF Examples.
*   [`dgf.io.write_feature_statistics`](api/dgf-io.md#section-write-feature-statistics): Saves feature statistics to disk in a json format.
*   [`dgf.io.write_graph`](api/dgf-io.md#section-write-graph): Writes an in-memory graph and schema to a GF Graph directory.
*   [`dgf.io.write_schema`](api/dgf-io.md#section-write-schema): Saves graph schema to disk in a json format.
*   [`dgf.io.write_text_proto`](api/dgf-io.md#section-write-text-proto): Writes a proto to disk in text format.
*   [`dgf.io.write_tfgnn_graphs`](api/dgf-io.md#section-write-tfgnn-graphs): Writes a set of in-memory graphs to disk as TF Examples.




### Module `dgf.jax`    # {: #section-dgf-jax}

Machine Learning and Graph Neural Networks using JAX.

*   [`dgf.jax.JaxBaseConfig`](api/dgf-jax.md#section-jaxbaseconfig): Base class for a GNN implemented in JAX.
*   [`dgf.jax.get_activation`](api/dgf-jax.md#section-get-activation): Get an activation function by (string) name.
*   [`dgf.jax.jnp_dtype_from_string`](api/dgf-jax.md#section-jnp-dtype-from-string): Return a JAX numpy type from a string name.
*   [`dgf.jax.jnp_name_from_dtype`](api/dgf-jax.md#section-jnp-name-from-dtype): Return a string name for a jnp.dtype object.
*   [`dgf.jax.train`](api/dgf-jax.md#section-train): Trains a Flax module with a flexible and feature-rich training loop.


#### Module `dgf.jax.layers`    # {: #section-dgf-jax-layers}

Flax modules implementing low level GNN operations.

*   [`dgf.jax.layers.ClassificationHead`](api/dgf-jax-layers.md#section-classificationhead): Simple classification head.
*   [`dgf.jax.layers.ClassificationHeadConfig`](api/dgf-jax-layers.md#section-classificationheadconfig): Configuration for a classification head.
*   [`dgf.jax.layers.ConditionalGIN`](api/dgf-jax-layers.md#section-conditionalgin): Conditional GIN with a labeling trick: https://arxiv.org/abs/2106.06935.
*   [`dgf.jax.layers.EmbedAndHomogenizeGraph`](api/dgf-jax-layers.md#section-embedandhomogenizegraph): Convert a heterogeneous graph into a homogeneous one.
*   [`dgf.jax.layers.EmbedAndHomogenizeGraphConfig`](api/dgf-jax-layers.md#section-embedandhomogenizegraphconfig): Config for EmbedAndHomogenizeGraph.
*   [`dgf.jax.layers.EmbedFeatureSet`](api/dgf-jax-layers.md#section-embedfeatureset): Computes a fixed sized dense embedding for a set of feature values.
*   [`dgf.jax.layers.EmbedFeatureSetConfig`](api/dgf-jax-layers.md#section-embedfeaturesetconfig): Configuration for the EmbedFeatureSet layer.
*   [`dgf.jax.layers.EmbedGraph`](api/dgf-jax-layers.md#section-embedgraph): Compute a fixed sized dense embedding for all the features in a graph.
*   [`dgf.jax.layers.EmbedGraphConfig`](api/dgf-jax-layers.md#section-embedgraphconfig): Configuration for "EmbedGraph".
*   [`dgf.jax.layers.GCN`](api/dgf-jax-layers.md#section-gcn): Graph convolutional network: https://arxiv.org/pdf/1609.02907.pdf.
*   [`dgf.jax.layers.GCNConfig`](api/dgf-jax-layers.md#section-gcnconfig): Makeable GCN config class with sensible defaults.
*   [`dgf.jax.layers.GIN`](api/dgf-jax-layers.md#section-gin): Graph isomorphism network: https://arxiv.org/pdf/1810.00826.pdf.
*   [`dgf.jax.layers.GINConfig`](api/dgf-jax-layers.md#section-ginconfig): Makeable GIN config class with sensible defaults.
*   [`dgf.jax.layers.GenericBlock`](api/dgf-jax-layers.md#section-genericblock): A generic configurable neural network block.
*   [`dgf.jax.layers.GenericBlockConfig`](api/dgf-jax-layers.md#section-genericblockconfig): Configuration for a generic block parsed from a string.
*   [`dgf.jax.layers.HeterogeneousGraphAttentionNetwork`](api/dgf-jax-layers.md#section-heterogeneousgraphattentionnetwork): A single layer of heterogeneous Graph Attention Network.
*   [`dgf.jax.layers.HeterogeneousGraphAttentionNetworkConfig`](api/dgf-jax-layers.md#section-heterogeneousgraphattentionnetworkconfig): Configuration for HeterogeneousGraphAttentionNetwork.
*   [`dgf.jax.layers.HeterogeneousGraphConvolution`](api/dgf-jax-layers.md#section-heterogeneousgraphconvolution): A single layer of heterogeneous Graph Neural Network message passing.
*   [`dgf.jax.layers.HeterogeneousGraphConvolutionConfig`](api/dgf-jax-layers.md#section-heterogeneousgraphconvolutionconfig): Configuration for HeterogeneousGraphConvolution.
*   [`dgf.jax.layers.MLP`](api/dgf-jax-layers.md#section-mlp): A generic MLP followed by a linear layer.
*   [`dgf.jax.layers.MPNN`](api/dgf-jax-layers.md#section-mpnn): Message-Passing Neural Network: https://arxiv.org/abs/1704.01212.
*   [`dgf.jax.layers.MPNNConfig`](api/dgf-jax-layers.md#section-mpnnconfig): Makeable MPNN config class with sensible defaults.
*   [`dgf.jax.layers.Projector`](api/dgf-jax-layers.md#section-projector): Simple wrapper around the generic MLP layer for graph input/output.
*   [`dgf.jax.layers.ProjectorConfig`](api/dgf-jax-layers.md#section-projectorconfig): Makeable Projector config class with sensible defaults.
*   [`dgf.jax.layers.ResidualMLPV2`](api/dgf-jax-layers.md#section-residualmlpv2): A residual MLP layer. See ResidualMLPV2Config.
*   [`dgf.jax.layers.ResidualMLPV2Config`](api/dgf-jax-layers.md#section-residualmlpv2config): A residual MLP layer.
*   [`dgf.jax.layers.identity`](api/dgf-jax-layers.md#section-identity): Returns a GenericBlockConfig that acts as an identity block.
*   [`dgf.jax.layers.ingest_feature`](api/dgf-jax-layers.md#section-ingest-feature): Returns a GenericBlockConfig for feature ingestion.
*   [`dgf.jax.layers.modern_residual_mlp`](api/dgf-jax-layers.md#section-modern-residual-mlp): Returns a GenericBlockConfig for a modern residual MLP.
*   [`dgf.jax.layers.sequential_mlp`](api/dgf-jax-layers.md#section-sequential-mlp): Returns a GenericBlockConfig for a sequential MLP.






### Module `dgf.learning`    # {: #section-dgf-learning}

Top-level learning module.

*   [`dgf.learning.LinkPredictionModel`](api/dgf-learning.md#section-linkpredictionmodel): The user-visible returned model object for edge prediction.
*   [`dgf.learning.Model`](api/dgf-learning.md#section-model): A generic model from the 10-lines of code API.
*   [`dgf.learning.NodePredictionModel`](api/dgf-learning.md#section-nodepredictionmodel): The user-visible returned model object.
*   [`dgf.learning.load_model`](api/dgf-learning.md#section-load-model): Loads a model previously saved with `model.save()`.
*   [`dgf.learning.train_link_model`](api/dgf-learning.md#section-train-link-model): Trains a supervised Graph Neural Network model for edge prediction.
*   [`dgf.learning.train_node_model`](api/dgf-learning.md#section-train-node-model): Trains a supervised Graph Neural Network model for node-level prediction.




### Module `dgf.plot`    # {: #section-dgf-plot}

Functions to plot graphs, schemas, and other graph-related data.

*   [`dgf.plot.plot_graph`](api/dgf-plot.md#section-plot-graph): Plots an in-memory graph.
*   [`dgf.plot.plot_nx_graph`](api/dgf-plot.md#section-plot-nx-graph): Helper function to draw an nx graph.
*   [`dgf.plot.plot_schema`](api/dgf-plot.md#section-plot-schema): Plots the graphschema's meta-graph (i.e., its nodesets and edgesets).




### Module `dgf.print`    # {: #section-dgf-print}

Functions for printing structures.

*   [`dgf.print.padding`](api/dgf-print.md#section-padding): Generates a human-readable string representation of a graph padding.
*   [`dgf.print.sampling_plan`](api/dgf-print.md#section-sampling-plan): Generates a human-readable tree representation of a sampling plan.
*   [`dgf.print.schema`](api/dgf-print.md#section-schema): Generates a human-readable string representation of a graph schema.




### Module `dgf.sampling`    # {: #section-dgf-sampling}

Functions and classes to extract subsets of graphs for GNN training.

*   [`dgf.sampling.Sampler`](api/dgf-sampling.md#section-sampler): Sampler for generating subgraphs from an in-memory graph.
*   [`dgf.sampling.SamplingPlan`](api/dgf-sampling.md#section-samplingplan): Defines a complex sampling config.
*   [`dgf.sampling.SimpleSamplingConfig`](api/dgf-sampling.md#section-simplesamplingconfig): Configuration for simple neighborhood sampling.
*   [`dgf.sampling.SpannerGraphSampler`](api/dgf-sampling.md#section-spannergraphsampler): Sampler that executes queries on Spanner directly to fetch subgraphs.
*   [`dgf.sampling.create_graph_spanner_sampler`](api/dgf-sampling.md#section-create-graph-spanner-sampler): Creates a SpannerGraphSampler instance.
*   [`dgf.sampling.create_sampler`](api/dgf-sampling.md#section-create-sampler): Creates an in-memory sampler.
*   [`dgf.sampling.extract_beam_nodes_ids`](api/dgf-sampling.md#section-extract-beam-nodes-ids): Extracts all the node ids of a given nodeset.
*   [`dgf.sampling.sample_with_beam_semi_distributed_sampler`](api/dgf-sampling.md#section-sample-with-beam-semi-distributed-sampler): Samples subgraphs from a distributed graph using a semi-distributed algo.
*   [`dgf.sampling.sample_with_beam_semi_distributed_sampler_v2`](api/dgf-sampling.md#section-sample-with-beam-semi-distributed-sampler-v2): Samples subgraphs from a distributed graph using a semi-distributed algo.
*   [`dgf.sampling.simple_sampling_config_to_sampling_plan`](api/dgf-sampling.md#section-simple-sampling-config-to-sampling-plan): Converts a SimpleSamplingConfig to a more general SamplingPlan.




### Module `dgf.train`    # {: #section-dgf-train}

Functions and classes to train core GNN models.

*   [`dgf.train.EmbedNodesetFeaturesModule`](api/dgf-train.md#section-embednodesetfeaturesmodule): A FLAX module to transform a set of features into a fixed-size embedding.




### Module `dgf.transform`    # {: #section-dgf-transform}

Transforms graph data into other graph structures or formats.

*   [`dgf.transform.AutoNormalizeConfig`](api/dgf-transform.md#section-autonormalizeconfig): Configuration for automatic feature normalization for GNNs.
*   [`dgf.transform.ContainsLabelPredicate`](api/dgf-transform.md#section-containslabelpredicate): Predicate for filtering subgraphs if they have a positive label.
*   [`dgf.transform.DictionaryIndexNormalizer`](api/dgf-transform.md#section-dictionaryindexnormalizer): Normalizes features by mapping dictionary keys to their integer indices.
*   [`dgf.transform.GNNDatasetPreparator`](api/dgf-transform.md#section-gnndatasetpreparator): Generates graph samples to train node prediction models.
*   [`dgf.transform.GraphNormalizer`](api/dgf-transform.md#section-graphnormalizer): Applies a collection of individual AbstractFeatureNormalizer on a graph.
*   [`dgf.transform.GraphNormalizerConfig`](api/dgf-transform.md#section-graphnormalizerconfig): Raw information of a GraphNormalizer for easy serialization.
*   [`dgf.transform.IdentityNormalizer`](api/dgf-transform.md#section-identitynormalizer): A normalizer that simply pass a feature without changing it.
*   [`dgf.transform.NumNodesPredicate`](api/dgf-transform.md#section-numnodespredicate): Predicate for filtering by number of nodes.
*   [`dgf.transform.SoftQuantileNormalizer`](api/dgf-transform.md#section-softquantilenormalizer): Normalizes a numerical feature by replacing it with its soft quantile -0.5.
*   [`dgf.transform.apply_feature`](api/dgf-transform.md#section-apply-feature): Applies feature processors to the node and edge sets of a graph.
*   [`dgf.transform.auto_normalize`](api/dgf-transform.md#section-auto-normalize): Create a generally good GraphNormalizer from feature statistics.
*   [`dgf.transform.batch_indices_generator`](api/dgf-transform.md#section-batch-indices-generator): Generates batches of indices.
*   [`dgf.transform.drop_edge_features`](api/dgf-transform.md#section-drop-edge-features): Drops all edge features from a graph and its schema.
*   [`dgf.transform.drop_edge_features_from_schema`](api/dgf-transform.md#section-drop-edge-features-from-schema): Drops all edge features from a schema.
*   [`dgf.transform.filter_graph`](api/dgf-transform.md#section-filter-graph): Creates an in-memory graph with a subset of nodesets/edgesets/features.
*   [`dgf.transform.filter_graphs`](api/dgf-transform.md#section-filter-graphs): Filters a sequence of graphs based on user defined predicates.
*   [`dgf.transform.filter_schema`](api/dgf-transform.md#section-filter-schema): Extracts a subset of the nodesets/edgesets/features from a schema.
*   [`dgf.transform.homogeneous_graph_piece_to_nx`](api/dgf-transform.md#section-homogeneous-graph-piece-to-nx): Convert InMemoryGraph to an nx.Graph object.
*   [`dgf.transform.homogenize`](api/dgf-transform.md#section-homogenize): Homogenizes a heterogeneous graph into a homogeneous one.
*   [`dgf.transform.merge_graphs`](api/dgf-transform.md#section-merge-graphs): Merges multiple `InMemoryGraph` instances into a single graph.
*   [`dgf.transform.propagate_timestamp_to_edges`](api/dgf-transform.md#section-propagate-timestamp-to-edges): Propagates timestamps from nodes to edges.
*   [`dgf.transform.remove_padding_sentinels`](api/dgf-transform.md#section-remove-padding-sentinels): Removes the sentinel nodes and edges added by `merge_graphs`.
*   [`dgf.transform.table2graph`](api/dgf-transform.md#section-table2graph): Converts a table (dict of arrays or DataFrame) into an InMemoryGraph and Schema.




### Module `dgf.validate`    # {: #section-dgf-validate}

Functions to validate graph data.

*   [`dgf.validate.validate_graph`](api/dgf-validate.md#section-validate-graph): Validates an in memory graph object.




### Module `dgf.beam`    # {: #section-dgf-beam}

Apache Beam-related functions and classes.

*   [`dgf.beam.program_started`](api/dgf-beam.md#section-program-started): Call this function at the beginning of all GraphFlow Beam jobs.
*   [`dgf.beam.runner_from_name`](api/dgf-beam.md#section-runner-from-name): Returns a Beam runner based on the provided name.
*   [`dgf.beam.runner_from_options`](api/dgf-beam.md#section-runner-from-options): Returns a Beam runner based on the provided options.


#### Module `dgf.beam.analyse`    # {: #section-dgf-beam-analyse}

Functions to analyze graphs using Beam, e.g., feature and graph statistics.

*   [`dgf.beam.analyse.feature_statistics`](api/dgf-beam-analyse.md#section-feature-statistics): Computes the feature statistics for a distributed Graph.
*   [`dgf.beam.analyse.feature_statistics_from_graphs`](api/dgf-beam-analyse.md#section-feature-statistics-from-graphs): Computes the feature statistics for a set of InMemoryGraphs.




#### Module `dgf.beam.data`    # {: #section-dgf-beam-data}

Classes that represent graph data. Contains no functions or algorithms.

*   [`dgf.beam.data.Edge`](api/dgf-beam-data.md#section-edge): A single flat edge.
*   [`dgf.beam.data.Graph`](api/dgf-beam-data.md#section-graph): A (potentially distributed) heterogeneous graph.
*   [`dgf.beam.data.HeterogeniousGraph`](api/dgf-beam-data.md#section-heterogeniousgraph): A (potentially distributed) heterogeneous graph.
*   [`dgf.beam.data.HomogeneousGraph`](api/dgf-beam-data.md#section-homogeneousgraph): A (potentially distributed) homogeneous graph.
*   [`dgf.beam.data.KeyedInMemoryGraph`](api/dgf-beam-data.md#section-keyedinmemorygraph): KeyedInMemoryGraph(key, graph)
*   [`dgf.beam.data.Node`](api/dgf-beam-data.md#section-node): Node(id: bytes | int, features: Optional[Dict[str, numpy.ndarray]] = None)




#### Module `dgf.beam.io`    # {: #section-dgf-beam-io}

Functions to read and write graphs, schemas, and related data using Beam.

*   [`dgf.beam.io.CreateSpannerTables`](api/dgf-beam-io.md#section-createspannertables): Creates Spanner tables for a graph schema.
*   [`dgf.beam.io.read_bigquery_graph`](api/dgf-beam-io.md#section-read-bigquery-graph): Read BigQuery Graph via Beam and return a distributed GraphFlow graph.
*   [`dgf.beam.io.read_graph`](api/dgf-beam-io.md#section-read-graph): Reads a GF graph into a distributed graph.
*   [`dgf.beam.io.read_graphai_hgraph`](api/dgf-beam-io.md#section-read-graphai-hgraph): Reads a distributed HGraph using Beam.
*   [`dgf.beam.io.read_spanner_graph`](api/dgf-beam-io.md#section-read-spanner-graph): Read Spanner Graph via Beam and return a distributed GraphFlow graph.
*   [`dgf.beam.io.read_tfgnn_graphs`](api/dgf-beam-io.md#section-read-tfgnn-graphs): Read a collection of TF GNN Graphs.
*   [`dgf.beam.io.write_edge_set_to_spanner`](api/dgf-beam-io.md#section-write-edge-set-to-spanner): Writes an edge set to a Spanner table using SpannerInsertOrUpdate.
*   [`dgf.beam.io.write_feature_statistics`](api/dgf-beam-io.md#section-write-feature-statistics): Writes a beam pcollection of feature statistics to disk in json format.
*   [`dgf.beam.io.write_graph`](api/dgf-beam-io.md#section-write-graph): Writes a GF Graph from a distributed graph (beam).
*   [`dgf.beam.io.write_graphai_hgraph`](api/dgf-beam-io.md#section-write-graphai-hgraph): Initializes the WriteToHGraph PTransform.
*   [`dgf.beam.io.write_node_set_to_spanner`](api/dgf-beam-io.md#section-write-node-set-to-spanner): Writes a node set to a Spanner table using SpannerInsertOrUpdate.
*   [`dgf.beam.io.write_spanner`](api/dgf-beam-io.md#section-write-spanner): Writes a heterogeneous graph to Spanner.
*   [`dgf.beam.io.write_tfgnn_graphs`](api/dgf-beam-io.md#section-write-tfgnn-graphs): Writes a collection of TF Graph Samples on disk.




#### Module `dgf.beam.sampling`    # {: #section-dgf-beam-sampling}

Functions to extract subsets of graphs for GNN training using Beam.

*   [`dgf.beam.sampling.extract_nodes_ids`](api/dgf-beam-sampling.md#section-extract-nodes-ids): Extracts all the node ids of a given nodeset.
*   [`dgf.beam.sampling.semi_distributed_sampler_v1`](api/dgf-beam-sampling.md#section-semi-distributed-sampler-v1): Samples subgraphs from a distributed graph using a semi-distributed algo.
*   [`dgf.beam.sampling.semi_distributed_sampler_v2`](api/dgf-beam-sampling.md#section-semi-distributed-sampler-v2): Samples subgraphs from a distributed graph using a semi-distributed algo.




#### Module `dgf.beam.transform`    # {: #section-dgf-beam-transform}

Transforms graph data into other graph formats using Beam.

*   [`dgf.beam.transform.reverse_edges`](api/dgf-beam-transform.md#section-reverse-edges): Reverse the direction of edges in a graph.








