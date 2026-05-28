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

*   [`dgf.analyse.feature_statistics`](api_details.md#section-dgf-analyse-feature-statistics): Computes the feature stats from a single graph.
*   [`dgf.analyse.feature_statistics_from_graphs`](api_details.md#section-dgf-analyse-feature-statistics-from-graphs): Computes the feature stats from multiple graphs.
*   [`dgf.analyse.padding_from_graph_generator`](api_details.md#section-dgf-analyse-padding-from-graph-generator): Creates a padding configuration from a set of in-memory graphs.
*   [`dgf.analyse.print_schema`](api_details.md#section-dgf-analyse-print-schema): Generates a human-readable string representation of a graph schema.




### Module `dgf.convert`    # {: #section-dgf-convert}

Converts object formats, e.g., a graph to a Sparse Deferred struct.

*   [`dgf.convert.graph_dict_to_graph`](api_details.md#section-dgf-convert-graph-dict-to-graph): Converts a TF GNN Graph Sample Dict to an InMemoryGraph.
*   [`dgf.convert.graph_to_jax_graph`](api_details.md#section-dgf-convert-graph-to-jax-graph): Converts a (NumPy) in-memory graph into a JAX in-memory graph.
*   [`dgf.convert.graph_to_networkx`](api_details.md#section-dgf-convert-graph-to-networkx): Converts an InMemoryGraph into a NetworkX MultiDiGraph.
*   [`dgf.convert.graph_to_serialized_tfgnn_graph`](api_details.md#section-dgf-convert-graph-to-serialized-tfgnn-graph): Converts an InMemoryGraph into a serialized TF-GNN graph sample proto.
*   [`dgf.convert.graph_to_sparse_deferred_struct`](api_details.md#section-dgf-convert-graph-to-sparse-deferred-struct): Converts an in-memory graph into a Sparse Deferred struct.
*   [`dgf.convert.graph_to_tf_graph`](api_details.md#section-dgf-convert-graph-to-tf-graph): Converts a graph to a TF in-memory graph.
*   [`dgf.convert.graph_to_tfgnn_graph`](api_details.md#section-dgf-convert-graph-to-tfgnn-graph): Converts an InMemoryGraph to a TF GNN Graph Sample.
*   [`dgf.convert.graph_to_tfgnn_graph_dict`](api_details.md#section-dgf-convert-graph-to-tfgnn-graph-dict): Converts an InMemoryGraph to a TF GNN Graph Sample Dict.
*   [`dgf.convert.graphs_to_serialized_tfgnn_graphs`](api_details.md#section-dgf-convert-graphs-to-serialized-tfgnn-graphs): Converts a sequence of InMemoryGraphs into serialized TF-GNN graph sample protos.
*   [`dgf.convert.networkx_to_graph`](api_details.md#section-dgf-convert-networkx-to-graph): Converts a NetworkX graph into an InMemoryGraph and its schema.
*   [`dgf.convert.schema_to_spanner_ddl`](api_details.md#section-dgf-convert-schema-to-spanner-ddl): Converts a GraphSchema to a string of CREATE statements for Spanner.
*   [`dgf.convert.schema_to_sparse_deferred_schema`](api_details.md#section-dgf-convert-schema-to-sparse-deferred-schema): Converts a DGF `GraphSchema` into a Sparse Deferred schema.
*   [`dgf.convert.schema_to_tfgnn_schema`](api_details.md#section-dgf-convert-schema-to-tfgnn-schema): Converts a GraphSchema object into a TF-GNN schema proto.
*   [`dgf.convert.sparse_deferred_struct_to_graph`](api_details.md#section-dgf-convert-sparse-deferred-struct-to-graph): Converts a Sparse Deferred struct into an in-memory graph.
*   [`dgf.convert.tf_graph_dict_to_tf_graph`](api_details.md#section-dgf-convert-tf-graph-dict-to-tf-graph): Converts a flattened TFInMemoryGraphDict back into a TFInMemoryGraph.
*   [`dgf.convert.tf_graph_to_tf_graph_dict`](api_details.md#section-dgf-convert-tf-graph-to-tf-graph-dict): Converts a TFInMemoryGraph into a flattened TFInMemoryGraphDict.
*   [`dgf.convert.tfgnn_graph_to_graph`](api_details.md#section-dgf-convert-tfgnn-graph-to-graph): Converts a TF GNN Graph Sample to an InMemoryGraph.
*   [`dgf.convert.tfgnn_schema_to_schema`](api_details.md#section-dgf-convert-tfgnn-schema-to-schema): Converts a TF-GNN schema proto into a GraphSchema object.




### Module `dgf.data`    # {: #section-dgf-data}

Classes that represent graph data. Contains no functions or algorithms.

*   [`dgf.data.EdgeSchema`](api_details.md#section-dgf-data-edgeschema): EdgeSchema(source: str, target: str, features: Dict[str, dgf.src.data.schema.FeatureSchema] = <factory>)
*   [`dgf.data.EdgeSetPadding`](api_details.md#section-dgf-data-edgesetpadding): EdgeSetPadding(num_edges: int)
*   [`dgf.data.FeatureFormat`](api_details.md#section-dgf-data-featureformat): How a value is represented / stored.
*   [`dgf.data.FeatureSchema`](api_details.md#section-dgf-data-featureschema): Schema for a single feature.
*   [`dgf.data.FeatureSemantic`](api_details.md#section-dgf-data-featuresemantic): How a value should be interpreted.
*   [`dgf.data.FeatureSetStatistics`](api_details.md#section-dgf-data-featuresetstatistics): Statistics for a set of features.
*   [`dgf.data.FeatureStatistics`](api_details.md#section-dgf-data-featurestatistics): Statistics for a feature.
*   [`dgf.data.GraphFeatureStatistics`](api_details.md#section-dgf-data-graphfeaturestatistics): Statistics about the features in a graph.
*   [`dgf.data.GraphSchema`](api_details.md#section-dgf-data-graphschema): GraphSchema(node_sets: Dict[str, dgf.src.data.schema.NodeSchema], edge_sets: Dict[str, dgf.src.data.schema.EdgeSchema])
*   [`dgf.data.GraphSchemaFilter`](api_details.md#section-dgf-data-graphschemafilter): Filters a GraphSchema to sub-select node sets, edge sets, and features.
*   [`dgf.data.GraphSchemaV2`](api_details.md#section-dgf-data-graphschemav2): GraphSchema(node_sets: Dict[str, dgf.src.data.schema.NodeSchema], edge_sets: Dict[str, dgf.src.data.schema.EdgeSchema])
*   [`dgf.data.InMemoryEdgeSet`](api_details.md#section-dgf-data-inmemoryedgeset): An Edge Set.
*   [`dgf.data.InMemoryGraph`](api_details.md#section-dgf-data-inmemorygraph): An in-memory generic graph.
*   [`dgf.data.InMemoryNodeSet`](api_details.md#section-dgf-data-inmemorynodeset): A Node Set.
*   [`dgf.data.JaxInMemoryEdgeSet`](api_details.md#section-dgf-data-jaxinmemoryedgeset): An Edge Set.
*   [`dgf.data.JaxInMemoryGraph`](api_details.md#section-dgf-data-jaxinmemorygraph): An in-memory generic graph.
*   [`dgf.data.JaxInMemoryNodeSet`](api_details.md#section-dgf-data-jaxinmemorynodeset): A Node Set.
*   [`dgf.data.NodeSchema`](api_details.md#section-dgf-data-nodeschema): NodeSchema(features: Dict[str, dgf.src.data.schema.FeatureSchema] = <factory>)
*   [`dgf.data.NodeSetPadding`](api_details.md#section-dgf-data-nodesetpadding): NodeSetPadding(num_nodes: int)
*   [`dgf.data.Padding`](api_details.md#section-dgf-data-padding): Information to pad a graph.
*   [`dgf.data.TFInMemoryEdgeSet`](api_details.md#section-dgf-data-tfinmemoryedgeset): An Edge Set.
*   [`dgf.data.TFInMemoryGraph`](api_details.md#section-dgf-data-tfinmemorygraph): An in-memory generic graph.
*   [`dgf.data.TFInMemoryNodeSet`](api_details.md#section-dgf-data-tfinmemorynodeset): A Node Set.




### Module `dgf.exception`    # {: #section-dgf-exception}

DGF-specific exceptions.

*   [`dgf.exception.InsufficientPaddingError`](api_details.md#section-dgf-exception-insufficientpaddingerror): Inappropriate argument value (of correct type).




### Module `dgf.filesystem`    # {: #section-dgf-filesystem}

GraphFlow unified filesystem API.

*   [`dgf.filesystem.create_gcs_bucket`](api_details.md#section-dgf-filesystem-create-gcs-bucket): Creates a GCS bucket.
*   [`dgf.filesystem.exists`](api_details.md#section-dgf-filesystem-exists): Returns True if the path exists.
*   [`dgf.filesystem.glob`](api_details.md#section-dgf-filesystem-glob): Returns a list of files and directories matching a pattern.
*   [`dgf.filesystem.is_gcs_path`](api_details.md#section-dgf-filesystem-is-gcs-path): Returns True if the path is a Google Cloud Storage (GCS) path.
*   [`dgf.filesystem.makedirs`](api_details.md#section-dgf-filesystem-makedirs): Creates directories if it does not exist.
*   [`dgf.filesystem.open_read`](api_details.md#section-dgf-filesystem-open-read): Opens a file for reading and return a python file handle.
*   [`dgf.filesystem.remove_paths`](api_details.md#section-dgf-filesystem-remove-paths): Removes all the files in parallel.
*   [`dgf.filesystem.rename`](api_details.md#section-dgf-filesystem-rename): Renames (moves) a file or directory from old_path to new_path.
*   [`dgf.filesystem.rmtree`](api_details.md#section-dgf-filesystem-rmtree): Recursively removes a directory and its contents.




### Module `dgf.generate`    # {: #section-dgf-generate}

Tools to generate synthetic data.

*   [`dgf.generate.EdgeNeighborGenerator`](api_details.md#section-dgf-generate-edgeneighborgenerator): Generates the idx of the positive and negative pairs of nodes.
*   [`dgf.generate.RandomNegativeSampler`](api_details.md#section-dgf-generate-randomnegativesampler): Replace the target node with a random node.
*   [`dgf.generate.RandomWalkNegativeSampler`](api_details.md#section-dgf-generate-randomwalknegativesampler): Replace the target node with a random walk generated node.
*   [`dgf.generate.SyntheticGraphSampleConfig`](api_details.md#section-dgf-generate-syntheticgraphsampleconfig): Configuration for generating synthetic graph samples.
*   [`dgf.generate.generate_synthetic_graph_sample`](api_details.md#section-dgf-generate-generate-synthetic-graph-sample): Generates a single synthetic graph sample based on a sampling plan.
*   [`dgf.generate.write_synthetic_graph_sample_as_tfgnn_graphs`](api_details.md#section-dgf-generate-write-synthetic-graph-sample-as-tfgnn-graphs): Generates and writes synthetic graph samples as TF-GNN graphs.




### Module `dgf.io`    # {: #section-dgf-io}

Functions to read and write graphs, schemas, and related data.

*   [`dgf.io.cache`](api_details.md#section-dgf-io-cache): Returns and caches the variable(s) created by "create_fn".
*   [`dgf.io.create_spanner_tables_from_graph_schema`](api_details.md#section-dgf-io-create-spanner-tables-from-graph-schema): Creates Spanner tables for a graph schema.
*   [`dgf.io.export_bigquery_to_disk`](api_details.md#section-dgf-io-export-bigquery-to-disk): Reads a BigQuery Graph in-process and returns a GraphFlow in-memory graph.
*   [`dgf.io.fetch_graphland_graph`](api_details.md#section-dgf-io-fetch-graphland-graph): Downloads and loads a Graphland dataset into memory.
*   [`dgf.io.fetch_ogb_graph`](api_details.md#section-dgf-io-fetch-ogb-graph): Downloads and loads an OGB node property prediction dataset into memory.
*   [`dgf.io.read_bigquery_graph`](api_details.md#section-dgf-io-read-bigquery-graph): Reads a BigQuery Graph in-process and returns a GraphFlow in-memory graph.
*   [`dgf.io.read_bigquery_graph_schema`](api_details.md#section-dgf-io-read-bigquery-graph-schema): Reads the schema of a BigQuery graph into a GF schema.
*   [`dgf.io.read_feature_statistics`](api_details.md#section-dgf-io-read-feature-statistics): Reads feature statistics from disk in a JSON format.
*   [`dgf.io.read_graph`](api_details.md#section-dgf-io-read-graph): Reads a GF graph from a directory to an in-memory graph.
*   [`dgf.io.read_graphai_hgraph`](api_details.md#section-dgf-io-read-graphai-hgraph): Reads an on-disk HGraph into an in-memory representation.
*   [`dgf.io.read_schema`](api_details.md#section-dgf-io-read-schema): Loads graph schema from disk in a json format.
*   [`dgf.io.read_spanner_graph`](api_details.md#section-dgf-io-read-spanner-graph): Reads a Spanner Graph in-process and returns a GraphFlow in-memory graph.
*   [`dgf.io.read_spanner_graph_schema`](api_details.md#section-dgf-io-read-spanner-graph-schema): Reads the schema of a Spanner Graph.
*   [`dgf.io.read_text_proto`](api_details.md#section-dgf-io-read-text-proto): Read a proto from disk in text format.
*   [`dgf.io.read_tfgnn_graphs`](api_details.md#section-dgf-io-read-tfgnn-graphs): Reads a set of in-memory graphs from disk stored as TF Examples.
*   [`dgf.io.write_feature_statistics`](api_details.md#section-dgf-io-write-feature-statistics): Saves feature statistics to disk in a json format.
*   [`dgf.io.write_graph`](api_details.md#section-dgf-io-write-graph): Writes an in-memory graph and schema to a GF Graph directory.
*   [`dgf.io.write_schema`](api_details.md#section-dgf-io-write-schema): Saves graph schema to disk in a json format.
*   [`dgf.io.write_text_proto`](api_details.md#section-dgf-io-write-text-proto): Writes a proto to disk in text format.
*   [`dgf.io.write_tfgnn_graphs`](api_details.md#section-dgf-io-write-tfgnn-graphs): Writes a set of in-memory graphs to disk as TF Examples.




### Module `dgf.jax`    # {: #section-dgf-jax}

Machine Learning and Graph Neural Networks using JAX.

*   [`dgf.jax.JaxBaseConfig`](api_details.md#section-dgf-jax-jaxbaseconfig): Base class for a GNN implemented in JAX.
*   [`dgf.jax.get_activation`](api_details.md#section-dgf-jax-get-activation): Get an activation function by (string) name.
*   [`dgf.jax.jnp_dtype_from_string`](api_details.md#section-dgf-jax-jnp-dtype-from-string): Return a JAX numpy type from a string name.
*   [`dgf.jax.jnp_name_from_dtype`](api_details.md#section-dgf-jax-jnp-name-from-dtype): Return a string name for a jnp.dtype object.
*   [`dgf.jax.train`](api_details.md#section-dgf-jax-train): Trains a Flax module with a flexible and feature-rich training loop.


#### Module `dgf.jax.layers`    # {: #section-dgf-jax-layers}

Flax modules implementing low level GNN operations.

*   [`dgf.jax.layers.ClassificationHead`](api_details.md#section-dgf-jax-layers-classificationhead): Simple classification head.
*   [`dgf.jax.layers.ClassificationHeadConfig`](api_details.md#section-dgf-jax-layers-classificationheadconfig): Configuration for a classification head.
*   [`dgf.jax.layers.ConditionalGIN`](api_details.md#section-dgf-jax-layers-conditionalgin): Conditional GIN with a labeling trick: https://arxiv.org/abs/2106.06935.
*   [`dgf.jax.layers.EmbedAndHomogenizeGraph`](api_details.md#section-dgf-jax-layers-embedandhomogenizegraph): Convert a heterogeneous graph into a homogeneous one.
*   [`dgf.jax.layers.EmbedAndHomogenizeGraphConfig`](api_details.md#section-dgf-jax-layers-embedandhomogenizegraphconfig): Config for EmbedAndHomogenizeGraph.
*   [`dgf.jax.layers.EmbedFeatureSet`](api_details.md#section-dgf-jax-layers-embedfeatureset): Computes a fixed sized dense embedding for a set of feature values.
*   [`dgf.jax.layers.EmbedFeatureSetConfig`](api_details.md#section-dgf-jax-layers-embedfeaturesetconfig): Configuration for the EmbedFeatureSet layer.
*   [`dgf.jax.layers.EmbedGraph`](api_details.md#section-dgf-jax-layers-embedgraph): Compute a fixed sized dense embedding for all the features in a graph.
*   [`dgf.jax.layers.EmbedGraphConfig`](api_details.md#section-dgf-jax-layers-embedgraphconfig): Configuration for "EmbedGraph".
*   [`dgf.jax.layers.GCN`](api_details.md#section-dgf-jax-layers-gcn): Graph convolutional network: https://arxiv.org/pdf/1609.02907.pdf.
*   [`dgf.jax.layers.GCNConfig`](api_details.md#section-dgf-jax-layers-gcnconfig): Makeable GCN config class with sensible defaults.
*   [`dgf.jax.layers.GIN`](api_details.md#section-dgf-jax-layers-gin): Graph isomorphism network: https://arxiv.org/pdf/1810.00826.pdf.
*   [`dgf.jax.layers.GINConfig`](api_details.md#section-dgf-jax-layers-ginconfig): Makeable GIN config class with sensible defaults.
*   [`dgf.jax.layers.GenericBlock`](api_details.md#section-dgf-jax-layers-genericblock): A generic configurable neural network block.
*   [`dgf.jax.layers.GenericBlockConfig`](api_details.md#section-dgf-jax-layers-genericblockconfig): Configuration for a generic block parsed from a string.
*   [`dgf.jax.layers.HeterogeneousGraphAttentionNetwork`](api_details.md#section-dgf-jax-layers-heterogeneousgraphattentionnetwork): A single layer of heterogeneous Graph Attention Network.
*   [`dgf.jax.layers.HeterogeneousGraphAttentionNetworkConfig`](api_details.md#section-dgf-jax-layers-heterogeneousgraphattentionnetworkconfig): Configuration for HeterogeneousGraphAttentionNetwork.
*   [`dgf.jax.layers.HeterogeneousGraphConvolution`](api_details.md#section-dgf-jax-layers-heterogeneousgraphconvolution): A single layer of heterogeneous Graph Neural Network message passing.
*   [`dgf.jax.layers.HeterogeneousGraphConvolutionConfig`](api_details.md#section-dgf-jax-layers-heterogeneousgraphconvolutionconfig): Configuration for HeterogeneousGraphConvolution.
*   [`dgf.jax.layers.MLP`](api_details.md#section-dgf-jax-layers-mlp): A generic MLP followed by a linear layer.
*   [`dgf.jax.layers.MPNN`](api_details.md#section-dgf-jax-layers-mpnn): Message-Passing Neural Network: https://arxiv.org/abs/1704.01212.
*   [`dgf.jax.layers.MPNNConfig`](api_details.md#section-dgf-jax-layers-mpnnconfig): Makeable MPNN config class with sensible defaults.
*   [`dgf.jax.layers.Projector`](api_details.md#section-dgf-jax-layers-projector): Simple wrapper around the generic MLP layer for graph input/output.
*   [`dgf.jax.layers.ProjectorConfig`](api_details.md#section-dgf-jax-layers-projectorconfig): Makeable Projector config class with sensible defaults.
*   [`dgf.jax.layers.ResidualMLPV2`](api_details.md#section-dgf-jax-layers-residualmlpv2): A residual MLP layer. See ResidualMLPV2Config.
*   [`dgf.jax.layers.ResidualMLPV2Config`](api_details.md#section-dgf-jax-layers-residualmlpv2config): A residual MLP layer.
*   [`dgf.jax.layers.identity`](api_details.md#section-dgf-jax-layers-identity): Returns a GenericBlockConfig that acts as an identity block.
*   [`dgf.jax.layers.ingest_feature`](api_details.md#section-dgf-jax-layers-ingest-feature): Returns a GenericBlockConfig for feature ingestion.
*   [`dgf.jax.layers.modern_residual_mlp`](api_details.md#section-dgf-jax-layers-modern-residual-mlp): Returns a GenericBlockConfig for a modern residual MLP.
*   [`dgf.jax.layers.sequential_mlp`](api_details.md#section-dgf-jax-layers-sequential-mlp): Returns a GenericBlockConfig for a sequential MLP.






### Module `dgf.learning`    # {: #section-dgf-learning}

Top-level learning module.

*   [`dgf.learning.LinkPredictionModel`](api_details.md#section-dgf-learning-linkpredictionmodel): The user-visible returned model object for edge prediction.
*   [`dgf.learning.Model`](api_details.md#section-dgf-learning-model): A generic model from the 10-lines of code API.
*   [`dgf.learning.NodePredictionModel`](api_details.md#section-dgf-learning-nodepredictionmodel): The user-visible returned model object.
*   [`dgf.learning.load_model`](api_details.md#section-dgf-learning-load-model): Loads a model previously saved with `model.save()`.
*   [`dgf.learning.train_link_model`](api_details.md#section-dgf-learning-train-link-model): Trains a supervised Graph Neural Network model for edge prediction.
*   [`dgf.learning.train_node_model`](api_details.md#section-dgf-learning-train-node-model): Trains a supervised Graph Neural Network model for node-level prediction.




### Module `dgf.plot`    # {: #section-dgf-plot}

Functions to plot graphs, schemas, and other graph-related data.

*   [`dgf.plot.plot_graph`](api_details.md#section-dgf-plot-plot-graph): Plots an in-memory graph.
*   [`dgf.plot.plot_nx_graph`](api_details.md#section-dgf-plot-plot-nx-graph): Helper function to draw an nx graph.
*   [`dgf.plot.plot_schema`](api_details.md#section-dgf-plot-plot-schema): Plots the graphschema's meta-graph (i.e., its nodesets and edgesets).




### Module `dgf.print`    # {: #section-dgf-print}

Functions for printing structures.

*   [`dgf.print.padding`](api_details.md#section-dgf-print-padding): Generates a human-readable string representation of a graph padding.
*   [`dgf.print.sampling_plan`](api_details.md#section-dgf-print-sampling-plan): Generates a human-readable tree representation of a sampling plan.
*   [`dgf.print.schema`](api_details.md#section-dgf-print-schema): Generates a human-readable string representation of a graph schema.




### Module `dgf.sampling`    # {: #section-dgf-sampling}

Functions and classes to extract subsets of graphs for GNN training.

*   [`dgf.sampling.Sampler`](api_details.md#section-dgf-sampling-sampler): Sampler for generating subgraphs from an in-memory graph.
*   [`dgf.sampling.SamplingPlan`](api_details.md#section-dgf-sampling-samplingplan): Defines a complex sampling config.
*   [`dgf.sampling.SimpleSamplingConfig`](api_details.md#section-dgf-sampling-simplesamplingconfig): Configuration for simple neighborhood sampling.
*   [`dgf.sampling.SpannerGraphSampler`](api_details.md#section-dgf-sampling-spannergraphsampler): Sampler that executes queries on Spanner directly to fetch subgraphs.
*   [`dgf.sampling.create_graph_spanner_sampler`](api_details.md#section-dgf-sampling-create-graph-spanner-sampler): Creates a SpannerGraphSampler instance.
*   [`dgf.sampling.create_sampler`](api_details.md#section-dgf-sampling-create-sampler): Creates an in-memory sampler.
*   [`dgf.sampling.extract_beam_nodes_ids`](api_details.md#section-dgf-sampling-extract-beam-nodes-ids): Extracts all the node ids of a given nodeset.
*   [`dgf.sampling.sample_with_beam_semi_distributed_sampler`](api_details.md#section-dgf-sampling-sample-with-beam-semi-distributed-sampler): Samples subgraphs from a distributed graph using a semi-distributed algo.
*   [`dgf.sampling.sample_with_beam_semi_distributed_sampler_v2`](api_details.md#section-dgf-sampling-sample-with-beam-semi-distributed-sampler-v2): Samples subgraphs from a distributed graph using a semi-distributed algo.
*   [`dgf.sampling.simple_sampling_config_to_sampling_plan`](api_details.md#section-dgf-sampling-simple-sampling-config-to-sampling-plan): Converts a SimpleSamplingConfig to a more general SamplingPlan.




### Module `dgf.train`    # {: #section-dgf-train}

Functions and classes to train core GNN models.

*   [`dgf.train.EmbedNodesetFeaturesModule`](api_details.md#section-dgf-train-embednodesetfeaturesmodule): A FLAX module to transform a set of features into a fixed-size embedding.




### Module `dgf.transform`    # {: #section-dgf-transform}

Transforms graph data into other graph structures or formats.

*   [`dgf.transform.AutoNormalizeConfig`](api_details.md#section-dgf-transform-autonormalizeconfig): Configuration for automatic feature normalization for GNNs.
*   [`dgf.transform.ContainsLabelPredicate`](api_details.md#section-dgf-transform-containslabelpredicate): Predicate for filtering subgraphs if they have a positive label.
*   [`dgf.transform.DictionaryIndexNormalizer`](api_details.md#section-dgf-transform-dictionaryindexnormalizer): Normalizes features by mapping dictionary keys to their integer indices.
*   [`dgf.transform.GNNDatasetPreparator`](api_details.md#section-dgf-transform-gnndatasetpreparator): Generates graph samples to train node prediction models.
*   [`dgf.transform.GraphNormalizer`](api_details.md#section-dgf-transform-graphnormalizer): Applies a collection of individual AbstractFeatureNormalizer on a graph.
*   [`dgf.transform.GraphNormalizerConfig`](api_details.md#section-dgf-transform-graphnormalizerconfig): Raw information of a GraphNormalizer for easy serialization.
*   [`dgf.transform.IdentityNormalizer`](api_details.md#section-dgf-transform-identitynormalizer): A normalizer that simply pass a feature without changing it.
*   [`dgf.transform.NumNodesPredicate`](api_details.md#section-dgf-transform-numnodespredicate): Predicate for filtering by number of nodes.
*   [`dgf.transform.SoftQuantileNormalizer`](api_details.md#section-dgf-transform-softquantilenormalizer): Normalizes a numerical feature by replacing it with its soft quantile -0.5.
*   [`dgf.transform.apply_feature`](api_details.md#section-dgf-transform-apply-feature): Applies feature processors to the node and edge sets of a graph.
*   [`dgf.transform.auto_normalize`](api_details.md#section-dgf-transform-auto-normalize): Create a generally good GraphNormalizer from feature statistics.
*   [`dgf.transform.batch_indices_generator`](api_details.md#section-dgf-transform-batch-indices-generator): Generates batches of indices.
*   [`dgf.transform.drop_edge_features`](api_details.md#section-dgf-transform-drop-edge-features): Drops all edge features from a graph and its schema.
*   [`dgf.transform.drop_edge_features_from_schema`](api_details.md#section-dgf-transform-drop-edge-features-from-schema): Drops all edge features from a schema.
*   [`dgf.transform.filter_graph`](api_details.md#section-dgf-transform-filter-graph): Creates an in-memory graph with a subset of nodesets/edgesets/features.
*   [`dgf.transform.filter_graphs`](api_details.md#section-dgf-transform-filter-graphs): Filters a sequence of graphs based on user defined predicates.
*   [`dgf.transform.filter_schema`](api_details.md#section-dgf-transform-filter-schema): Extracts a subset of the nodesets/edgesets/features from a schema.
*   [`dgf.transform.homogeneous_graph_piece_to_nx`](api_details.md#section-dgf-transform-homogeneous-graph-piece-to-nx): Convert InMemoryGraph to an nx.Graph object.
*   [`dgf.transform.homogenize`](api_details.md#section-dgf-transform-homogenize): Homogenizes a heterogeneous graph into a homogeneous one.
*   [`dgf.transform.merge_graphs`](api_details.md#section-dgf-transform-merge-graphs): Merges multiple `InMemoryGraph` instances into a single graph.
*   [`dgf.transform.propagate_timestamp_to_edges`](api_details.md#section-dgf-transform-propagate-timestamp-to-edges): Propagates timestamps from nodes to edges.
*   [`dgf.transform.remove_padding_sentinels`](api_details.md#section-dgf-transform-remove-padding-sentinels): Removes the sentinel nodes and edges added by `merge_graphs`.
*   [`dgf.transform.table2graph`](api_details.md#section-dgf-transform-table2graph): Converts a table (dict of arrays or DataFrame) into an InMemoryGraph and Schema.




### Module `dgf.validate`    # {: #section-dgf-validate}

Functions to validate graph data.

*   [`dgf.validate.validate_graph`](api_details.md#section-dgf-validate-validate-graph): Validates an in memory graph object.




### Module `dgf.beam`    # {: #section-dgf-beam}

Apache Beam-related functions and classes.

*   [`dgf.beam.program_started`](api_details.md#section-dgf-beam-program-started): Call this function at the beginning of all GraphFlow Beam jobs.
*   [`dgf.beam.runner_from_name`](api_details.md#section-dgf-beam-runner-from-name): Returns a Beam runner based on the provided name.
*   [`dgf.beam.runner_from_options`](api_details.md#section-dgf-beam-runner-from-options): Returns a Beam runner based on the provided options.


#### Module `dgf.beam.analyse`    # {: #section-dgf-beam-analyse}

Functions to analyze graphs using Beam, e.g., feature and graph statistics.

*   [`dgf.beam.analyse.feature_statistics`](api_details.md#section-dgf-beam-analyse-feature-statistics): Computes the feature statistics for a distributed Graph.
*   [`dgf.beam.analyse.feature_statistics_from_graphs`](api_details.md#section-dgf-beam-analyse-feature-statistics-from-graphs): Computes the feature statistics for a set of InMemoryGraphs.




#### Module `dgf.beam.data`    # {: #section-dgf-beam-data}

Classes that represent graph data. Contains no functions or algorithms.

*   [`dgf.beam.data.Edge`](api_details.md#section-dgf-beam-data-edge): A single flat edge.
*   [`dgf.beam.data.Graph`](api_details.md#section-dgf-beam-data-graph): A (potentially distributed) heterogeneous graph.
*   [`dgf.beam.data.HeterogeniousGraph`](api_details.md#section-dgf-beam-data-heterogeniousgraph): A (potentially distributed) heterogeneous graph.
*   [`dgf.beam.data.HomogeneousGraph`](api_details.md#section-dgf-beam-data-homogeneousgraph): A (potentially distributed) homogeneous graph.
*   [`dgf.beam.data.KeyedInMemoryGraph`](api_details.md#section-dgf-beam-data-keyedinmemorygraph): KeyedInMemoryGraph(key, graph)
*   [`dgf.beam.data.Node`](api_details.md#section-dgf-beam-data-node): Node(id: bytes | int, features: Optional[Dict[str, numpy.ndarray]] = None)




#### Module `dgf.beam.io`    # {: #section-dgf-beam-io}

Functions to read and write graphs, schemas, and related data using Beam.

*   [`dgf.beam.io.CreateSpannerTables`](api_details.md#section-dgf-beam-io-createspannertables): Creates Spanner tables for a graph schema.
*   [`dgf.beam.io.read_bigquery_graph`](api_details.md#section-dgf-beam-io-read-bigquery-graph): Read BigQuery Graph via Beam and return a distributed GraphFlow graph.
*   [`dgf.beam.io.read_graph`](api_details.md#section-dgf-beam-io-read-graph): Reads a GF graph into a distributed graph.
*   [`dgf.beam.io.read_graphai_hgraph`](api_details.md#section-dgf-beam-io-read-graphai-hgraph): Reads a distributed HGraph using Beam.
*   [`dgf.beam.io.read_spanner_graph`](api_details.md#section-dgf-beam-io-read-spanner-graph): Read Spanner Graph via Beam and return a distributed GraphFlow graph.
*   [`dgf.beam.io.read_tfgnn_graphs`](api_details.md#section-dgf-beam-io-read-tfgnn-graphs): Read a collection of TF GNN Graphs.
*   [`dgf.beam.io.write_edge_set_to_spanner`](api_details.md#section-dgf-beam-io-write-edge-set-to-spanner): Writes an edge set to a Spanner table using SpannerInsertOrUpdate.
*   [`dgf.beam.io.write_feature_statistics`](api_details.md#section-dgf-beam-io-write-feature-statistics): Writes a beam pcollection of feature statistics to disk in json format.
*   [`dgf.beam.io.write_graph`](api_details.md#section-dgf-beam-io-write-graph): Writes a GF Graph from a distributed graph (beam).
*   [`dgf.beam.io.write_graphai_hgraph`](api_details.md#section-dgf-beam-io-write-graphai-hgraph): Initializes the WriteToHGraph PTransform.
*   [`dgf.beam.io.write_node_set_to_spanner`](api_details.md#section-dgf-beam-io-write-node-set-to-spanner): Writes a node set to a Spanner table using SpannerInsertOrUpdate.
*   [`dgf.beam.io.write_spanner`](api_details.md#section-dgf-beam-io-write-spanner): Writes a heterogeneous graph to Spanner.
*   [`dgf.beam.io.write_tfgnn_graphs`](api_details.md#section-dgf-beam-io-write-tfgnn-graphs): Writes a collection of TF Graph Samples on disk.




#### Module `dgf.beam.sampling`    # {: #section-dgf-beam-sampling}

Functions to extract subsets of graphs for GNN training using Beam.

*   [`dgf.beam.sampling.extract_nodes_ids`](api_details.md#section-dgf-beam-sampling-extract-nodes-ids): Extracts all the node ids of a given nodeset.
*   [`dgf.beam.sampling.semi_distributed_sampler_v1`](api_details.md#section-dgf-beam-sampling-semi-distributed-sampler-v1): Samples subgraphs from a distributed graph using a semi-distributed algo.
*   [`dgf.beam.sampling.semi_distributed_sampler_v2`](api_details.md#section-dgf-beam-sampling-semi-distributed-sampler-v2): Samples subgraphs from a distributed graph using a semi-distributed algo.




#### Module `dgf.beam.transform`    # {: #section-dgf-beam-transform}

Transforms graph data into other graph formats using Beam.

*   [`dgf.beam.transform.reverse_edges`](api_details.md#section-dgf-beam-transform-reverse-edges): Reverse the direction of edges in a graph.








