# (Distributed) GraphFlow API

All the Apache Beam distributed functions / classes are defined in `dgf.beam.*`. All the other (e.g., in-process, in-memory) functions and classes are defined in `dgf.*` directly:


To see the documentation of a function, use the python built-in `help` method or `?` in a colab e.g., `?dgf.io.read_feature_statistics`.

*   `dgf.analyse.*`: Analyze graph e.g. feature statistics, graph statistics.
*   `dgf.convert.*`: Convert object formats e.g. convert a graph to a Sparse Deferred struct.
*   `dgf.data.*`: Classes to represent graph data. Contains no functions / algorithms.
*   `dgf.exception.*`: DGF specific exceptions.
*   `dgf.filesystem.*`: GraphFlow unified filesystem API.
*   `dgf.generate.*`: Tools to generate synthetic data.
*   `dgf.io.*`: Read and write graph, schema, and other related data.
*   `dgf.jax.*`: Machine Learning and Graph Neural Networks using JAX.
*   `dgf.learning.*`: Top level learning module.
*   `dgf.plot.*`: Plot graph, schema, and other graph-related data.
*   `dgf.print.*`: Functions for printing structures.
*   `dgf.sampling.*`: Extracts subsets of graphs for GNN training.
*   `dgf.train.*`: Train core GNN models.
*   `dgf.transform.*`: Transform graph data into other graph data.
*   `dgf.validate.*`: Validate data.
*   `dgf.beam.*`: Apache Beam related functions and classes.


Functions not yet part of the official API are available under `dgf.src.*`. Those are not listed in this page.



### Module `dgf.analyse`

Analyze graph e.g. feature statistics, graph statistics.

*   `dgf.analyse.connected_components`: Computes the connected components of an in-memory homogeneous graph.
*   `dgf.analyse.feature_statistics`: Computes the feature stats from a single graph.
*   `dgf.analyse.feature_statistics_from_graphs`: Computes the feature stats from multiple graphs.
*   `dgf.analyse.padding_from_graph_generator`: Creates a padding configuration from a set of in-memory graphs.
*   `dgf.analyse.print_schema`: Generates a human-readable string representation of a graph schema.




### Module `dgf.convert`

Convert object formats e.g. convert a graph to a Sparse Deferred struct.

*   `dgf.convert.graph_dict_to_graph`: Converts a TF GNN Graph Sample Dict to an InMemoryGraph.
*   `dgf.convert.graph_to_jax_graph`: Converts a (NumPy) in-memory graph into a JAX in-memory graph.
*   `dgf.convert.graph_to_networkx`: Converts an InMemoryGraph into a NetworkX MultiDiGraph.
*   `dgf.convert.graph_to_serialized_tfgnn_graph`: Converts an InMemoryGraph into a serialized TF-GNN graph sample proto.
*   `dgf.convert.graph_to_sparse_deferred_struct`: Converts an in-memory graph into a Sparse Deferred struct.
*   `dgf.convert.graph_to_tf_graph`: Converts a graph to a TF in-memory graph.
*   `dgf.convert.graph_to_tfgnn_graph`: Converts an InMemoryGraph to a TF GNN Graph Sample.
*   `dgf.convert.graph_to_tfgnn_graph_dict`: Converts an InMemoryGraph to a TF GNN Graph Sample Dict.
*   `dgf.convert.graphs_to_serialized_tfgnn_graphs`: Converts a sequence of InMemoryGraphs into serialized TF-GNN graph sample protos.
*   `dgf.convert.networkx_to_graph`: Converts a NetworkX graph into an InMemoryGraph and its schema.
*   `dgf.convert.sampling_config_to_flume_sampler_sampling_spec`: Converts a DGF sampling configuration to a Flume Sampler SamplingSpec.
*   `dgf.convert.schema_to_spanner_ddl`: Converts a GraphSchema to a string of CREATE statements for Spanner.
*   `dgf.convert.schema_to_sparse_deferred_schema`: Converts a DGF `GraphSchema` into a Sparse Deferred schema.
*   `dgf.convert.schema_to_tfgnn_schema`: Converts a GraphSchema object into a TF-GNN schema proto.
*   `dgf.convert.sparse_deferred_struct_to_graph`: Converts a Sparse Deferred struct into an in-memory graph.
*   `dgf.convert.tf_graph_dict_to_tf_graph`: Converts a flattened TFInMemoryGraphDict back into a TFInMemoryGraph.
*   `dgf.convert.tf_graph_to_tf_graph_dict`: Converts a TFInMemoryGraph into a flattened TFInMemoryGraphDict.
*   `dgf.convert.tfgnn_graph_to_graph`: Converts a TF GNN Graph Sample to an InMemoryGraph.
*   `dgf.convert.tfgnn_schema_to_schema`: Converts a TF-GNN schema proto into a GraphSchema object.




### Module `dgf.data`

Classes to represent graph data. Contains no functions / algorithms.

*   `dgf.data.ComputeNodeSpec`: Configuration of a given compute node (e.g., worker, manager).
*   `dgf.data.EdgeSchema`: EdgeSchema(source: str, target: str, features: Dict[str, dgf.src.data.schema.FeatureSchema] = <factory>)
*   `dgf.data.EdgeSetPadding`: EdgeSetPadding(num_edges: int)
*   `dgf.data.FeatureFormat`: How a value is represented / stored.
*   `dgf.data.FeatureSchema`: Schema for a single feature.
*   `dgf.data.FeatureSemantic`: How a value should be interpreted.
*   `dgf.data.FeatureSetStatistics`: Statistics for a set of features.
*   `dgf.data.FeatureStatistics`: Statistics for a feature.
*   `dgf.data.GraphFeatureStatistics`: Statistics about the features in a graph.
*   `dgf.data.GraphSchema`: GraphSchema(node_sets: Dict[str, dgf.src.data.schema.NodeSchema], edge_sets: Dict[str, dgf.src.data.schema.EdgeSchema])
*   `dgf.data.GraphSchemaFilter`: Filters a GraphSchema to sub-select node sets, edge sets, and features.
*   `dgf.data.GraphSchemaV2`: GraphSchema(node_sets: Dict[str, dgf.src.data.schema.NodeSchema], edge_sets: Dict[str, dgf.src.data.schema.EdgeSchema])
*   `dgf.data.InMemoryEdgeSet`: An Edge Set.
*   `dgf.data.InMemoryGraph`: An in-memory generic graph.
*   `dgf.data.InMemoryNodeSet`: A Node Set.
*   `dgf.data.JaxInMemoryEdgeSet`: An Edge Set.
*   `dgf.data.JaxInMemoryGraph`: An in-memory generic graph.
*   `dgf.data.JaxInMemoryNodeSet`: A Node Set.
*   `dgf.data.NodeSchema`: NodeSchema(features: Dict[str, dgf.src.data.schema.FeatureSchema] = <factory>)
*   `dgf.data.NodeSetPadding`: NodeSetPadding(num_nodes: int)
*   `dgf.data.Padding`: Information to pad a graph.
*   `dgf.data.TFInMemoryEdgeSet`: An Edge Set.
*   `dgf.data.TFInMemoryGraph`: An in-memory generic graph.
*   `dgf.data.TFInMemoryNodeSet`: A Node Set.




### Module `dgf.exception`

DGF specific exceptions.

*   `dgf.exception.InsufficientPaddingError`: Inappropriate argument value (of correct type).




### Module `dgf.filesystem`

GraphFlow unified filesystem API.

*   `dgf.filesystem.create_gcs_bucket`: Creates a GCS bucket.
*   `dgf.filesystem.exists`: None
*   `dgf.filesystem.glob`: Returns a list of files and directories matching a pattern.
*   `dgf.filesystem.is_gcs_path`: None
*   `dgf.filesystem.makedirs`: Creates directories if they do not exist.
*   `dgf.filesystem.open_read`: Opens a file for reading and returns a Python file handle.
*   `dgf.filesystem.remove_paths`: Removes all the files in parallel.
*   `dgf.filesystem.rename`: Renames (moves) a file or directory from old_path to new_path.
*   `dgf.filesystem.rmtree`: Recursively removes a directory and its contents.




### Module `dgf.generate`

Tools to generate synthetic data.

*   `dgf.generate.EdgeNeighborGenerator`: Generates the indices of the positive and negative pairs of nodes.
*   `dgf.generate.RandomNegativeSampler`: Replace the target node with a random node.
*   `dgf.generate.RandomWalkNegativeSampler`: Replace the target node with a random walk generated node.
*   `dgf.generate.SyntheticGraphSampleConfig`: Configuration for generating synthetic graph samples.
*   `dgf.generate.generate_synthetic_graph_sample`: Generates a single synthetic graph sample based on a sampling plan.
*   `dgf.generate.write_synthetic_graph_sample_as_tfgnn_graphs`: Generates and writes synthetic graph samples as TF-GNN graphs.




### Module `dgf.io`

Read and write graph, schema, and other related data.

*   `dgf.io.cache`: Returns and caches the variable(s) created by "create_fn".
*   `dgf.io.create_spanner_tables_from_graph_schema`: Creates Spanner tables for a graph schema.
*   `dgf.io.export_bigquery_to_disk`: Reads a BigQuery Graph in-process and returns a GraphFlow in-memory graph.
*   `dgf.io.fetch_graphland_graph`: Downloads and loads a Graphland dataset into memory.
*   `dgf.io.fetch_ogb_graph`: Downloads and loads an OGB node property prediction dataset into memory.
*   `dgf.io.read_bigquery_graph`: Reads a BigQuery Graph in-process and returns a GraphFlow in-memory graph.
*   `dgf.io.read_bigquery_graph_schema`: Reads the schema of a BigQuery graph into a GF schema.
*   `dgf.io.read_feature_statistics`: Reads feature statistics from disk in a JSON format.
*   `dgf.io.read_graph`: Reads a GF graph from a directory to an in-memory graph.
*   `dgf.io.read_graphai_hgraph`: Reads an on-disk HGraph into an in-memory representation.
*   `dgf.io.read_schema`: Loads graph schema from disk in a json format.
*   `dgf.io.read_spanner_graph`: Reads a Spanner Graph in-process and returns a GraphFlow in-memory graph.
*   `dgf.io.read_spanner_graph_schema`: Reads the schema of a Spanner Graph.
*   `dgf.io.read_text_proto`: Read a proto from disk in text format.
*   `dgf.io.read_tfgnn_graphs`: Reads a set of in-memory graphs from disk stored as TF Examples.
*   `dgf.io.write_feature_statistics`: Saves feature statistics to disk in a json format.
*   `dgf.io.write_graph`: Writes an in-memory graph and schema to a GF Graph directory.
*   `dgf.io.write_schema`: Saves graph schema to disk in a json format.
*   `dgf.io.write_text_proto`: Writes a proto to disk in text format.
*   `dgf.io.write_tfgnn_graphs`: Writes a set of in-memory graphs to disk as TF Examples.




### Module `dgf.jax`

Machine Learning and Graph Neural Networks using JAX.

*   `dgf.jax.JaxBaseConfig`: Base class for a GNN implemented in JAX.
*   `dgf.jax.get_activation`: Get an activation function by (string) name.
*   `dgf.jax.jnp_dtype_from_string`: Return a JAX numpy type from a string name.
*   `dgf.jax.jnp_name_from_dtype`: Return a string name for a jnp.dtype object.
*   `dgf.jax.train`: Trains a Flax module with a flexible and feature-rich training loop.


#### Module `dgf.jax.layers`

Flax modules implementing low level GNN operations.

*   `dgf.jax.layers.ClassificationHead`: Simple classification head.
*   `dgf.jax.layers.ClassificationHeadConfig`: Configuration for a classification head.
*   `dgf.jax.layers.ConditionalGIN`: Conditional GIN with a labeling trick: https://arxiv.org/abs/2106.06935.
*   `dgf.jax.layers.EmbedAndHomogenizeGraph`: Convert a heterogeneous graph into a homogeneous one.
*   `dgf.jax.layers.EmbedAndHomogenizeGraphConfig`: Config for EmbedAndHomogenizeGraph.
*   `dgf.jax.layers.EmbedFeatureSet`: Computes a fixed sized dense embedding for a set of feature values.
*   `dgf.jax.layers.EmbedFeatureSetConfig`: Configuration for the EmbedFeatureSet layer.
*   `dgf.jax.layers.EmbedGraph`: Compute a fixed sized dense embedding for all the features in a graph.
*   `dgf.jax.layers.EmbedGraphConfig`: Configuration for "EmbedGraph".
*   `dgf.jax.layers.GCN`: Graph convolutional network: https://arxiv.org/pdf/1609.02907.pdf.
*   `dgf.jax.layers.GCNConfig`: Makeable GCN config class with sensible defaults.
*   `dgf.jax.layers.GIN`: Graph isomorphism network: https://arxiv.org/pdf/1810.00826.pdf.
*   `dgf.jax.layers.GINConfig`: Makeable GIN config class with sensible defaults.
*   `dgf.jax.layers.GenericBlock`: A generic configurable neural network block.
*   `dgf.jax.layers.GenericBlockConfig`: Configuration for a generic block parsed from a string.
*   `dgf.jax.layers.HeterogeneousGraphAttentionNetwork`: A single layer of heterogeneous Graph Attention Network.
*   `dgf.jax.layers.HeterogeneousGraphAttentionNetworkConfig`: Configuration for HeterogeneousGraphAttentionNetwork.
*   `dgf.jax.layers.HeterogeneousGraphConvolution`: A single layer of heterogeneous Graph Neural Network message passing.
*   `dgf.jax.layers.HeterogeneousGraphConvolutionConfig`: Configuration for HeterogeneousGraphConvolution.
*   `dgf.jax.layers.MLP`: A generic MLP followed by a linear layer.
*   `dgf.jax.layers.MPNN`: Message-Passing Neural Network: https://arxiv.org/abs/1704.01212.
*   `dgf.jax.layers.MPNNConfig`: Makeable MPNN config class with sensible defaults.
*   `dgf.jax.layers.Projector`: Simple wrapper around the generic MLP layer for graph input/output.
*   `dgf.jax.layers.ProjectorConfig`: Makeable Projector config class with sensible defaults.
*   `dgf.jax.layers.ResidualMLPV2`: A residual MLP layer. See ResidualMLPV2Config.
*   `dgf.jax.layers.ResidualMLPV2Config`: A residual MLP layer.
*   `dgf.jax.layers.identity`: None
*   `dgf.jax.layers.ingest_feature`: None
*   `dgf.jax.layers.modern_residual_mlp`: None
*   `dgf.jax.layers.sequential_mlp`: None






### Module `dgf.learning`

Top level learning module.

*   `dgf.learning.LinkPredictionModel`: The user-visible returned model object for edge prediction.
*   `dgf.learning.Model`: A generic model from the 10-lines of code API.
*   `dgf.learning.NodePredictionModel`: The user-visible returned model object.
*   `dgf.learning.load_model`: Loads a model previously saved with `model.save()`.
*   `dgf.learning.train_link_model`: Trains a supervised Graph Neural Network model for edge prediction.
*   `dgf.learning.train_node_model`: Trains a supervised Graph Neural Network model for node-level prediction.




### Module `dgf.plot`

Plot graph, schema, and other graph related data.

*   `dgf.plot.plot_graph`: Plots an in-memory graph.
*   `dgf.plot.plot_graph_with_graphscope`: Creates an interactive plot of an in-memory graph using GraphScope.
*   `dgf.plot.plot_nx_graph`: Helper function to draw an nx graph.
*   `dgf.plot.plot_schema`: Plots the graphschema's meta-graph (i.e., its nodesets and edgesets).




### Module `dgf.print`

Printing structures.

*   `dgf.print.padding`: Generates a human-readable string representation of a graph padding.
*   `dgf.print.sampling_plan`: Generates a human-readable tree representation of a sampling plan.
*   `dgf.print.schema`: Generates a human-readable string representation of a graph schema.




### Module `dgf.sampling`

Extracts subsets of graphs for GNN training.

*   `dgf.sampling.Sampler`: None
*   `dgf.sampling.SamplingPlan`: Defines a complex sampling config.
*   `dgf.sampling.SimpleSamplingConfig`: Configuration for simple neighborhood sampling.
*   `dgf.sampling.SpannerGraphSampler`: Sampler that executes queries on Spanner directly to fetch subgraphs.
*   `dgf.sampling.create_distributed_sampler`: Create and return a sampling manager.
*   `dgf.sampling.create_graph_spanner_sampler`: Creates a SpannerGraphSampler instance.
*   `dgf.sampling.create_sampler`: Creates an in-memory sampler.
*   `dgf.sampling.extract_beam_nodes_ids`: Extracts all the node ids of a given nodeset.
*   `dgf.sampling.flume_sampler`: Configures and prints the CLI command to run the Flume Sampler.
*   `dgf.sampling.sample_with_beam_semi_distributed_sampler`: Samples subgraphs from a distributed graph using a semi-distributed algo.
*   `dgf.sampling.sample_with_beam_semi_distributed_sampler_v2`: Samples subgraphs from a distributed graph using a semi-distributed algo.
*   `dgf.sampling.sample_with_distributed_batching`: Creates samples for each node of a graph in batch mode.
*   `dgf.sampling.simple_sampling_config_to_sampling_plan`: Converts a SimpleSamplingConfig to a more general SamplingPlan.
*   `dgf.sampling.start_worker`: Start an online sampling worker.




### Module `dgf.train`

Train core GNN models.

*   `dgf.train.EmbedNodesetFeaturesModule`: A FLAX module to transform a set of features into a fixed-size embedding.




### Module `dgf.transform`

Transform graph data into other graphs data.

*   `dgf.transform.AutoNormalizeConfig`: Configuration for automatic feature normalization for GNNs.
*   `dgf.transform.ContainsLabelPredicate`: Predicate for filtering subgraphs if they have a positive label.
*   `dgf.transform.DictionaryIndexNormalizer`: Normalizes features by mapping dictionary keys to their integer indices.
*   `dgf.transform.GNNDatasetPreparator`: Generates graph samples to train node prediction models.
*   `dgf.transform.GraphNormalizer`: Applies a collection of individual AbstractFeatureNormalizer on a graph.
*   `dgf.transform.GraphNormalizerConfig`: Raw information of a GraphNormalizer for easy serialization.
*   `dgf.transform.IdentityNormalizer`: A normalizer that simply pass a feature without changing it.
*   `dgf.transform.NumNodesPredicate`: Predicate for filtering by number of nodes.
*   `dgf.transform.SoftQuantileNormalizer`: Normalizes a numerical feature by replacing it with its soft quantile -0.5.
*   `dgf.transform.apply_feature`: Applies feature processors to the node and edge sets of a graph.
*   `dgf.transform.auto_normalize`: Create a generally good GraphNormalizer from feature statistics.
*   `dgf.transform.batch_indices_generator`: Generates batches of indices.
*   `dgf.transform.drop_edge_features`: Drops all edge features from a graph and its schema.
*   `dgf.transform.drop_edge_features_from_schema`: Drops all edge features from a schema.
*   `dgf.transform.filter_graph`: Creates an in-memory graph with a subset of nodesets/edgesets/features.
*   `dgf.transform.filter_graphs`: Filters a sequence of graphs based on user defined predicates.
*   `dgf.transform.filter_schema`: Extracts a subset of the nodesets/edgesets/features from a schema.
*   `dgf.transform.homogeneous_graph_piece_to_nx`: Convert InMemoryGraph to an nx.Graph object.
*   `dgf.transform.homogenize`: Homogenizes a heterogeneous graph into a homogeneous one.
*   `dgf.transform.merge_graphs`: Merges multiple `InMemoryGraph` instances into a single graph.
*   `dgf.transform.propagate_timestamp_to_edges`: Propagates timestamps from nodes to edges.
*   `dgf.transform.remove_padding_sentinels`: Removes the sentinel nodes and edges added by `merge_graphs`.
*   `dgf.transform.table2graph`: Converts a table (dict of arrays or DataFrame) into an InMemoryGraph and Schema.




### Module `dgf.validate`

Validate data.

*   `dgf.validate.validate_graph`: Validates an in memory graph object.




### Module `dgf.beam`

Apache Beam related functions and classes.

*   `dgf.beam.program_started`: Call this function at the beginning of all GraphFlow Beam jobs.
*   `dgf.beam.runner_from_name`: Returns a Beam runner based on the provided name.
*   `dgf.beam.runner_from_options`: Returns a Beam runner based on the provided options.


#### Module `dgf.beam.analyse`

Analyse graph e.g. feature statistics, graph statistics.

*   `dgf.beam.analyse.feature_statistics`: Computes the feature statistics for a Graph.
*   `dgf.beam.analyse.feature_statistics_from_graphs`: Computes the feature statistics for a set of InMemoryGraph.




#### Module `dgf.beam.data`

Classes to represent graph data. Contains no functions / algorithms.

*   `dgf.beam.data.Edge`: A single flat edge.
*   `dgf.beam.data.Graph`: A (potentially distributed) heterogeneous graph.
*   `dgf.beam.data.HeterogeniousGraph`: A (potentially distributed) heterogeneous graph.
*   `dgf.beam.data.HomogeneousGraph`: A (potentially distributed) homogeneous graph.
*   `dgf.beam.data.KeyedInMemoryGraph`: KeyedInMemoryGraph(key, graph)
*   `dgf.beam.data.Node`: Node(id: bytes | int, features: Optional[Dict[str, np.ndarray]] = None)




#### Module `dgf.beam.io`

Read and write graph, schema, and other related data.

*   `dgf.beam.io.CreateSpannerTables`: Creates Spanner tables for a graph schema.
*   `dgf.beam.io.read_bigquery_graph`: Read BigQuery Graph via Beam and return a distributed GraphFlow graph.
*   `dgf.beam.io.read_graph`: Reads a GF graph into a distributed graph.
*   `dgf.beam.io.read_graphai_hgraph`: Reads a distributed HGraph using Beam.
*   `dgf.beam.io.read_spanner_graph`: Read Spanner Graph via Beam and return a distributed GraphFlow graph.
*   `dgf.beam.io.read_tfgnn_graphs`: Read a collection of TF GNN Graphs.
*   `dgf.beam.io.write_edge_set_to_spanner`: Writes an edge set to a Spanner table using SpannerInsertOrUpdate.
*   `dgf.beam.io.write_feature_statistics`: Writes a beam pcollection of feature statistics to disk in json format.
*   `dgf.beam.io.write_graph`: Writes a GF Graph from a distributed graph (beam).
*   `dgf.beam.io.write_graphai_hgraph`: Initializes the WriteToHGraph PTransform.
*   `dgf.beam.io.write_node_set_to_spanner`: Writes a node set to a Spanner table using SpannerInsertOrUpdate.
*   `dgf.beam.io.write_spanner`: Writes a heterogeneous graph to Spanner.
*   `dgf.beam.io.write_tfgnn_graphs`: Writes a collection of TF Graph Samples on disk.




#### Module `dgf.beam.sampling`

Extracts subsets of graphs for GNN training.

*   `dgf.beam.sampling.extract_nodes_ids`: Extracts all the node ids of a given nodeset.
*   `dgf.beam.sampling.semi_distributed_sampler_v1`: Samples subgraphs from a distributed graph using a semi-distributed algo.
*   `dgf.beam.sampling.semi_distributed_sampler_v2`: Samples subgraphs from a distributed graph using a semi-distributed algo.




#### Module `dgf.beam.transform`

Transform graph data into other graphs data.

*   `dgf.beam.transform.reverse_edges`: Reverse the direction of edges in a graph.








