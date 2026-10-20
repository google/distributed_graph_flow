-- Creates the BigQuery graphs used by bigquery_graph_sampler_integration_test.
--
-- Run with:
--   bq --project_id=biggraphs-poc query --nouse_legacy_sql \
--     < third_party/py/dgf/src/sampling/gcp/testdata/create_sampler_test_graphs.sql

-- STRING keys, all supported feature types with NULLs, a property alias,
-- parallel edges, a self-loop, an isolated node, and an edge table without
-- properties.
CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_person (
  id STRING,
  age INT64,
  score FLOAT64,
  user_name STRING,
  active BOOL,
  emb ARRAY<FLOAT64>,
  joined TIMESTAMP
)
CLUSTER BY id
AS
SELECT * FROM UNNEST([
  STRUCT<id STRING, age INT64, score FLOAT64, user_name STRING, active BOOL, emb ARRAY<FLOAT64>, joined TIMESTAMP>
  ('p0', 30, 0.5, 'Zoë', TRUE, [1.0, 2.0], TIMESTAMP '2020-01-01 00:00:00+00'),
  ('p1', NULL, NULL, NULL, NULL, [], NULL),
  ('p2', 25, -1.5, '日本', FALSE, [3.0], TIMESTAMP '2021-06-15 12:30:00+00'),
  ('p3', 40, 2.0, 'isolated', TRUE, [4.0, 5.0, 6.0], TIMESTAMP '2019-03-01 00:00:00+00')
]);

CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_item (id STRING)
CLUSTER BY id
AS
SELECT * FROM UNNEST(['i0', 'i1', 'i2']) AS id;

CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_buys (
  source STRING,
  target STRING,
  weight FLOAT64
)
CLUSTER BY source, target
AS
SELECT * FROM UNNEST([
  STRUCT<source STRING, target STRING, weight FLOAT64>
  ('p0', 'i0', 1.0),
  ('p0', 'i0', 2.0),
  ('p0', 'i1', NULL),
  ('p1', 'i1', 3.0),
  ('p2', 'i2', 4.0)
]);

CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_follows (
  source STRING,
  target STRING
)
CLUSTER BY source, target
AS
SELECT * FROM UNNEST([
  STRUCT<source STRING, target STRING>
  ('p0', 'p1'),
  ('p1', 'p2'),
  ('p2', 'p0'),
  ('p2', 'p2')
]);

CREATE OR REPLACE PROPERTY GRAPH dgf_io_tests.dgf_sampler_string_graph
  NODE TABLES (
    dgf_io_tests.dgf_sampler_person AS person
      KEY (id)
      LABEL person PROPERTIES (id, age, score, user_name AS name, active, emb, joined),
    dgf_io_tests.dgf_sampler_item AS item
      KEY (id)
      LABEL item PROPERTIES (id)
  )
  EDGE TABLES (
    dgf_io_tests.dgf_sampler_buys AS buys
      KEY (source, target)
      SOURCE KEY (source) REFERENCES person (id)
      DESTINATION KEY (target) REFERENCES item (id)
      LABEL buys PROPERTIES (weight),
    dgf_io_tests.dgf_sampler_follows AS follows
      KEY (source, target)
      SOURCE KEY (source) REFERENCES person (id)
      DESTINATION KEY (target) REFERENCES person (id)
      LABEL follows NO PROPERTIES
  );

-- BYTES keys and BYTES features, including non-UTF-8 values. BigQuery cannot
-- cluster on BYTES columns. Trailing NUL bytes are avoided: numpy `np.bytes_`
-- (used by DGF) strips them.
CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_bytes_node (id BYTES, blob BYTES)
AS
SELECT * FROM UNNEST([
  STRUCT<id BYTES, blob BYTES>
  (b'\x00\xff', b'\x01\x02'),
  (b'ascii', NULL),
  (b'\xc3\x28', b'\xfe')
]);

CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_bytes_edge (source BYTES, target BYTES)
AS
SELECT * FROM UNNEST([
  STRUCT<source BYTES, target BYTES>
  (b'\x00\xff', b'ascii'),
  (b'ascii', b'\xc3\x28'),
  (b'\xc3\x28', b'\x00\xff')
]);

CREATE OR REPLACE PROPERTY GRAPH dgf_io_tests.dgf_sampler_bytes_graph
  NODE TABLES (
    dgf_io_tests.dgf_sampler_bytes_node AS node
      KEY (id)
      LABEL node PROPERTIES (id, blob)
  )
  EDGE TABLES (
    dgf_io_tests.dgf_sampler_bytes_edge AS edge
      KEY (source, target)
      SOURCE KEY (source) REFERENCES node (id)
      DESTINATION KEY (target) REFERENCES node (id)
      LABEL edge NO PROPERTIES
  );

-- Composite keys (unsupported by the sampler).
CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_composite_node (a STRING, b STRING)
AS
SELECT * FROM UNNEST([STRUCT<a STRING, b STRING>('x', '1'), ('y', '2')]);

CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_composite_edge (
  source_a STRING,
  source_b STRING,
  target_a STRING,
  target_b STRING
)
AS
SELECT * FROM UNNEST([
  STRUCT<source_a STRING, source_b STRING, target_a STRING, target_b STRING>
  ('x', '1', 'y', '2')
]);

CREATE OR REPLACE PROPERTY GRAPH dgf_io_tests.dgf_sampler_composite_graph
  NODE TABLES (
    dgf_io_tests.dgf_sampler_composite_node AS node
      KEY (a, b)
      LABEL node PROPERTIES (a, b)
  )
  EDGE TABLES (
    dgf_io_tests.dgf_sampler_composite_edge AS edge
      KEY (source_a, source_b, target_a, target_b)
      SOURCE KEY (source_a, source_b) REFERENCES node (a, b)
      DESTINATION KEY (target_a, target_b) REFERENCES node (a, b)
      LABEL edge NO PROPERTIES
  );

-- Integer keys (unsupported by the sampler).
CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_int_node (id INT64)
AS
SELECT * FROM UNNEST([1, 2]) AS id;

CREATE OR REPLACE TABLE dgf_io_tests.dgf_sampler_int_edge (source INT64, target INT64)
AS
SELECT * FROM UNNEST([STRUCT<source INT64, target INT64>(1, 2)]);

CREATE OR REPLACE PROPERTY GRAPH dgf_io_tests.dgf_sampler_int_graph
  NODE TABLES (
    dgf_io_tests.dgf_sampler_int_node AS node
      KEY (id)
      LABEL node PROPERTIES (id)
  )
  EDGE TABLES (
    dgf_io_tests.dgf_sampler_int_edge AS edge
      KEY (source, target)
      SOURCE KEY (source) REFERENCES node (id)
      DESTINATION KEY (target) REFERENCES node (id)
      LABEL edge NO PROPERTIES
  );
