WITH
  source_node AS (
    SELECT n0.`id` AS seed_id, n0.`id` AS n0_id
    FROM `biggraphs-poc.dgf_io_tests.dgf_sampler_bytes_node` AS n0
    WHERE n0.`id` IN UNNEST(@seed_ids)
  ),
  hop_1 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`target` AS n1_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.dgf_io_tests.dgf_sampler_bytes_edge` AS e ON p.n0_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n0_id, e.`target`)))) <= 4
  ),
  hop_2 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`source` AS n2_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.dgf_io_tests.dgf_sampler_bytes_edge` AS e ON p.n0_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n0_id, e.`source`)))) <= 4
  ),
  nodes_node AS (
    SELECT DISTINCT seed_id, node_id FROM (
      SELECT seed_id, n0_id AS node_id FROM source_node
      UNION ALL
      SELECT seed_id, n1_id AS node_id FROM hop_1
      UNION ALL
      SELECT seed_id, n2_id AS node_id FROM hop_2
    )
  )
SELECT
  u.seed_id AS seed_id,
  u.node_id AS node_id,
  'node' AS element_type,
  'node' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  TO_JSON_STRING(STRUCT(n.`id` AS `id`, n.`blob` AS `blob`)) AS properties_json
FROM nodes_node AS u
INNER JOIN `biggraphs-poc.dgf_io_tests.dgf_sampler_bytes_node` AS n ON u.node_id = n.`id`
UNION ALL
SELECT
  e.seed_id AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'edge' AS element_class,
  e.source_id AS source_id,
  e.target_id AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT seed_id, source_id, target_id, MIN(edge_json) AS edge_json FROM (
    SELECT seed_id, parent_id AS source_id, n1_id AS target_id, edge_json FROM hop_1
    UNION ALL
    SELECT seed_id, n2_id AS source_id, parent_id AS target_id, edge_json FROM hop_2
  )
  GROUP BY seed_id, source_id, target_id
) AS e