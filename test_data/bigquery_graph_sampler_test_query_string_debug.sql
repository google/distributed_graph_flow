WITH
  source_node AS (
    SELECT n0.`id` AS seed_id, n0.`id` AS n0_id
    FROM `biggraphs-poc.dgf_io_tests.dgf_sampler_person` AS n0
    WHERE n0.`id` IN UNNEST(@seed_ids)
  ),
  hop_1 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`target` AS n1_id, TO_JSON_STRING(STRUCT(e.`weight` AS `weight`)) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.dgf_io_tests.dgf_sampler_buys` AS e ON p.n0_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY e.`target`) <= 4
  ),
  hop_2 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`target` AS n2_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.dgf_io_tests.dgf_sampler_follows` AS e ON p.n0_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY e.`target`) <= 4
  ),
  nodes_person AS (
    SELECT DISTINCT seed_id, node_id FROM (
      SELECT seed_id, n0_id AS node_id FROM source_node
      UNION ALL
      SELECT seed_id, n2_id AS node_id FROM hop_2
    )
  ),
  nodes_item AS (
    SELECT DISTINCT seed_id, node_id FROM (
      SELECT seed_id, n1_id AS node_id FROM hop_1
    )
  )
SELECT
  CAST(u.seed_id AS BYTES) AS seed_id,
  CAST(u.node_id AS BYTES) AS node_id,
  'node' AS element_type,
  'person' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  TO_JSON_STRING(STRUCT(n.`id` AS `id`, n.`age` AS `age`, n.`score` AS `score`, n.`user_name` AS `name`, n.`active` AS `active`, n.`emb` AS `emb`, n.`joined` AS `joined`)) AS properties_json
FROM nodes_person AS u
INNER JOIN `biggraphs-poc.dgf_io_tests.dgf_sampler_person` AS n ON u.node_id = n.`id`
UNION ALL
SELECT
  CAST(u.seed_id AS BYTES) AS seed_id,
  CAST(u.node_id AS BYTES) AS node_id,
  'node' AS element_type,
  'item' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  TO_JSON_STRING(STRUCT(n.`id` AS `id`)) AS properties_json
FROM nodes_item AS u
INNER JOIN `biggraphs-poc.dgf_io_tests.dgf_sampler_item` AS n ON u.node_id = n.`id`
UNION ALL
SELECT
  CAST(e.seed_id AS BYTES) AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'buys' AS element_class,
  CAST(e.source_id AS BYTES) AS source_id,
  CAST(e.target_id AS BYTES) AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT seed_id, source_id, target_id, MIN(edge_json) AS edge_json FROM (
    SELECT seed_id, parent_id AS source_id, n1_id AS target_id, edge_json FROM hop_1
  )
  GROUP BY seed_id, source_id, target_id
) AS e
UNION ALL
SELECT
  CAST(e.seed_id AS BYTES) AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'follows' AS element_class,
  CAST(e.source_id AS BYTES) AS source_id,
  CAST(e.target_id AS BYTES) AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT seed_id, source_id, target_id, MIN(edge_json) AS edge_json FROM (
    SELECT seed_id, parent_id AS source_id, n2_id AS target_id, edge_json FROM hop_2
  )
  GROUP BY seed_id, source_id, target_id
) AS e