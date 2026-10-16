WITH
  source_node AS (
    SELECT n0.id AS seed_id, n0.id AS n0_id
    FROM `biggraphs-poc.arxiv.nodes` AS n0
    WHERE n0.id IN UNNEST(@seed_ids)
  ),
  hop_1 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`#target` AS n1_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    JOIN `biggraphs-poc.arxiv.edges` AS e ON p.n0_id = e.`#source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY RAND()) <= 5
  ),
  nodes_nodes AS (
    SELECT seed_id, n0_id AS node_id FROM source_node
      UNION DISTINCT
      SELECT seed_id, n1_id AS node_id FROM hop_1
  )
SELECT seed_id, node_id, element_type, element_class, source_id, target_id, properties_json
FROM (
SELECT
  SAFE_CONVERT_BYTES_TO_STRING(u.seed_id) AS seed_id,
  SAFE_CONVERT_BYTES_TO_STRING(u.node_id) AS node_id,
  'node' AS element_type,
  'nodes' AS element_class,
  CAST(NULL AS STRING) AS source_id,
  CAST(NULL AS STRING) AS target_id,
  TO_JSON_STRING(STRUCT(STRUCT(SAFE_CONVERT_BYTES_TO_STRING(n.id) AS `id`) AS properties)) AS properties_json
FROM nodes_nodes AS u
JOIN `biggraphs-poc.arxiv.nodes` AS n ON u.node_id = n.id
UNION ALL
SELECT
  SAFE_CONVERT_BYTES_TO_STRING(e.seed_id) AS seed_id,
  CAST(NULL AS STRING) AS node_id,
  'edge' AS element_type,
  'edges' AS element_class,
  SAFE_CONVERT_BYTES_TO_STRING(e.source_id) AS source_id,
  SAFE_CONVERT_BYTES_TO_STRING(e.target_id) AS target_id,
  e.edge_json AS properties_json
FROM (
      SELECT seed_id, parent_id AS source_id, n1_id AS target_id, edge_json FROM hop_1
) AS e
)
