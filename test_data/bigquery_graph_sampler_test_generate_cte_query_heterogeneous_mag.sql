WITH
  source_node AS (
    SELECT n0.id AS seed_id, n0.id AS n0_id
    FROM `biggraphs-poc.mag.paper` AS n0
    WHERE n0.id IN UNNEST(@seed_ids)
  ),
  hop_1 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`#target` AS n1_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    JOIN `biggraphs-poc.mag.cites` AS e ON p.n0_id = e.`#source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY RAND()) <= 4
  ),
  hop_2 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, e.`#target` AS n2_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n1_id FROM hop_1) AS p
    JOIN `biggraphs-poc.mag.cites` AS e ON p.n1_id = e.`#source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n1_id ORDER BY RAND()) <= 4
  ),
  hop_3 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, e.`#source` AS n3_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n1_id FROM hop_1) AS p
    JOIN `biggraphs-poc.mag.cites` AS e ON p.n1_id = e.`#target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n1_id ORDER BY RAND()) <= 4
  ),
  hop_4 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, e.`#source` AS n4_id, TO_JSON_STRING(STRUCT(STRUCT(e.weight AS `weight`) AS properties)) AS edge_json
    FROM (SELECT DISTINCT seed_id, n1_id FROM hop_1) AS p
    JOIN `biggraphs-poc.mag.writes` AS e ON p.n1_id = e.`#target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n1_id ORDER BY RAND()) <= 4
  ),
  hop_5 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`#source` AS n5_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    JOIN `biggraphs-poc.mag.cites` AS e ON p.n0_id = e.`#target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY RAND()) <= 4
  ),
  hop_6 AS (
    SELECT p.seed_id, p.n5_id AS parent_id, e.`#target` AS n6_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n5_id FROM hop_5) AS p
    JOIN `biggraphs-poc.mag.cites` AS e ON p.n5_id = e.`#source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n5_id ORDER BY RAND()) <= 4
  ),
  hop_7 AS (
    SELECT p.seed_id, p.n5_id AS parent_id, e.`#source` AS n7_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n5_id FROM hop_5) AS p
    JOIN `biggraphs-poc.mag.cites` AS e ON p.n5_id = e.`#target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n5_id ORDER BY RAND()) <= 4
  ),
  hop_8 AS (
    SELECT p.seed_id, p.n5_id AS parent_id, e.`#source` AS n8_id, TO_JSON_STRING(STRUCT(STRUCT(e.weight AS `weight`) AS properties)) AS edge_json
    FROM (SELECT DISTINCT seed_id, n5_id FROM hop_5) AS p
    JOIN `biggraphs-poc.mag.writes` AS e ON p.n5_id = e.`#target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n5_id ORDER BY RAND()) <= 4
  ),
  hop_9 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`#source` AS n9_id, TO_JSON_STRING(STRUCT(STRUCT(e.weight AS `weight`) AS properties)) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    JOIN `biggraphs-poc.mag.writes` AS e ON p.n0_id = e.`#target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY RAND()) <= 4
  ),
  hop_10 AS (
    SELECT p.seed_id, p.n9_id AS parent_id, e.`#target` AS n10_id, TO_JSON_STRING(STRUCT(STRUCT(e.weight AS `weight`) AS properties)) AS edge_json
    FROM (SELECT DISTINCT seed_id, n9_id FROM hop_9) AS p
    JOIN `biggraphs-poc.mag.writes` AS e ON p.n9_id = e.`#source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n9_id ORDER BY RAND()) <= 4
  ),
  nodes_paper AS (
    SELECT seed_id, n0_id AS node_id FROM source_node
      UNION DISTINCT
      SELECT seed_id, n1_id AS node_id FROM hop_1
      UNION DISTINCT
      SELECT seed_id, n2_id AS node_id FROM hop_2
      UNION DISTINCT
      SELECT seed_id, n3_id AS node_id FROM hop_3
      UNION DISTINCT
      SELECT seed_id, n5_id AS node_id FROM hop_5
      UNION DISTINCT
      SELECT seed_id, n6_id AS node_id FROM hop_6
      UNION DISTINCT
      SELECT seed_id, n7_id AS node_id FROM hop_7
      UNION DISTINCT
      SELECT seed_id, n10_id AS node_id FROM hop_10
  ),
  nodes_author AS (
    SELECT seed_id, n4_id AS node_id FROM hop_4
      UNION DISTINCT
      SELECT seed_id, n8_id AS node_id FROM hop_8
      UNION DISTINCT
      SELECT seed_id, n9_id AS node_id FROM hop_9
  )
SELECT seed_id, node_id, element_type, element_class, source_id, target_id, properties_json
FROM (
SELECT
  SAFE_CONVERT_BYTES_TO_STRING(u.seed_id) AS seed_id,
  SAFE_CONVERT_BYTES_TO_STRING(u.node_id) AS node_id,
  'node' AS element_type,
  'paper' AS element_class,
  CAST(NULL AS STRING) AS source_id,
  CAST(NULL AS STRING) AS target_id,
  TO_JSON_STRING(STRUCT(STRUCT(SAFE_CONVERT_BYTES_TO_STRING(n.id) AS `id`, n.labels AS `labels`, n.year AS `year`) AS properties)) AS properties_json
FROM nodes_paper AS u
JOIN `biggraphs-poc.mag.paper` AS n ON u.node_id = n.id
UNION ALL
SELECT
  SAFE_CONVERT_BYTES_TO_STRING(u.seed_id) AS seed_id,
  SAFE_CONVERT_BYTES_TO_STRING(u.node_id) AS node_id,
  'node' AS element_type,
  'author' AS element_class,
  CAST(NULL AS STRING) AS source_id,
  CAST(NULL AS STRING) AS target_id,
  TO_JSON_STRING(STRUCT(STRUCT(SAFE_CONVERT_BYTES_TO_STRING(n.id) AS `id`) AS properties)) AS properties_json
FROM nodes_author AS u
JOIN `biggraphs-poc.mag.author` AS n ON u.node_id = n.id
UNION ALL
SELECT
  SAFE_CONVERT_BYTES_TO_STRING(e.seed_id) AS seed_id,
  CAST(NULL AS STRING) AS node_id,
  'edge' AS element_type,
  'cites' AS element_class,
  SAFE_CONVERT_BYTES_TO_STRING(e.source_id) AS source_id,
  SAFE_CONVERT_BYTES_TO_STRING(e.target_id) AS target_id,
  e.edge_json AS properties_json
FROM (
      SELECT seed_id, parent_id AS source_id, n1_id AS target_id, edge_json FROM hop_1
      UNION DISTINCT
      SELECT seed_id, parent_id AS source_id, n2_id AS target_id, edge_json FROM hop_2
      UNION DISTINCT
      SELECT seed_id, n3_id AS source_id, parent_id AS target_id, edge_json FROM hop_3
      UNION DISTINCT
      SELECT seed_id, n5_id AS source_id, parent_id AS target_id, edge_json FROM hop_5
      UNION DISTINCT
      SELECT seed_id, parent_id AS source_id, n6_id AS target_id, edge_json FROM hop_6
      UNION DISTINCT
      SELECT seed_id, n7_id AS source_id, parent_id AS target_id, edge_json FROM hop_7
) AS e
UNION ALL
SELECT
  SAFE_CONVERT_BYTES_TO_STRING(e.seed_id) AS seed_id,
  CAST(NULL AS STRING) AS node_id,
  'edge' AS element_type,
  'writes' AS element_class,
  SAFE_CONVERT_BYTES_TO_STRING(e.source_id) AS source_id,
  SAFE_CONVERT_BYTES_TO_STRING(e.target_id) AS target_id,
  e.edge_json AS properties_json
FROM (
      SELECT seed_id, n4_id AS source_id, parent_id AS target_id, edge_json FROM hop_4
      UNION ALL
      SELECT seed_id, n8_id AS source_id, parent_id AS target_id, edge_json FROM hop_8
      UNION ALL
      SELECT seed_id, n9_id AS source_id, parent_id AS target_id, edge_json FROM hop_9
      UNION ALL
      SELECT seed_id, parent_id AS source_id, n10_id AS target_id, edge_json FROM hop_10
) AS e
)
