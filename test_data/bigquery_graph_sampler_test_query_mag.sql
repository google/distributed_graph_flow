WITH
  source_node AS (
    SELECT n0.`id` AS seed_id, n0.`id` AS n0_id
    FROM `biggraphs-poc.mag.paper` AS n0
    WHERE n0.`id` IN UNNEST(@seed_ids)
  ),
  hop_1 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`target` AS n1_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.mag.cites` AS e ON p.n0_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n0_id, e.`target`)))) <= 4
  ),
  hop_2 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, e.`target` AS n2_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n1_id FROM hop_1) AS p
    INNER JOIN `biggraphs-poc.mag.cites` AS e ON p.n1_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n1_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n1_id, e.`target`)))) <= 4
  ),
  hop_3 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, e.`source` AS n3_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n1_id FROM hop_1) AS p
    INNER JOIN `biggraphs-poc.mag.cites` AS e ON p.n1_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n1_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n1_id, e.`source`)))) <= 4
  ),
  hop_4 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, e.`target` AS n4_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n1_id FROM hop_1) AS p
    INNER JOIN `biggraphs-poc.mag.has_topic` AS e ON p.n1_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n1_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n1_id, e.`target`)))) <= 4
  ),
  hop_5 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, e.`source` AS n5_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n1_id FROM hop_1) AS p
    INNER JOIN `biggraphs-poc.mag.writes` AS e ON p.n1_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n1_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n1_id, e.`source`)))) <= 4
  ),
  hop_6 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`source` AS n6_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.mag.cites` AS e ON p.n0_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n0_id, e.`source`)))) <= 4
  ),
  hop_7 AS (
    SELECT p.seed_id, p.n6_id AS parent_id, e.`target` AS n7_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n6_id FROM hop_6) AS p
    INNER JOIN `biggraphs-poc.mag.cites` AS e ON p.n6_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n6_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n6_id, e.`target`)))) <= 4
  ),
  hop_8 AS (
    SELECT p.seed_id, p.n6_id AS parent_id, e.`source` AS n8_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n6_id FROM hop_6) AS p
    INNER JOIN `biggraphs-poc.mag.cites` AS e ON p.n6_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n6_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n6_id, e.`source`)))) <= 4
  ),
  hop_9 AS (
    SELECT p.seed_id, p.n6_id AS parent_id, e.`target` AS n9_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n6_id FROM hop_6) AS p
    INNER JOIN `biggraphs-poc.mag.has_topic` AS e ON p.n6_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n6_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n6_id, e.`target`)))) <= 4
  ),
  hop_10 AS (
    SELECT p.seed_id, p.n6_id AS parent_id, e.`source` AS n10_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n6_id FROM hop_6) AS p
    INNER JOIN `biggraphs-poc.mag.writes` AS e ON p.n6_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n6_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n6_id, e.`source`)))) <= 4
  ),
  hop_11 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`target` AS n11_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.mag.has_topic` AS e ON p.n0_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n0_id, e.`target`)))) <= 4
  ),
  hop_12 AS (
    SELECT p.seed_id, p.n11_id AS parent_id, e.`source` AS n12_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n11_id FROM hop_11) AS p
    INNER JOIN `biggraphs-poc.mag.has_topic` AS e ON p.n11_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n11_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n11_id, e.`source`)))) <= 4
  ),
  hop_13 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, e.`source` AS n13_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n0_id FROM source_node) AS p
    INNER JOIN `biggraphs-poc.mag.writes` AS e ON p.n0_id = e.`target`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n0_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n0_id, e.`source`)))) <= 4
  ),
  hop_14 AS (
    SELECT p.seed_id, p.n13_id AS parent_id, e.`target` AS n14_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n13_id FROM hop_13) AS p
    INNER JOIN `biggraphs-poc.mag.affiliated_with` AS e ON p.n13_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n13_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n13_id, e.`target`)))) <= 4
  ),
  hop_15 AS (
    SELECT p.seed_id, p.n13_id AS parent_id, e.`target` AS n15_id, CAST(NULL AS STRING) AS edge_json
    FROM (SELECT DISTINCT seed_id, n13_id FROM hop_13) AS p
    INNER JOIN `biggraphs-poc.mag.writes` AS e ON p.n13_id = e.`source`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.seed_id, p.n13_id ORDER BY FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(@sampling_seed, p.seed_id, p.n13_id, e.`target`)))) <= 4
  ),
  nodes_paper AS (
    SELECT DISTINCT seed_id, node_id FROM (
      SELECT seed_id, n0_id AS node_id FROM source_node
      UNION ALL
      SELECT seed_id, n1_id AS node_id FROM hop_1
      UNION ALL
      SELECT seed_id, n2_id AS node_id FROM hop_2
      UNION ALL
      SELECT seed_id, n3_id AS node_id FROM hop_3
      UNION ALL
      SELECT seed_id, n6_id AS node_id FROM hop_6
      UNION ALL
      SELECT seed_id, n7_id AS node_id FROM hop_7
      UNION ALL
      SELECT seed_id, n8_id AS node_id FROM hop_8
      UNION ALL
      SELECT seed_id, n12_id AS node_id FROM hop_12
      UNION ALL
      SELECT seed_id, n15_id AS node_id FROM hop_15
    )
  ),
  nodes_field_of_study AS (
    SELECT DISTINCT seed_id, node_id FROM (
      SELECT seed_id, n4_id AS node_id FROM hop_4
      UNION ALL
      SELECT seed_id, n9_id AS node_id FROM hop_9
      UNION ALL
      SELECT seed_id, n11_id AS node_id FROM hop_11
    )
  ),
  nodes_author AS (
    SELECT DISTINCT seed_id, node_id FROM (
      SELECT seed_id, n5_id AS node_id FROM hop_5
      UNION ALL
      SELECT seed_id, n10_id AS node_id FROM hop_10
      UNION ALL
      SELECT seed_id, n13_id AS node_id FROM hop_13
    )
  ),
  nodes_institution AS (
    SELECT DISTINCT seed_id, node_id FROM (
      SELECT seed_id, n14_id AS node_id FROM hop_14
    )
  )
SELECT
  CAST(u.seed_id AS BYTES) AS seed_id,
  CAST(u.node_id AS BYTES) AS node_id,
  'node' AS element_type,
  'paper' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  TO_JSON_STRING(STRUCT(n.`id` AS `id`, n.`split` AS `split`, n.`labels` AS `labels`, n.`year` AS `year`, n.`feat` AS `feat`)) AS properties_json
FROM nodes_paper AS u
INNER JOIN `biggraphs-poc.mag.paper` AS n ON u.node_id = n.`id`
UNION ALL
SELECT
  CAST(u.seed_id AS BYTES) AS seed_id,
  CAST(u.node_id AS BYTES) AS node_id,
  'node' AS element_type,
  'field_of_study' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  TO_JSON_STRING(STRUCT(n.`id` AS `id`)) AS properties_json
FROM nodes_field_of_study AS u
INNER JOIN `biggraphs-poc.mag.field_of_study` AS n ON u.node_id = n.`id`
UNION ALL
SELECT
  CAST(u.seed_id AS BYTES) AS seed_id,
  CAST(u.node_id AS BYTES) AS node_id,
  'node' AS element_type,
  'author' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  TO_JSON_STRING(STRUCT(n.`id` AS `id`)) AS properties_json
FROM nodes_author AS u
INNER JOIN `biggraphs-poc.mag.author` AS n ON u.node_id = n.`id`
UNION ALL
SELECT
  CAST(u.seed_id AS BYTES) AS seed_id,
  CAST(u.node_id AS BYTES) AS node_id,
  'node' AS element_type,
  'institution' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  TO_JSON_STRING(STRUCT(n.`id` AS `id`)) AS properties_json
FROM nodes_institution AS u
INNER JOIN `biggraphs-poc.mag.institution` AS n ON u.node_id = n.`id`
UNION ALL
SELECT
  CAST(e.seed_id AS BYTES) AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'cites' AS element_class,
  CAST(e.source_id AS BYTES) AS source_id,
  CAST(e.target_id AS BYTES) AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT seed_id, source_id, target_id, MIN(edge_json) AS edge_json FROM (
    SELECT seed_id, parent_id AS source_id, n1_id AS target_id, edge_json FROM hop_1
    UNION ALL
    SELECT seed_id, parent_id AS source_id, n2_id AS target_id, edge_json FROM hop_2
    UNION ALL
    SELECT seed_id, n3_id AS source_id, parent_id AS target_id, edge_json FROM hop_3
    UNION ALL
    SELECT seed_id, n6_id AS source_id, parent_id AS target_id, edge_json FROM hop_6
    UNION ALL
    SELECT seed_id, parent_id AS source_id, n7_id AS target_id, edge_json FROM hop_7
    UNION ALL
    SELECT seed_id, n8_id AS source_id, parent_id AS target_id, edge_json FROM hop_8
  )
  GROUP BY seed_id, source_id, target_id
) AS e
UNION ALL
SELECT
  CAST(e.seed_id AS BYTES) AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'has_topic' AS element_class,
  CAST(e.source_id AS BYTES) AS source_id,
  CAST(e.target_id AS BYTES) AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT seed_id, source_id, target_id, MIN(edge_json) AS edge_json FROM (
    SELECT seed_id, parent_id AS source_id, n4_id AS target_id, edge_json FROM hop_4
    UNION ALL
    SELECT seed_id, parent_id AS source_id, n9_id AS target_id, edge_json FROM hop_9
    UNION ALL
    SELECT seed_id, parent_id AS source_id, n11_id AS target_id, edge_json FROM hop_11
    UNION ALL
    SELECT seed_id, n12_id AS source_id, parent_id AS target_id, edge_json FROM hop_12
  )
  GROUP BY seed_id, source_id, target_id
) AS e
UNION ALL
SELECT
  CAST(e.seed_id AS BYTES) AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'writes' AS element_class,
  CAST(e.source_id AS BYTES) AS source_id,
  CAST(e.target_id AS BYTES) AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT seed_id, source_id, target_id, MIN(edge_json) AS edge_json FROM (
    SELECT seed_id, n5_id AS source_id, parent_id AS target_id, edge_json FROM hop_5
    UNION ALL
    SELECT seed_id, n10_id AS source_id, parent_id AS target_id, edge_json FROM hop_10
    UNION ALL
    SELECT seed_id, n13_id AS source_id, parent_id AS target_id, edge_json FROM hop_13
    UNION ALL
    SELECT seed_id, parent_id AS source_id, n15_id AS target_id, edge_json FROM hop_15
  )
  GROUP BY seed_id, source_id, target_id
) AS e
UNION ALL
SELECT
  CAST(e.seed_id AS BYTES) AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'affiliated_with' AS element_class,
  CAST(e.source_id AS BYTES) AS source_id,
  CAST(e.target_id AS BYTES) AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT seed_id, source_id, target_id, MIN(edge_json) AS edge_json FROM (
    SELECT seed_id, parent_id AS source_id, n14_id AS target_id, edge_json FROM hop_14
  )
  GROUP BY seed_id, source_id, target_id
) AS e