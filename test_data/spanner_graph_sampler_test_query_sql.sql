WITH
  _dgf_source_node AS (
    SELECT n.`PaperId` AS seed_id, n.`PaperId` AS n0_id
    FROM `Papers` AS n
    WHERE n.`PaperId` IN UNNEST(@seed_ids)
  ),
  _dgf_hop_1 AS (
    SELECT p.seed_id, p.n0_id AS parent_id, c.child_id AS n1_id, c.edge_json
    FROM (SELECT DISTINCT q.seed_id, q.n0_id FROM _dgf_source_node AS q) AS p,
    UNNEST(ARRAY(
      SELECT AS STRUCT e.`AuthorId` AS child_id, JSON_OBJECT('position', e.`Position`) AS edge_json
      FROM `Writes` AS e
      WHERE e.`PaperId` = p.n0_id AND e.`AuthorId` IS NOT NULL
      ORDER BY FARM_FINGERPRINT(CONCAT(@sampling_seed, b'1:', p.seed_id, p.n0_id, CAST(e.`AuthorId` AS BYTES))) LIMIT 3
    )) AS c
  ),
  _dgf_hop_2 AS (
    SELECT p.seed_id, p.n1_id AS parent_id, c.child_id AS n2_id, c.edge_json
    FROM (SELECT DISTINCT q.seed_id, q.n1_id FROM _dgf_hop_1 AS q) AS p,
    UNNEST(ARRAY(
      SELECT AS STRUCT e.`PaperId` AS child_id, JSON_OBJECT('position', e.`Position`) AS edge_json
      FROM `Writes` AS e
      WHERE e.`AuthorId` = p.n1_id AND e.`PaperId` IS NOT NULL
      ORDER BY FARM_FINGERPRINT(CONCAT(@sampling_seed, b'2:', p.seed_id, CAST(p.n1_id AS BYTES), e.`PaperId`)) LIMIT 3
    )) AS c
  ),
  _dgf_nodes_paper AS (
    SELECT DISTINCT a.seed_id, a.node_id FROM (
      SELECT s.seed_id, s.n0_id AS node_id FROM _dgf_source_node AS s
      UNION ALL
      SELECT h.seed_id, h.n2_id AS node_id FROM _dgf_hop_2 AS h
    ) AS a
  ),
  _dgf_nodes_author AS (
    SELECT DISTINCT a.seed_id, a.node_id FROM (
      SELECT h.seed_id, h.n1_id AS node_id FROM _dgf_hop_1 AS h
    ) AS a
  )
SELECT
  u.seed_id AS seed_id,
  u.node_id AS node_id,
  'node' AS element_type,
  'paper' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  JSON_OBJECT('year', n.`Year`, 'tag', n.`Tag`, 'embedding', n.`Embedding`) AS properties_json
FROM _dgf_nodes_paper AS u
INNER JOIN `Papers` AS n ON u.node_id = n.`PaperId`
UNION ALL
SELECT
  u.seed_id AS seed_id,
  CAST(u.node_id AS BYTES) AS node_id,
  'node' AS element_type,
  'author' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  CAST(NULL AS JSON) AS properties_json
FROM _dgf_nodes_author AS u
INNER JOIN `Authors` AS n ON u.node_id = n.`AuthorId`
UNION ALL
SELECT
  e.seed_id AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'writes' AS element_class,
  CAST(e.source_id AS BYTES) AS source_id,
  e.target_id AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT h.seed_id, h.n1_id AS source_id, h.parent_id AS target_id, h.edge_json FROM _dgf_hop_1 AS h
  UNION ALL
  SELECT h.seed_id, h.parent_id AS source_id, h.n2_id AS target_id, h.edge_json FROM _dgf_hop_2 AS h
) AS e