WITH
  _dgf_source_node AS (
    SELECT n.`cell_id` AS seed_k0, n.`snapshot_ts` AS seed_k1, n.`cell_id` AS n0_k0, n.`snapshot_ts` AS n0_k1
    FROM (SELECT DISTINCT k0, k1 FROM UNNEST(@seed_ids_0) AS k0 WITH OFFSET AS o0 INNER JOIN UNNEST(@seed_ids_1) AS k1 WITH OFFSET AS o1 ON o1 = o0) AS s
    INNER JOIN `Cells` AS n ON n.`cell_id` = s.k0 AND n.`snapshot_ts` = TIMESTAMP_MICROS(s.k1)
  ),
  _dgf_hop_1 AS (
    SELECT p.seed_k0, p.seed_k1, p.n0_k0 AS parent_k0, p.n0_k1 AS parent_k1, c.child_k0 AS n1_k0, c.child_k1 AS n1_k1, c.edge_json
    FROM (SELECT DISTINCT q.seed_k0, q.seed_k1, q.n0_k0, q.n0_k1 FROM _dgf_source_node AS q) AS p,
    UNNEST(ARRAY(
      SELECT AS STRUCT e.`dst_cell` AS child_k0, e.`snapshot_ts` AS child_k1, CAST(NULL AS JSON) AS edge_json
      FROM `Links` AS e
      WHERE e.`src_cell` = p.n0_k0 AND e.`snapshot_ts` = p.n0_k1 AND e.`dst_cell` IS NOT NULL AND e.`snapshot_ts` IS NOT NULL
      ORDER BY FARM_FINGERPRINT(CONCAT(@sampling_seed, b'1:', CAST(CONCAT(REPLACE(REPLACE(p.seed_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(p.seed_k1) AS STRING)) AS BYTES), CAST(CONCAT(REPLACE(REPLACE(p.n0_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(p.n0_k1) AS STRING)) AS BYTES), CAST(CONCAT(REPLACE(REPLACE(e.`dst_cell`, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(e.`snapshot_ts`) AS STRING)) AS BYTES))) LIMIT 2
    )) AS c
  ),
  _dgf_hop_2 AS (
    SELECT p.seed_k0, p.seed_k1, p.n0_k0 AS parent_k0, p.n0_k1 AS parent_k1, c.child_k0 AS n2_k0, c.child_k1 AS n2_k1, c.edge_json
    FROM (SELECT DISTINCT q.seed_k0, q.seed_k1, q.n0_k0, q.n0_k1 FROM _dgf_source_node AS q) AS p,
    UNNEST(ARRAY(
      SELECT AS STRUCT e.`src_cell` AS child_k0, e.`snapshot_ts` AS child_k1, CAST(NULL AS JSON) AS edge_json
      FROM `Links` AS e
      WHERE e.`dst_cell` = p.n0_k0 AND e.`snapshot_ts` = p.n0_k1 AND e.`src_cell` IS NOT NULL AND e.`snapshot_ts` IS NOT NULL
      ORDER BY FARM_FINGERPRINT(CONCAT(@sampling_seed, b'2:', CAST(CONCAT(REPLACE(REPLACE(p.seed_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(p.seed_k1) AS STRING)) AS BYTES), CAST(CONCAT(REPLACE(REPLACE(p.n0_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(p.n0_k1) AS STRING)) AS BYTES), CAST(CONCAT(REPLACE(REPLACE(e.`src_cell`, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(e.`snapshot_ts`) AS STRING)) AS BYTES))) LIMIT 2
    )) AS c
  ),
  _dgf_nodes_Cell AS (
    SELECT DISTINCT a.seed_k0, a.seed_k1, a.node_k0, a.node_k1 FROM (
      SELECT s.seed_k0, s.seed_k1, s.n0_k0 AS node_k0, s.n0_k1 AS node_k1 FROM _dgf_source_node AS s
      UNION ALL
      SELECT h.seed_k0, h.seed_k1, h.n1_k0 AS node_k0, h.n1_k1 AS node_k1 FROM _dgf_hop_1 AS h
      UNION ALL
      SELECT h.seed_k0, h.seed_k1, h.n2_k0 AS node_k0, h.n2_k1 AS node_k1 FROM _dgf_hop_2 AS h
    ) AS a
  )
SELECT
  CAST(CONCAT(REPLACE(REPLACE(u.seed_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(u.seed_k1) AS STRING)) AS BYTES) AS seed_id,
  CAST(CONCAT(REPLACE(REPLACE(u.node_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(u.node_k1) AS STRING)) AS BYTES) AS node_id,
  'node' AS element_type,
  'Cell' AS element_class,
  CAST(NULL AS BYTES) AS source_id,
  CAST(NULL AS BYTES) AS target_id,
  JSON_OBJECT('snapshot_ts', UNIX_MICROS(n.`snapshot_ts`)) AS properties_json
FROM _dgf_nodes_Cell AS u
INNER JOIN `Cells` AS n ON u.node_k0 = n.`cell_id` AND u.node_k1 = n.`snapshot_ts`
UNION ALL
SELECT
  CAST(CONCAT(REPLACE(REPLACE(e.seed_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(e.seed_k1) AS STRING)) AS BYTES) AS seed_id,
  CAST(NULL AS BYTES) AS node_id,
  'edge' AS element_type,
  'LINKS' AS element_class,
  CAST(CONCAT(REPLACE(REPLACE(e.source_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(e.source_k1) AS STRING)) AS BYTES) AS source_id,
  CAST(CONCAT(REPLACE(REPLACE(e.target_k0, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(e.target_k1) AS STRING)) AS BYTES) AS target_id,
  e.edge_json AS properties_json
FROM (
  SELECT h.seed_k0, h.seed_k1, h.parent_k0 AS source_k0, h.parent_k1 AS source_k1, h.n1_k0 AS target_k0, h.n1_k1 AS target_k1, h.edge_json FROM _dgf_hop_1 AS h
  UNION ALL
  SELECT h.seed_k0, h.seed_k1, h.n2_k0 AS source_k0, h.n2_k1 AS source_k1, h.parent_k0 AS target_k0, h.parent_k1 AS target_k1, h.edge_json FROM _dgf_hop_2 AS h
) AS e