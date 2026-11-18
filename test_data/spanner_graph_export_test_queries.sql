EXPORT DATA OPTIONS (
  format = 'recordio',
  path = '/cns/xx-d/home/my-team/my_graph/nodesets/Paper@*.recordio',
  proto = 'global_proto_db.tensorflow.Example'
) AS
SELECT AS VALUE
  $EXAMPLE(([
    $BYTES_FEATURE('id', ([t.`PaperId`])),
    $INT64_FEATURE('year', ([IFNULL(t.`year`, 0)])),
    $FLOAT_FEATURE('score', ([IFNULL(t.`score`, 0.0)])),
    $INT64_FEATURE('active', ([IFNULL(CAST(t.`active` AS INT64), 0)])),
    $INT64_FEATURE('created', ([IFNULL(UNIX_MICROS(t.`created`), 0)])),
    $BYTES_FEATURE('title', ([IFNULL(CAST(t.`title` AS BYTES), b'')])),
    $FLOAT_FEATURE('emb', (IF(t.`emb` IS NULL, ARRAY(SELECT 0.0 FROM UNNEST(GENERATE_ARRAY(1, 2))), ARRAY(SELECT IFNULL(CAST(e AS FLOAT64), 0.0) FROM UNNEST(t.`emb`) AS e WITH OFFSET AS o ORDER BY o)))),
    $BYTES_FEATURE('tags', (IF(t.`tags` IS NULL, ARRAY(SELECT b'' FROM UNNEST(GENERATE_ARRAY(1, 0))), ARRAY(SELECT IFNULL(CAST(e AS BYTES), b'') FROM UNNEST(t.`tags`) AS e WITH OFFSET AS o ORDER BY o))))
  ]))
FROM /span/nonprod/my-team:my_database.`Papers` AS t
WHERE t.`snapshot` = TIMESTAMP_MICROS(@filter_snapshot);

EXPORT DATA OPTIONS (
  format = 'recordio',
  path = '/cns/xx-d/home/my-team/my_graph/nodesets/Cell@*.recordio',
  proto = 'global_proto_db.tensorflow.Example'
) AS
SELECT AS VALUE
  $EXAMPLE(([
    $BYTES_FEATURE('#id', ([CAST(CONCAT(REPLACE(REPLACE(t.`cell_id`, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(t.`snapshot`) AS STRING)) AS BYTES)])),
    $BYTES_FEATURE('cell_id', ([IFNULL(CAST(t.`cell_id` AS BYTES), b'')])),
    $INT64_FEATURE('snapshot', ([IFNULL(UNIX_MICROS(t.`snapshot`), 0)]))
  ]))
FROM /span/nonprod/my-team:my_database.`Cells` AS t
WHERE t.`snapshot` = TIMESTAMP_MICROS(@filter_snapshot);

EXPORT DATA OPTIONS (
  format = 'recordio',
  path = '/cns/xx-d/home/my-team/my_graph/edgesets/PaperInCell@*.recordio',
  proto = 'global_proto_db.tensorflow.Example'
) AS
SELECT AS VALUE
  $EXAMPLE(([
    $BYTES_FEATURE('#source', ([t.`PaperId`])),
    $BYTES_FEATURE('#target', ([CAST(CONCAT(REPLACE(REPLACE(t.`cell_id`, '\\', '\\\\'), '/', '\\/'), '/', CAST(UNIX_MICROS(t.`snapshot`) AS STRING)) AS BYTES)])),
    $FLOAT_FEATURE('weight', ([IFNULL(t.`weight`, 0.0)]))
  ]))
FROM /span/nonprod/my-team:my_database.`Links` AS t
WHERE t.`PaperId` IS NOT NULL AND t.`cell_id` IS NOT NULL AND t.`snapshot` IS NOT NULL AND t.`snapshot` = TIMESTAMP_MICROS(@filter_snapshot);
