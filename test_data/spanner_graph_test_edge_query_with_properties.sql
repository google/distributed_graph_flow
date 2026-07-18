SELECT 'dummy_id' as id, source_id as source_id, target_id as target_id, TO_JSON_STRING(JSON_OBJECT('properties', JSON_OBJECT('weight', weight))) as graph_element FROM EdgeTable
