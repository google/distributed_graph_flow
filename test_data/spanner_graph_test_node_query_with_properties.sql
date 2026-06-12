SELECT id as id, TO_JSON_STRING(JSON_OBJECT('properties', JSON_OBJECT('id', id, 'feat', feat))) as graph_element FROM NodeTable
